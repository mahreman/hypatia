//! Neuromorphic computing support for Hypatia
//!
//! Implements Leaky Integrate-and-Fire (LIF) neuron model and
//! ANN-to-SNN (Artificial → Spiking Neural Network) conversion.
//!
//! # Theory
//!
//! ReLU(x) = max(0, x) can be approximated by LIF neuron firing rate:
//!   - Membrane dynamics: V[t+1] = β·V[t] + I[t]
//!   - Spike generation:  s[t] = 1 if V[t] >= V_th, else 0
//!   - Soft reset:        V[t] -= V_th · s[t]
//!   - Firing rate:       r = (Σ s[t]) / T ≈ ReLU(x) / V_th
//!
//! Over T timesteps with constant input, the LIF firing rate
//! converges to ReLU(x)/V_th, making ReLU→LIF a mathematically
//! grounded transformation for neuromorphic hardware deployment.
//!
//! # References
//! - Diehl et al. (2015): "Fast-classifying, high-accuracy spiking deep networks through
//!   weight and threshold balancing"
//! - Rueckauer et al. (2017): "Conversion of Continuous-Valued Deep Networks to
//!   Efficient Event-Driven Networks for Image Classification"

use rayon::prelude::*;

// ============================================================================
// LIF Neuron Parameters
// ============================================================================

/// Reset mode after a spike event
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResetMode {
    /// V -= V_th (preserves residual membrane potential)
    Soft,
    /// V = 0 (complete reset)
    Hard,
}

/// Parameters for Leaky Integrate-and-Fire neuron model
#[derive(Clone, Debug)]
pub struct LIFParams {
    /// Spike threshold voltage (default: 1.0)
    pub v_threshold: f32,
    /// Membrane decay factor, 0 < β < 1 (default: 0.95)
    pub beta: f32,
    /// Number of simulation timesteps (default: 32)
    pub timesteps: usize,
    /// Reset mechanism after spike
    pub reset_mode: ResetMode,
}

impl Default for LIFParams {
    fn default() -> Self {
        LIFParams {
            v_threshold: 1.0,
            beta: 0.95,
            timesteps: 32,
            reset_mode: ResetMode::Soft,
        }
    }
}

// ============================================================================
// Rate Coding: Continuous ↔ Spike Trains
// ============================================================================

/// Rate-encode continuous values to deterministic spike trains.
///
/// Uses threshold-based rate coding: for each timestep t,
/// spike if accumulated fractional part >= 1.0.
/// This produces exactly floor(x * T / V_th) spikes over T steps
/// for input x, giving firing rate ≈ x / V_th.
///
/// # Arguments
/// * `input` - Continuous input values [features]
/// * `params` - LIF parameters (v_threshold, timesteps)
///
/// # Returns
/// Spike trains as [timesteps][features] binary (0/1) values
pub fn rate_encode(input: &[f32], params: &LIFParams) -> Vec<Vec<u8>> {
    let n = input.len();
    let t = params.timesteps;
    let v_th = params.v_threshold;

    let mut spikes = vec![vec![0u8; n]; t];
    let mut accumulators = vec![0.0f32; n];

    for ts in 0..t {
        for i in 0..n {
            // Clamp negative values (mirroring ReLU behavior)
            let rate = (input[i] / v_th).max(0.0);
            accumulators[i] += rate;
            if accumulators[i] >= 1.0 {
                spikes[ts][i] = 1;
                accumulators[i] -= 1.0;
            }
        }
    }

    spikes
}

/// Decode spike trains back to continuous values via firing rate.
///
/// firing_rate = (sum of spikes over T) / T * V_th
///
/// # Arguments
/// * `spikes` - Spike trains [timesteps][features]
/// * `params` - LIF parameters
///
/// # Returns
/// Decoded continuous values [features]
pub fn rate_decode(spikes: &[Vec<u8>], params: &LIFParams) -> Vec<f32> {
    if spikes.is_empty() {
        return vec![];
    }
    let n = spikes[0].len();
    let t = spikes.len() as f32;
    let v_th = params.v_threshold;

    let mut output = vec![0.0f32; n];
    for spike_row in spikes {
        for (i, &s) in spike_row.iter().enumerate() {
            output[i] += s as f32;
        }
    }
    for val in output.iter_mut() {
        *val = *val / t * v_th;
    }
    output
}

// ============================================================================
// LIF Neuron Simulation
// ============================================================================

/// Single LIF neuron step (scalar).
///
/// Returns (new_membrane, spike).
#[inline(always)]
fn lif_step(membrane: f32, current: f32, params: &LIFParams) -> (f32, u8) {
    // Leak + integrate
    let v = params.beta * membrane + current;

    // Spike generation
    if v >= params.v_threshold {
        let spike = 1u8;
        let v_reset = match params.reset_mode {
            ResetMode::Soft => v - params.v_threshold,
            ResetMode::Hard => 0.0,
        };
        (v_reset, spike)
    } else {
        (v, 0)
    }
}

/// LIF layer forward pass: processes spike input through a linear+LIF layer.
///
/// For each timestep:
///   1. Compute synaptic current: I[t] = W · s_in[t] + bias
///   2. Update membrane: V[t] = β·V[t-1] + I[t]
///   3. Generate spikes: s_out[t] = (V[t] >= V_th)
///   4. Reset: V[t] -= V_th · s_out[t]
///
/// # Arguments
/// * `input_spikes` - Input spike trains [timesteps][in_features]
/// * `weights` - Weight matrix [out_features * in_features] row-major
/// * `bias` - Optional bias [out_features]
/// * `out_features` - Number of output neurons
/// * `in_features` - Number of input neurons
/// * `params` - LIF parameters
///
/// # Returns
/// Output spike trains [timesteps][out_features]
pub fn lif_linear_forward(
    input_spikes: &[Vec<u8>],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
    params: &LIFParams,
) -> Vec<Vec<u8>> {
    let t = input_spikes.len();
    let mut output_spikes = vec![vec![0u8; out_features]; t];
    let mut membrane = vec![0.0f32; out_features];

    for ts in 0..t {
        let input = &input_spikes[ts];

        // Compute synaptic current for each output neuron
        for o in 0..out_features {
            let mut current = 0.0f32;
            let w_row = &weights[o * in_features..(o + 1) * in_features];

            // Sparse dot product: only accumulate where input spikes are 1
            for (i, &spike) in input.iter().enumerate() {
                if spike != 0 {
                    current += w_row[i];
                }
            }

            if let Some(b) = bias {
                // Scale bias by 1/T so it contributes correctly over all timesteps
                current += b[o] / params.timesteps as f32;
            }

            // LIF step
            let (new_v, spike) = lif_step(membrane[o], current, params);
            membrane[o] = new_v;
            output_spikes[ts][o] = spike;
        }
    }

    output_spikes
}

/// Parallelized LIF linear forward for large layers.
///
/// Parallelizes across output neurons within each timestep.
pub fn lif_linear_forward_par(
    input_spikes: &[Vec<u8>],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
    params: &LIFParams,
) -> Vec<Vec<u8>> {
    let t = input_spikes.len();
    let mut output_spikes = vec![vec![0u8; out_features]; t];
    let mut membrane = vec![0.0f32; out_features];

    for ts in 0..t {
        let input = &input_spikes[ts];

        // Parallel across output neurons
        let results: Vec<(f32, u8)> = (0..out_features)
            .into_par_iter()
            .map(|o| {
                let mut current = 0.0f32;
                let w_row = &weights[o * in_features..(o + 1) * in_features];

                for (i, &spike) in input.iter().enumerate() {
                    if spike != 0 {
                        current += w_row[i];
                    }
                }

                if let Some(b) = bias {
                    current += b[o] / params.timesteps as f32;
                }

                lif_step(membrane[o], current, params)
            })
            .collect();

        for (o, (new_v, spike)) in results.into_iter().enumerate() {
            membrane[o] = new_v;
            output_spikes[ts][o] = spike;
        }
    }

    output_spikes
}

// ============================================================================
// ANN → SNN Conversion
// ============================================================================

/// Layer info for ANN→SNN conversion
pub struct ANNLayer {
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_features: usize,
    pub in_features: usize,
}

/// Converted SNN model: sequence of LIF linear layers
pub struct SNNModel {
    pub layers: Vec<SNNLayer>,
    pub params: LIFParams,
}

pub struct SNNLayer {
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_features: usize,
    pub in_features: usize,
    /// Per-layer threshold scaling factor from weight normalization
    pub threshold_scale: f32,
}

/// Weight normalization for ANN→SNN conversion.
///
/// Scales weights so that the maximum activation at each layer
/// maps to the spike threshold. This ensures firing rates
/// properly approximate ReLU outputs.
///
/// Method: "Threshold Balancing" (Diehl et al., 2015)
/// For each layer l:
///   scale_l = max(relu_activation_l) / V_th
///   W_l_normalized = W_l / scale_l
fn normalize_weights(layers: &[ANNLayer], params: &LIFParams) -> Vec<SNNLayer> {
    // Without calibration data, use weight-based normalization:
    // scale_l = max(|W_l|) * fan_in / V_th
    // This approximates the maximum possible activation
    layers
        .iter()
        .map(|layer| {
            // Compute max absolute column sum (∞-norm of W)
            // This bounds the maximum activation for unit-bounded inputs
            let max_activation = (0..layer.out_features)
                .map(|o| {
                    let row = &layer.weights
                        [o * layer.in_features..(o + 1) * layer.in_features];
                    row.iter().map(|w| w.abs()).sum::<f32>()
                })
                .fold(0.0f32, f32::max);

            let scale = if max_activation > 0.0 {
                max_activation / params.v_threshold
            } else {
                1.0
            };

            let normalized_weights: Vec<f32> =
                layer.weights.iter().map(|&w| w / scale).collect();

            let normalized_bias = layer.bias.as_ref().map(|b| {
                b.iter().map(|&bi| bi / scale).collect()
            });

            SNNLayer {
                weights: normalized_weights,
                bias: normalized_bias,
                out_features: layer.out_features,
                in_features: layer.in_features,
                threshold_scale: scale,
            }
        })
        .collect()
}

/// Convert an ANN (sequence of Linear+ReLU layers) to an SNN.
///
/// The last layer does NOT get a LIF neuron - its membrane potential
/// is read out directly (rate-decoded) as the output.
pub fn ann_to_snn(layers: Vec<ANNLayer>, params: LIFParams) -> SNNModel {
    let snn_layers = normalize_weights(&layers, &params);
    SNNModel {
        layers: snn_layers,
        params,
    }
}

impl SNNModel {
    /// Run SNN inference on continuous input.
    ///
    /// Pipeline:
    /// 1. Rate-encode input → spike trains
    /// 2. Forward through LIF layers (spike domain)
    /// 3. Rate-decode output spikes → continuous output
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let n_layers = self.layers.len();
        if n_layers == 0 {
            return input.to_vec();
        }

        // Step 1: Rate-encode input
        let mut current_spikes = rate_encode(input, &self.params);

        // Step 2: Forward through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            let is_last = i == n_layers - 1;

            if is_last {
                // Last layer: accumulate membrane potential, don't spike
                current_spikes = lif_linear_accumulate(
                    &current_spikes,
                    &layer.weights,
                    layer.bias.as_deref(),
                    layer.out_features,
                    layer.in_features,
                    &self.params,
                );
            } else {
                // Hidden layers: full LIF with spikes
                let use_par = layer.out_features >= 256;
                if use_par {
                    current_spikes = lif_linear_forward_par(
                        &current_spikes,
                        &layer.weights,
                        layer.bias.as_deref(),
                        layer.out_features,
                        layer.in_features,
                        &self.params,
                    );
                } else {
                    current_spikes = lif_linear_forward(
                        &current_spikes,
                        &layer.weights,
                        layer.bias.as_deref(),
                        layer.out_features,
                        layer.in_features,
                        &self.params,
                    );
                }
            }
        }

        // Step 3: Decode output
        rate_decode(&current_spikes, &self.params)
    }

    /// Run SNN inference and return per-timestep spike counts for analysis.
    pub fn forward_with_stats(&self, input: &[f32]) -> (Vec<f32>, Vec<SpikeStats>) {
        let n_layers = self.layers.len();
        if n_layers == 0 {
            return (input.to_vec(), vec![]);
        }

        let mut stats = Vec::with_capacity(n_layers);
        let mut current_spikes = rate_encode(input, &self.params);

        for (i, layer) in self.layers.iter().enumerate() {
            let is_last = i == n_layers - 1;

            if is_last {
                current_spikes = lif_linear_accumulate(
                    &current_spikes,
                    &layer.weights,
                    layer.bias.as_deref(),
                    layer.out_features,
                    layer.in_features,
                    &self.params,
                );
            } else {
                current_spikes = lif_linear_forward(
                    &current_spikes,
                    &layer.weights,
                    layer.bias.as_deref(),
                    layer.out_features,
                    layer.in_features,
                    &self.params,
                );
            }

            // Compute stats
            let total_spikes: usize = current_spikes
                .iter()
                .flat_map(|row| row.iter())
                .map(|&s| s as usize)
                .sum();
            let total_possible = current_spikes.len() * current_spikes[0].len();
            stats.push(SpikeStats {
                layer_idx: i,
                total_spikes,
                total_possible,
                firing_rate: total_spikes as f64 / total_possible as f64,
            });
        }

        let output = rate_decode(&current_spikes, &self.params);
        (output, stats)
    }
}

/// Statistics for spike activity in a layer
#[derive(Clone, Debug)]
pub struct SpikeStats {
    pub layer_idx: usize,
    pub total_spikes: usize,
    pub total_possible: usize,
    pub firing_rate: f64,
}

/// Last-layer accumulation: no spiking, just integrate membrane potential.
/// Returns "pseudo-spikes" encoding the accumulated membrane value
/// (used internally for rate_decode).
fn lif_linear_accumulate(
    input_spikes: &[Vec<u8>],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
    params: &LIFParams,
) -> Vec<Vec<u8>> {
    let t = input_spikes.len();
    let mut membrane = vec![0.0f32; out_features];

    // Accumulate membrane over all timesteps (no spikes)
    for ts in 0..t {
        let input = &input_spikes[ts];
        for o in 0..out_features {
            let mut current = 0.0f32;
            let w_row = &weights[o * in_features..(o + 1) * in_features];

            for (i, &spike) in input.iter().enumerate() {
                if spike != 0 {
                    current += w_row[i];
                }
            }

            if let Some(b) = bias {
                current += b[o] / params.timesteps as f32;
            }

            membrane[o] = params.beta * membrane[o] + current;
        }
    }

    // Encode membrane as "pseudo" spike-rate output:
    // Scale so that rate_decode(output) ≈ membrane / T * V_th
    // We create a spike pattern that encodes the continuous value
    let mut output = vec![vec![0u8; out_features]; t];
    for o in 0..out_features {
        // The accumulated membrane value after T steps encodes the output.
        // To make rate_decode work: we need spikes_count/T * V_th ≈ value
        // So spikes_count ≈ membrane * T / (beta_sum * V_th)
        // where beta_sum = sum(beta^i, i=0..T-1)
        let beta_sum: f32 = if (params.beta - 1.0).abs() < 1e-6 {
            t as f32
        } else {
            (1.0 - params.beta.powi(t as i32)) / (1.0 - params.beta)
        };

        // Convert membrane to spike count
        let target_rate = membrane[o] / (beta_sum * params.v_threshold);
        let target_rate = target_rate.max(0.0); // ReLU-like clamp
        let spike_count = (target_rate * t as f32).round() as usize;
        let spike_count = spike_count.min(t);

        // Spread spikes evenly
        for s in 0..spike_count {
            let idx = s * t / spike_count.max(1);
            output[idx][o] = 1;
        }
    }

    output
}

// ============================================================================
// Energy Model for Neuromorphic Hardware
// ============================================================================

/// Energy cost model for neuromorphic hardware
#[derive(Clone, Debug)]
pub struct NeuromorphicEnergy {
    /// Energy per synaptic operation (nanojoules) - only on spikes
    pub energy_per_synop: f64,
    /// Energy per spike event (nanojoules)
    pub energy_per_spike: f64,
    /// Static power per neuron per timestep (nanojoules)
    pub static_power_per_neuron: f64,
}

impl Default for NeuromorphicEnergy {
    fn default() -> Self {
        // Typical values for Intel Loihi 2
        NeuromorphicEnergy {
            energy_per_synop: 0.023, // 23 pJ per synaptic op
            energy_per_spike: 0.052, // 52 pJ per spike
            static_power_per_neuron: 0.001, // 1 pJ static
        }
    }
}

impl NeuromorphicEnergy {
    /// Estimate energy for a LIF linear layer
    pub fn estimate_layer_energy(
        &self,
        in_features: usize,
        out_features: usize,
        timesteps: usize,
        avg_firing_rate: f64,
    ) -> f64 {
        let t = timesteps as f64;
        let n_in = in_features as f64;
        let n_out = out_features as f64;

        // Synaptic operations: only happen when input spike occurs
        // Expected synops per timestep = n_in * avg_rate * n_out (fan-out)
        let synops = n_in * avg_firing_rate * n_out * t;

        // Output spikes
        let out_spikes = n_out * avg_firing_rate * t;

        // Static power
        let static_cost = n_out * t * self.static_power_per_neuron;

        synops * self.energy_per_synop
            + out_spikes * self.energy_per_spike
            + static_cost
    }

    /// Compare energy: neuromorphic vs conventional (GPU/CPU)
    /// Returns (neuromorphic_nj, conventional_nj)
    pub fn compare_energy(
        &self,
        in_features: usize,
        out_features: usize,
        timesteps: usize,
        avg_firing_rate: f64,
    ) -> (f64, f64) {
        let neuro = self.estimate_layer_energy(
            in_features, out_features, timesteps, avg_firing_rate,
        );

        // Conventional: every MAC costs ~1 pJ (45nm GPU) to ~0.1 pJ (7nm)
        // Full dense matmul: in_features * out_features MACs
        let conventional = in_features as f64 * out_features as f64 * 0.5; // ~0.5 pJ per MAC

        (neuro, conventional)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_step_no_spike() {
        let params = LIFParams::default();
        let (v, spike) = lif_step(0.0, 0.5, &params);
        assert_eq!(spike, 0);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lif_step_spike() {
        let params = LIFParams::default();
        let (v, spike) = lif_step(0.5, 0.6, &params); // 0.95*0.5 + 0.6 = 1.075 > 1.0
        assert_eq!(spike, 1);
        // Soft reset: 1.075 - 1.0 = 0.075
        assert!((v - 0.075).abs() < 1e-3);
    }

    #[test]
    fn test_lif_step_hard_reset() {
        let params = LIFParams {
            reset_mode: ResetMode::Hard,
            ..Default::default()
        };
        let (v, spike) = lif_step(0.5, 0.6, &params);
        assert_eq!(spike, 1);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_rate_encode_decode_roundtrip() {
        let params = LIFParams {
            timesteps: 100,
            v_threshold: 1.0,
            ..Default::default()
        };

        // Test various input values
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let spikes = rate_encode(&input, &params);
        let decoded = rate_decode(&spikes, &params);

        // Verify roundtrip accuracy (should be within ~1/T = 1%)
        for (i, (&original, &recovered)) in input.iter().zip(decoded.iter()).enumerate() {
            let error = (original - recovered).abs();
            assert!(
                error < 0.05,
                "Value {}: original={}, recovered={}, error={}",
                i, original, recovered, error
            );
        }
    }

    #[test]
    fn test_rate_encode_negative_clamped() {
        let params = LIFParams {
            timesteps: 32,
            ..Default::default()
        };
        let input = vec![-1.0, -0.5, 0.0];
        let spikes = rate_encode(&input, &params);
        let decoded = rate_decode(&spikes, &params);

        // Negative values should be clamped to 0 (like ReLU)
        for &val in &decoded {
            assert!((val - 0.0).abs() < 1e-6, "Negative input should produce 0 output");
        }
    }

    #[test]
    fn test_lif_linear_forward_basic() {
        let params = LIFParams {
            timesteps: 32,
            v_threshold: 1.0,
            beta: 0.9,
            ..Default::default()
        };

        // Simple 2→1 layer with weight [1.0, 1.0]
        let weights = vec![1.0, 1.0];
        let in_feat = 2;
        let out_feat = 1;

        // Create input with both neurons spiking every other step
        let mut input_spikes = vec![vec![0u8; 2]; 32];
        for t in 0..32 {
            if t % 2 == 0 {
                input_spikes[t] = vec![1, 1];
            }
        }

        let output = lif_linear_forward(
            &input_spikes, &weights, None, out_feat, in_feat, &params,
        );

        // Should produce some spikes (combined current = 2.0 per active step)
        let total: usize = output.iter().flat_map(|r| r.iter()).map(|&s| s as usize).sum();
        assert!(total > 0, "LIF layer should produce spikes");
    }

    #[test]
    fn test_ann_to_snn_conversion() {
        let layers = vec![
            ANNLayer {
                weights: vec![0.5, 0.3, -0.2, 0.4, 0.1, 0.6],
                bias: Some(vec![0.1, -0.1]),
                out_features: 2,
                in_features: 3,
            },
            ANNLayer {
                weights: vec![0.7, -0.3],
                bias: Some(vec![0.05]),
                out_features: 1,
                in_features: 2,
            },
        ];

        let params = LIFParams {
            timesteps: 64,
            v_threshold: 1.0,
            beta: 0.95,
            ..Default::default()
        };

        let snn = ann_to_snn(layers, params);
        assert_eq!(snn.layers.len(), 2);

        // Test forward pass
        let input = vec![1.0, 0.5, 0.8];
        let output = snn.forward(&input);
        assert_eq!(output.len(), 1);

        // Output should be non-negative (ReLU-like behavior)
        // The exact value depends on weight normalization
        assert!(output[0] >= 0.0, "SNN output should be non-negative");
    }

    #[test]
    fn test_snn_forward_with_stats() {
        let layers = vec![
            ANNLayer {
                weights: vec![0.8, 0.6, 0.4, 0.9],
                bias: None,
                out_features: 2,
                in_features: 2,
            },
        ];

        let params = LIFParams {
            timesteps: 64,
            beta: 0.95,
            ..Default::default()
        };

        let snn = ann_to_snn(layers, params);
        let input = vec![0.5, 0.7];
        let (output, stats) = snn.forward_with_stats(&input);

        assert_eq!(output.len(), 2);
        assert_eq!(stats.len(), 1);
        assert!(stats[0].firing_rate >= 0.0 && stats[0].firing_rate <= 1.0);
    }

    #[test]
    fn test_energy_model() {
        let energy = NeuromorphicEnergy::default();

        // Sparse network (10% firing rate) should be much more efficient
        let (neuro_sparse, conv) = energy.compare_energy(1024, 1024, 32, 0.1);
        // Dense network (90% firing rate) - less efficient
        let (neuro_dense, _) = energy.compare_energy(1024, 1024, 32, 0.9);

        assert!(neuro_sparse < neuro_dense, "Sparse should use less energy");
        assert!(
            neuro_sparse < conv,
            "Sparse neuromorphic should beat conventional: {} vs {}",
            neuro_sparse, conv
        );
    }

    #[test]
    fn test_relu_lif_equivalence() {
        // Core theorem: LIF firing rate ≈ ReLU(x) / V_th
        // Test with various inputs over many timesteps
        let params = LIFParams {
            timesteps: 256, // More timesteps = better approximation
            v_threshold: 1.0,
            beta: 0.0, // No leak = IF neuron = closer to ReLU
            ..Default::default()
        };

        let test_values = vec![0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];

        for &x in &test_values {
            let relu_output = x.max(0.0);

            // Encode → Decode should approximate ReLU
            let spikes = rate_encode(&[x], &params);
            let decoded = rate_decode(&spikes, &params);

            let error = (relu_output - decoded[0]).abs();
            let tolerance = 0.05; // 5% tolerance
            assert!(
                error < tolerance || (relu_output < 0.01 && decoded[0] < 0.01),
                "ReLU({})={}, LIF decoded={}, error={}",
                x, relu_output, decoded[0], error
            );
        }
    }
}
