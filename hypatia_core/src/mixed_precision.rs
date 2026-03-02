//! Mixed Precision Compute: FP16/BF16 storage with FP32 accumulation
//!
//! For inference, storing weights in half precision (FP16/BF16) provides:
//! - 2x memory reduction: FP16 weight = 2 bytes vs FP32 = 4 bytes
//! - 2x bandwidth savings: less data to load from memory
//! - Faster GEMM on hardware with native FP16/BF16 support
//!
//! Strategy: "store narrow, compute wide"
//! - Weights stored as FP16 or BF16 (2 bytes each)
//! - Computation uses FP32 accumulation for numerical stability
//! - Output in FP32 (or cast back to half if chaining layers)
//!
//! FP16 (IEEE 754 half): 1 sign + 5 exponent + 10 mantissa
//!   Range: ±65504, precision: ~3.3 decimal digits
//!   Best for: inference, activations, gradients (with loss scaling)
//!
//! BF16 (Brain Float 16): 1 sign + 8 exponent + 7 mantissa
//!   Range: same as FP32 (±3.4e38), precision: ~2.4 decimal digits
//!   Best for: training (no loss scaling needed), large value ranges

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Half-Precision Types
// ============================================================================

/// Precision format for half-precision storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HalfPrecision {
    /// IEEE 754 FP16: 1+5+10 bits, range ±65504
    FP16,
    /// BFloat16: 1+8+7 bits, same range as FP32
    BF16,
}

/// Packed half-precision weight matrix.
/// Weights stored as u16 (raw bits of FP16 or BF16), computation in FP32.
#[derive(Clone, Debug)]
pub struct HalfWeights {
    /// Raw u16 half-precision values (row-major [rows, cols])
    pub data: Vec<u16>,
    /// Number of rows (out_features)
    pub rows: usize,
    /// Number of columns (in_features)
    pub cols: usize,
    /// Precision format
    pub format: HalfPrecision,
}

impl HalfWeights {
    /// Memory size in bytes (half-precision).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 2 // 2 bytes per u16
    }

    /// Dense FP32 memory size for comparison.
    pub fn fp32_memory_bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// Compression ratio vs FP32 (always ~2.0).
    pub fn compression_ratio(&self) -> f32 {
        2.0
    }
}

// ============================================================================
// FP16 Conversion (IEEE 754 Half-Precision)
// ============================================================================

/// Convert f32 to FP16 (u16 raw bits).
/// Handles overflow (clamp to ±65504), underflow (flush to zero), NaN, Inf.
#[inline]
pub fn f32_to_fp16(val: f32) -> u16 {
    // Use F16C instruction if available (x86_64)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return unsafe { f32_to_fp16_f16c(val) };
        }
    }
    f32_to_fp16_soft(val)
}

/// Convert FP16 (u16 raw bits) to f32.
#[inline]
pub fn fp16_to_f32(bits: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return unsafe { fp16_to_f32_f16c(bits) };
        }
    }
    fp16_to_f32_soft(bits)
}

/// Software FP16→f32 conversion.
fn fp16_to_f32_soft(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: 2^(-14) * (mant/1024)
        let val = (mant as f32) / 1024.0 * (2.0f32).powi(-14);
        return if sign == 1 { -val } else { val };
    }

    if exp == 31 {
        if mant == 0 {
            // Infinity
            return f32::from_bits((sign << 31) | 0x7F800000);
        }
        // NaN
        return f32::from_bits((sign << 31) | 0x7FC00000);
    }

    // Normal: (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_mant = mant << 13; // 10-bit → 23-bit mantissa
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
}

/// Software f32→FP16 conversion with rounding.
fn f32_to_fp16_soft(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // Handle special cases
    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            return ((sign << 15) | 0x7C00) as u16; // Inf
        }
        return ((sign << 15) | 0x7E00) as u16; // NaN
    }

    let unbiased_exp = exp - 127;

    if unbiased_exp > 15 {
        // Overflow → clamp to max FP16
        return ((sign << 15) | 0x7BFF) as u16; // ±65504
    }

    if unbiased_exp < -24 {
        // Too small → zero
        return (sign << 15) as u16;
    }

    if unbiased_exp < -14 {
        // Subnormal FP16
        let shift = -14 - unbiased_exp;
        let mant16 = ((0x800000 | mant) >> (shift + 13)) as u16;
        return ((sign << 15) as u16) | mant16;
    }

    // Normal FP16
    let fp16_exp = (unbiased_exp + 15) as u32;
    let fp16_mant = mant >> 13; // Round to 10-bit mantissa
    ((sign << 15) | (fp16_exp << 10) | fp16_mant) as u16
}

/// Hardware FP16→f32 using F16C instruction set.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn fp16_to_f32_f16c(bits: u16) -> f32 {
    let input = _mm_set1_epi16(bits as i16);
    let result = _mm_cvtph_ps(input);
    _mm_cvtss_f32(result)
}

/// Hardware f32→FP16 using F16C instruction set.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn f32_to_fp16_f16c(val: f32) -> u16 {
    let input = _mm_set_ss(val);
    let result = _mm_cvtps_ph(input, 0); // 0 = round to nearest even
    _mm_extract_epi16(result, 0) as u16
}

// ============================================================================
// BF16 Conversion (Brain Float 16)
// ============================================================================

/// Convert f32 to BF16 (u16 raw bits).
/// Simply truncates lower 16 bits of f32 mantissa.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even: add 0x7FFF + ((bits >> 16) & 1)
    let rounding = 0x7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding);
    (rounded >> 16) as u16
}

/// Convert BF16 (u16 raw bits) to f32.
/// Simply shifts left by 16 bits (fills lower mantissa with zeros).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// FP32 → Half Conversion (Bulk)
// ============================================================================

/// Convert a dense f32 weight matrix to half-precision.
pub fn to_half_weights(
    weights: &[f32],
    rows: usize,
    cols: usize,
    format: HalfPrecision,
) -> HalfWeights {
    debug_assert_eq!(weights.len(), rows * cols);

    let convert_fn: fn(f32) -> u16 = match format {
        HalfPrecision::FP16 => f32_to_fp16,
        HalfPrecision::BF16 => f32_to_bf16,
    };

    let data: Vec<u16> = weights.iter().map(|&v| convert_fn(v)).collect();

    HalfWeights {
        data,
        rows,
        cols,
        format,
    }
}

/// Convert half-precision weights back to f32 (for validation).
pub fn to_f32_weights(half: &HalfWeights) -> Vec<f32> {
    let convert_fn: fn(u16) -> f32 = match half.format {
        HalfPrecision::FP16 => fp16_to_f32,
        HalfPrecision::BF16 => bf16_to_f32,
    };

    half.data.iter().map(|&v| convert_fn(v)).collect()
}

// ============================================================================
// Mixed-Precision GEMM: half weights, FP32 compute
// ============================================================================

/// Fused mixed-precision linear: output = activation(input_f32 @ half_weight.T + bias_f32)
///
/// Strategy: dequantize weight row → FP32 dot product → accumulate
/// This avoids materializing the full FP32 weight matrix.
///
/// # Arguments
/// - `input`: [batch, in_feat] f32 row-major
/// - `half_weight`: HalfWeights [out_feat, in_feat]
/// - `bias`: optional [out_feat] f32
/// - `batch`: number of batch rows
/// - `relu`: apply ReLU activation
pub fn mixed_precision_linear(
    input: &[f32],
    half_weight: &HalfWeights,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    let in_feat = half_weight.cols;
    let out_feat = half_weight.rows;
    debug_assert_eq!(input.len(), batch * in_feat);

    let convert_fn: fn(u16) -> f32 = match half_weight.format {
        HalfPrecision::FP16 => fp16_to_f32,
        HalfPrecision::BF16 => bf16_to_f32,
    };

    let mut output = vec![0.0f32; batch * out_feat];

    for b in 0..batch {
        let input_row = &input[b * in_feat..(b + 1) * in_feat];
        let output_row = &mut output[b * out_feat..(b + 1) * out_feat];

        for i in 0..out_feat {
            let weight_row = &half_weight.data[i * in_feat..(i + 1) * in_feat];
            let mut acc = 0.0f32;

            for j in 0..in_feat {
                acc += convert_fn(weight_row[j]) * input_row[j];
            }

            if let Some(bias) = bias {
                acc += bias[i];
            }

            if relu {
                acc = acc.max(0.0);
            }

            output_row[i] = acc;
        }
    }

    output
}

/// Parallel mixed-precision linear using Rayon.
pub fn mixed_precision_linear_parallel(
    input: &[f32],
    half_weight: &HalfWeights,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    let in_feat = half_weight.cols;
    let out_feat = half_weight.rows;
    debug_assert_eq!(input.len(), batch * in_feat);

    let convert_fn: fn(u16) -> f32 = match half_weight.format {
        HalfPrecision::FP16 => fp16_to_f32,
        HalfPrecision::BF16 => bf16_to_f32,
    };

    let results: Vec<Vec<f32>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let input_row = &input[b * in_feat..(b + 1) * in_feat];
            let mut row_output = vec![0.0f32; out_feat];

            for i in 0..out_feat {
                let weight_row = &half_weight.data[i * in_feat..(i + 1) * in_feat];
                let mut acc = 0.0f32;

                for j in 0..in_feat {
                    acc += convert_fn(weight_row[j]) * input_row[j];
                }

                if let Some(bias) = bias {
                    acc += bias[i];
                }

                if relu {
                    acc = acc.max(0.0);
                }

                row_output[i] = acc;
            }

            row_output
        })
        .collect();

    let mut output = Vec::with_capacity(batch * out_feat);
    for row in results {
        output.extend_from_slice(&row);
    }
    output
}

// ============================================================================
// Precision Statistics
// ============================================================================

/// Analyze precision loss from FP32→half conversion.
pub fn precision_stats(weights: &[f32], format: HalfPrecision) -> PrecisionStats {
    let (to_half, from_half): (fn(f32) -> u16, fn(u16) -> f32) = match format {
        HalfPrecision::FP16 => (f32_to_fp16, fp16_to_f32),
        HalfPrecision::BF16 => (f32_to_bf16, bf16_to_f32),
    };

    let mut max_abs_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    let mut overflow_count = 0usize;
    let mut underflow_count = 0usize;

    for &val in weights {
        let half_bits = to_half(val);
        let recovered = from_half(half_bits);
        let error = (val - recovered).abs();

        if error > max_abs_error {
            max_abs_error = error;
        }
        sum_sq_error += (error as f64) * (error as f64);

        // FP16 overflow check
        if format == HalfPrecision::FP16 && val.abs() > 65504.0 {
            overflow_count += 1;
        }

        // Underflow check (value too small to represent)
        if val != 0.0 && recovered == 0.0 {
            underflow_count += 1;
        }
    }

    let n = weights.len() as f64;
    let rmse = (sum_sq_error / n.max(1.0)).sqrt() as f32;

    PrecisionStats {
        total_elements: weights.len(),
        max_abs_error,
        rmse,
        overflow_count,
        underflow_count,
        format,
    }
}

/// Statistics about precision loss from half-precision conversion.
#[derive(Debug, Clone)]
pub struct PrecisionStats {
    pub total_elements: usize,
    pub max_abs_error: f32,
    pub rmse: f32,
    pub overflow_count: usize,
    pub underflow_count: usize,
    pub format: HalfPrecision,
}

// ============================================================================
// Multi-layer Mixed-Precision MLP
// ============================================================================

/// Mixed-precision MLP forward: multiple half-precision linear layers.
pub fn mixed_precision_mlp_forward(
    input: &[f32],
    batch: usize,
    in_features: usize,
    layers: &[(&HalfWeights, Option<&[f32]>, bool)],
) -> Vec<f32> {
    let mut current = input.to_vec();
    let mut current_feat = in_features;

    for (half_weight, bias, is_relu) in layers {
        debug_assert_eq!(half_weight.cols, current_feat);

        if batch >= 4 {
            current = mixed_precision_linear_parallel(&current, half_weight, *bias, batch, *is_relu);
        } else {
            current = mixed_precision_linear(&current, half_weight, *bias, batch, *is_relu);
        }
        current_feat = half_weight.rows;
    }

    current
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- FP16 conversion tests ---

    #[test]
    fn test_fp16_zero() {
        assert_eq!(f32_to_fp16(0.0), 0x0000);
        assert_eq!(fp16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_fp16_neg_zero() {
        let nz = f32_to_fp16(-0.0);
        assert_eq!(nz, 0x8000);
        assert_eq!(fp16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_fp16_one() {
        let one = f32_to_fp16(1.0);
        assert_eq!(one, 0x3C00);
        assert_eq!(fp16_to_f32(0x3C00), 1.0);
    }

    #[test]
    fn test_fp16_neg_one() {
        let neg_one = f32_to_fp16(-1.0);
        assert_eq!(neg_one, 0xBC00);
        assert_eq!(fp16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_fp16_roundtrip_normal() {
        let values = [0.5, -0.5, 2.0, -2.0, 100.0, 0.001];
        for &v in &values {
            let half = f32_to_fp16(v);
            let back = fp16_to_f32(half);
            let error = (v - back).abs();
            assert!(error < v.abs() * 0.01 + 1e-4,
                "FP16 roundtrip error for {}: {} (back={})", v, error, back);
        }
    }

    #[test]
    fn test_fp16_overflow() {
        // Values > 65504 should clamp
        let big = f32_to_fp16(100000.0);
        let back = fp16_to_f32(big);
        assert!(back <= 65504.0);
    }

    #[test]
    fn test_fp16_inf() {
        let inf = f32_to_fp16(f32::INFINITY);
        assert_eq!(inf, 0x7C00);
        assert!(fp16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn test_fp16_nan() {
        let nan = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(nan).is_nan());
    }

    // --- BF16 conversion tests ---

    #[test]
    fn test_bf16_zero() {
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_bf16_one() {
        let one = f32_to_bf16(1.0);
        assert_eq!(one, 0x3F80);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.5, -0.5, 2.0, -2.0, 100.0, 1e30, -1e30];
        for &v in &values {
            let half = f32_to_bf16(v);
            let back = bf16_to_f32(half);
            let error = (v - back).abs();
            assert!(error < v.abs() * 0.01 + 1e-6,
                "BF16 roundtrip error for {}: {} (back={})", v, error, back);
        }
    }

    #[test]
    fn test_bf16_large_values() {
        // BF16 has same range as FP32
        let large = f32_to_bf16(1e38);
        let back = bf16_to_f32(large);
        assert!(back > 9e37, "BF16 should handle large values: {}", back);
    }

    // --- Mixed-precision GEMM tests ---

    #[test]
    fn test_half_linear_identity_fp16() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let hw = to_half_weights(&weight, 2, 2, HalfPrecision::FP16);

        let input = vec![3.0, 7.0];
        let output = mixed_precision_linear(&input, &hw, None, 1, false);
        assert!((output[0] - 3.0).abs() < 1e-3);
        assert!((output[1] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_half_linear_identity_bf16() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let hw = to_half_weights(&weight, 2, 2, HalfPrecision::BF16);

        let input = vec![3.0, 7.0];
        let output = mixed_precision_linear(&input, &hw, None, 1, false);
        assert!((output[0] - 3.0).abs() < 1e-3);
        assert!((output[1] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_half_linear_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 2.0];
        let hw = to_half_weights(&weight, 2, 2, HalfPrecision::FP16);
        let bias = vec![10.0, 20.0];

        let input = vec![3.0, 4.0];
        let output = mixed_precision_linear(&input, &hw, Some(&bias), 1, false);
        assert!((output[0] - 13.0).abs() < 0.1);
        assert!((output[1] - 28.0).abs() < 0.1);
    }

    #[test]
    fn test_half_linear_relu() {
        let weight = vec![1.0, 0.0, 0.0, -1.0];
        let hw = to_half_weights(&weight, 2, 2, HalfPrecision::FP16);

        let input = vec![3.0, 5.0];
        let output = mixed_precision_linear(&input, &hw, None, 1, true);
        assert!((output[0] - 3.0).abs() < 0.1);
        assert_eq!(output[1], 0.0); // ReLU clamps negative
    }

    #[test]
    fn test_half_linear_batch() {
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let hw = to_half_weights(&weight, 2, 2, HalfPrecision::BF16);

        let input = vec![1.0, 0.0, 0.0, 1.0]; // batch=2
        let output = mixed_precision_linear(&input, &hw, None, 2, false);

        // batch 0: [1*1+0*2, 1*3+0*4] = [1, 3]
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!((output[1] - 3.0).abs() < 0.1);
        // batch 1: [0*1+1*2, 0*3+1*4] = [2, 4]
        assert!((output[2] - 2.0).abs() < 0.1);
        assert!((output[3] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_half_vs_fp32_accuracy() {
        // Verify mixed-precision result is close to FP32 result
        let weight = vec![
            0.5, -0.3, 0.7, 0.1,
            -0.2, 0.8, -0.4, 0.6,
            0.3, 0.1, -0.5, 0.9,
        ];
        let rows = 3;
        let cols = 4;

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3];

        // FP32 reference
        let mut fp32_out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut acc = bias[i];
            for j in 0..cols {
                acc += weight[i * cols + j] * input[j];
            }
            fp32_out[i] = acc;
        }

        // FP16 result
        let hw16 = to_half_weights(&weight, rows, cols, HalfPrecision::FP16);
        let fp16_out = mixed_precision_linear(&input, &hw16, Some(&bias), 1, false);

        // BF16 result
        let hw_bf = to_half_weights(&weight, rows, cols, HalfPrecision::BF16);
        let bf16_out = mixed_precision_linear(&input, &hw_bf, Some(&bias), 1, false);

        for i in 0..rows {
            assert!((fp16_out[i] - fp32_out[i]).abs() < 0.05,
                "FP16 output[{}]: {} vs FP32: {}", i, fp16_out[i], fp32_out[i]);
            assert!((bf16_out[i] - fp32_out[i]).abs() < 0.05,
                "BF16 output[{}]: {} vs FP32: {}", i, bf16_out[i], fp32_out[i]);
        }
    }

    #[test]
    fn test_parallel_matches_serial() {
        let weight: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let hw = to_half_weights(&weight, 3, 4, HalfPrecision::FP16);
        let bias = vec![0.1, 0.2, 0.3];
        let input = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // batch=2

        let serial = mixed_precision_linear(&input, &hw, Some(&bias), 2, true);
        let parallel = mixed_precision_linear_parallel(&input, &hw, Some(&bias), 2, true);

        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert!((s - p).abs() < 1e-6, "serial={} parallel={}", s, p);
        }
    }

    #[test]
    fn test_precision_stats_fp16() {
        let weights = vec![0.1, -0.5, 1.0, 100.0, 0.001, -0.001];
        let stats = precision_stats(&weights, HalfPrecision::FP16);

        assert_eq!(stats.total_elements, 6);
        assert!(stats.rmse < 0.1); // FP16 should be quite accurate for these values
        assert_eq!(stats.overflow_count, 0);
    }

    #[test]
    fn test_precision_stats_bf16() {
        let weights = vec![0.1, -0.5, 1.0, 1e30, 0.001, -1e30];
        let stats = precision_stats(&weights, HalfPrecision::BF16);

        assert_eq!(stats.total_elements, 6);
        assert_eq!(stats.overflow_count, 0); // BF16 handles large values
    }

    #[test]
    fn test_to_f32_roundtrip() {
        let weights = vec![1.0, -1.0, 0.5, -0.5, 0.0, 100.0];
        let hw = to_half_weights(&weights, 2, 3, HalfPrecision::FP16);
        let recovered = to_f32_weights(&hw);

        for (orig, rec) in weights.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.1,
                "Roundtrip error: {} -> {}", orig, rec);
        }
    }

    #[test]
    fn test_memory_savings() {
        let hw = HalfWeights {
            data: vec![0u16; 1000],
            rows: 10,
            cols: 100,
            format: HalfPrecision::FP16,
        };

        assert_eq!(hw.memory_bytes(), 2000);
        assert_eq!(hw.fp32_memory_bytes(), 4000);
        assert!((hw.compression_ratio() - 2.0).abs() < 0.01);
    }
}
