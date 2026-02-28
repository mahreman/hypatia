//! Native fused tensor operations for Hypatia.
//!
//! Provides fused forward and backward passes that bypass PyTorch's
//! per-operator dispatch overhead. For small-to-medium models where
//! dispatch overhead dominates compute, this can give 2-5x speedup.
//!
//! GEMM backend (auto-detected at runtime):
//! - Intel MKL via PyTorch's libtorch_cpu.so (7x faster than OpenBLAS)
//! - OpenBLAS cblas_sgemm as fallback
//!
//! Key optimizations:
//! - Single Python->Rust crossing for entire MLP forward pass
//! - Fused bias + activation in single memory pass after GEMM
//! - Zero intermediate tensor allocations

use std::sync::OnceLock;

// ============================================================================
// GEMM Backend: MKL (via PyTorch) with OpenBLAS fallback
// ============================================================================

// Fortran sgemm_ function pointer type (MKL)
// sgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
type SgemmFortranFn = unsafe extern "C" fn(
    transa: *const u8, transb: *const u8,
    m: *const i32, n: *const i32, k: *const i32,
    alpha: *const f32, a: *const f32, lda: *const i32,
    b: *const f32, ldb: *const i32,
    beta: *const f32, c: *mut f32, ldc: *const i32,
);

static MKL_SGEMM: OnceLock<Option<SgemmFortranFn>> = OnceLock::new();

/// Try to find MKL's sgemm_ in the already-loaded PyTorch library.
/// Returns None if PyTorch/MKL is not loaded.
fn get_mkl_sgemm() -> Option<SgemmFortranFn> {
    *MKL_SGEMM.get_or_init(|| {
        unsafe {
            // PyTorch's libtorch_cpu.so is already loaded by Python.
            // We need to find it and get MKL's sgemm_ specifically.
            // Strategy: dlopen libtorch_cpu.so with RTLD_NOLOAD to get handle
            // to the already-loaded library, then dlsym for sgemm_.
            let lib_name = std::ffi::CString::new("libtorch_cpu.so").ok()?;
            let handle = libc::dlopen(
                lib_name.as_ptr(),
                libc::RTLD_NOLOAD | libc::RTLD_NOW,
            );
            if handle.is_null() {
                log::debug!("MKL not available: libtorch_cpu.so not loaded");
                return None;
            }
            let sym_name = std::ffi::CString::new("sgemm_").ok()?;
            let sym = libc::dlsym(handle, sym_name.as_ptr());
            if sym.is_null() {
                log::debug!("MKL not available: sgemm_ not found in libtorch_cpu.so");
                return None;
            }
            log::info!("MKL sgemm_ found in libtorch_cpu.so - using MKL for GEMM");
            Some(std::mem::transmute(sym))
        }
    })
}

// OpenBLAS FFI bindings (fallback)
extern "C" {
    fn cblas_sgemm(
        order: i32,   // CblasRowMajor = 101
        transa: i32,  // CblasNoTrans = 111, CblasTrans = 112
        transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// GEMM: C = A @ B^T
/// A: [m, k], B: [n, k] (transposed), C: [m, n]
/// Uses MKL if available (7x faster), falls back to OpenBLAS.
#[inline]
fn gemm_nt(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if let Some(mkl_sgemm) = get_mkl_sgemm() {
        // MKL Fortran interface for row-major C = A @ B^T:
        // C_cm[N,M] = B_cm^T[N,K] @ A_cm[K,M]
        // sgemm_('T', 'N', N, M, K, 1.0, B, K, A, K, 0.0, C, N)
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            mkl_sgemm(
                b"T" as *const u8, b"N" as *const u8,
                &n_i, &m_i, &k_i,
                &alpha, b.as_ptr(), &k_i,
                a.as_ptr(), &k_i,
                &beta, c.as_mut_ptr(), &n_i,
            );
        }
    } else {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), k as i32,
                0.0, c.as_mut_ptr(), n as i32,
            );
        }
    }
}

/// GEMM: C = A @ B (no transpose)
/// A: [m, k], B: [k, n], C: [m, n]
#[inline]
fn gemm_nn(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if let Some(mkl_sgemm) = get_mkl_sgemm() {
        // MKL Fortran for row-major C = A @ B:
        // C_cm[N,M] = B_cm[N,K] @ A_cm[K,M]
        // sgemm_('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N)
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            mkl_sgemm(
                b"N" as *const u8, b"N" as *const u8,
                &n_i, &m_i, &k_i,
                &alpha, b.as_ptr(), &n_i,
                a.as_ptr(), &k_i,
                &beta, c.as_mut_ptr(), &n_i,
            );
        }
    } else {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0, c.as_mut_ptr(), n as i32,
            );
        }
    }
}

/// GEMM: C = A.T @ B
/// A: [k, m] stored as [m, k] row-major (transposed via flag), B: [k, n], C: [m, n]
#[inline]
fn gemm_tn(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    lda: i32,
) {
    if let Some(mkl_sgemm) = get_mkl_sgemm() {
        // MKL Fortran for row-major C = A^T @ B:
        // C_cm[N,M] = B_cm[N,K] @ (A_cm^T)[K,M]
        // A row-major [k_orig, m] with lda → need trans on A_cm
        // sgemm_('N', 'T', N, M, K, 1.0, B, N, A, lda, 0.0, C, N)
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            mkl_sgemm(
                b"N" as *const u8, b"T" as *const u8,
                &n_i, &m_i, &k_i,
                &alpha, b.as_ptr(), &n_i,
                a.as_ptr(), &lda,
                &beta, c.as_mut_ptr(), &n_i,
            );
        }
    } else {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), lda,
                b.as_ptr(), n as i32,
                0.0, c.as_mut_ptr(), n as i32,
            );
        }
    }
}

/// Fused forward: output = activation(input @ weight.T + bias)
///
/// All arrays are f32 row-major:
/// - input:  [batch, in_feat]
/// - weight: [out_feat, in_feat]
/// - bias:   [out_feat] (optional)
/// - output: [batch, out_feat] (returned)
pub fn fused_linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    relu: bool,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), batch * in_feat);
    debug_assert_eq!(weight.len(), out_feat * in_feat);

    let mut output = vec![0.0f32; batch * out_feat];

    // GEMM via OpenBLAS: output = input @ weight.T
    gemm_nt(input, weight, &mut output, batch, in_feat, out_feat);

    // Fused bias + activation (single pass over output)
    match (bias, relu) {
        (Some(b), true) => {
            for i in 0..batch {
                let row = &mut output[i * out_feat..(i + 1) * out_feat];
                for j in 0..out_feat {
                    row[j] = (row[j] + b[j]).max(0.0);
                }
            }
        }
        (Some(b), false) => {
            for i in 0..batch {
                let row = &mut output[i * out_feat..(i + 1) * out_feat];
                for j in 0..out_feat {
                    row[j] += b[j];
                }
            }
        }
        (None, true) => {
            for val in output.iter_mut() {
                *val = val.max(0.0);
            }
        }
        (None, false) => {
            // No-op
        }
    }

    output
}

/// Multi-layer MLP forward pass. All layers fused in a single Rust call.
pub fn mlp_forward(
    input: &[f32],
    batch: usize,
    in_features: usize,
    layers: &[(&[f32], Option<&[f32]>, usize, bool)],
) -> Vec<f32> {
    let mut current = input.to_vec();
    let mut current_feat = in_features;

    for (weight, bias, out_feat, is_relu) in layers {
        current = fused_linear(&current, weight, *bias, batch, current_feat, *out_feat, *is_relu);
        current_feat = *out_feat;
    }

    current
}

/// Backward pass for a single fused linear + relu layer.
///
/// Returns (grad_input, grad_weight, grad_bias)
pub fn fused_linear_backward(
    grad_output: &[f32], // [batch, out_feat]
    input: &[f32],       // [batch, in_feat]
    weight: &[f32],      // [out_feat, in_feat]
    output: &[f32],      // [batch, out_feat] (post-activation, for relu mask)
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    relu: bool,
    needs_input_grad: bool,
) -> (Option<Vec<f32>>, Vec<f32>, Vec<f32>) {
    // 1. Activation backward
    let grad_pre = if relu {
        let mut gp = vec![0.0f32; batch * out_feat];
        for i in 0..batch * out_feat {
            gp[i] = if output[i] > 0.0 {
                grad_output[i]
            } else {
                0.0
            };
        }
        gp
    } else {
        grad_output.to_vec()
    };

    // 2. grad_input = grad_pre @ weight
    // [batch, out_feat] @ [out_feat, in_feat] = [batch, in_feat]
    let grad_input = if needs_input_grad {
        let mut gi = vec![0.0f32; batch * in_feat];
        gemm_nn(&grad_pre, weight, &mut gi, batch, out_feat, in_feat);
        Some(gi)
    } else {
        None
    };

    // 3. grad_weight = grad_pre.T @ input
    // [out_feat, batch] @ [batch, in_feat] = [out_feat, in_feat]
    let mut grad_weight = vec![0.0f32; out_feat * in_feat];
    // grad_pre is [batch, out_feat] row-major. We want grad_pre.T @ input.
    // Using cblas: Trans(A) @ B where A=[batch, out_feat], so A.T=[out_feat, batch]
    // m=out_feat, k=batch, n=in_feat, lda=out_feat (stride of A in memory)
    gemm_tn(&grad_pre, input, &mut grad_weight, out_feat, batch, in_feat, out_feat as i32);

    // 4. grad_bias = grad_pre.sum(dim=0)
    let mut grad_bias = vec![0.0f32; out_feat];
    for i in 0..batch {
        for j in 0..out_feat {
            grad_bias[j] += grad_pre[i * out_feat + j];
        }
    }

    (grad_input, grad_weight, grad_bias)
}

/// Complete training step: forward -> MSE loss -> backward -> SGD update
/// Returns the loss value.
///
/// Modifies weights and biases in-place.
pub fn train_step_sgd(
    input: &[f32],
    target: &[f32],
    batch: usize,
    in_features: usize,
    out_features_final: usize,
    weights: &mut [Vec<f32>],
    biases: &mut [Option<Vec<f32>>],
    layer_dims: &[(usize, usize, bool)], // (in_feat, out_feat, is_relu) per layer
    lr: f32,
) -> f32 {
    let num_layers = weights.len();

    // === FORWARD PASS (save intermediates) ===
    let mut intermediates: Vec<Vec<f32>> = Vec::with_capacity(num_layers + 1);
    intermediates.push(input.to_vec());

    let mut current_feat = in_features;
    for i in 0..num_layers {
        let (_, out_feat, is_relu) = layer_dims[i];
        let bias_slice = biases[i].as_deref();
        let output = fused_linear(
            &intermediates[i],
            &weights[i],
            bias_slice,
            batch,
            current_feat,
            out_feat,
            is_relu,
        );
        intermediates.push(output);
        current_feat = out_feat;
    }

    let final_output = &intermediates[num_layers];

    // === MSE LOSS ===
    let n = (batch * out_features_final) as f32;
    let mut loss = 0.0f32;
    let mut grad_loss = vec![0.0f32; batch * out_features_final];
    for i in 0..batch * out_features_final {
        let diff = final_output[i] - target[i];
        loss += diff * diff;
        grad_loss[i] = 2.0 * diff / n;
    }
    loss /= n;

    // === BACKWARD PASS ===
    let mut grad_output = grad_loss;

    for i in (0..num_layers).rev() {
        let (in_f, out_f, is_relu) = layer_dims[i];
        let needs_input_grad = i > 0;

        let (grad_input, grad_weight, grad_bias) = fused_linear_backward(
            &grad_output,
            &intermediates[i],
            &weights[i],
            &intermediates[i + 1],
            batch,
            in_f,
            out_f,
            is_relu,
            needs_input_grad,
        );

        // === SGD UPDATE ===
        for j in 0..weights[i].len() {
            weights[i][j] -= lr * grad_weight[j];
        }
        if let Some(ref mut bias) = biases[i] {
            for j in 0..bias.len() {
                bias[j] -= lr * grad_bias[j];
            }
        }

        if let Some(gi) = grad_input {
            grad_output = gi;
        }
    }

    loss
}

// ============================================================================
// LayerNorm
// ============================================================================

/// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// Two-pass: sum for mean, then sum-of-squares for variance.
/// Uses f32 throughout for speed (values are well-conditioned in practice).
///
/// input: [batch, features], gamma: [features], beta: [features]
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    features: usize,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), batch * features);
    debug_assert_eq!(gamma.len(), features);
    debug_assert_eq!(beta.len(), features);

    let mut output = vec![0.0f32; batch * features];
    let inv_n = 1.0 / features as f32;

    for b in 0..batch {
        let row = &input[b * features..(b + 1) * features];
        let out_row = &mut output[b * features..(b + 1) * features];

        // Pass 1: mean
        let mut sum = 0.0f32;
        for &v in row.iter() {
            sum += v;
        }
        let mean = sum * inv_n;

        // Pass 2: variance + normalize + affine (fused)
        let mut var_sum = 0.0f32;
        for &v in row.iter() {
            let d = v - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum * inv_n + eps).sqrt();

        for j in 0..features {
            out_row[j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
        }
    }

    output
}

// ============================================================================
// GELU Activation
// ============================================================================

/// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// In-place activation. Uses standard tanh for numerical accuracy
/// (error accumulates over many transformer layers).
pub fn gelu(input: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.7978845608; // sqrt(2/pi)
    const C: f32 = 0.044715;

    for x in input.iter_mut() {
        let v = *x;
        let inner = SQRT_2_PI * (v + C * v * v * v);
        *x = 0.5 * v * (1.0 + inner.tanh());
    }
}

// ============================================================================
// Softmax
// ============================================================================

/// Row-wise softmax: softmax(x[i]) = exp(x[i] - max) / sum(exp(x[j] - max))
///
/// input: [rows, cols], computed in-place
pub fn softmax_inplace(input: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut input[r * cols..(r + 1) * cols];

        // Numerical stability: subtract max
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in row.iter_mut() {
            *v = (*v - max).exp();
        }

        // Normalize
        let sum: f32 = row.iter().sum();
        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

/// Multi-head self-attention with causal masking.
///
/// input: [total_rows, hidden] where total_rows = batch * seq_len
/// wq, wk, wv: [hidden, hidden]  (Q/K/V projections)
/// wo: [hidden, hidden]  (output projection)
/// batch: number of independent sequences
/// seq_len: tokens per sequence (total_rows / batch)
///
/// For single-token (seq_len=1) this reduces to identity after V projection.
/// For multi-token, computes full causal self-attention per batch element.
///
/// Optimization: Uses strided GEMM to work directly on interleaved Q/K/V
/// projections, eliminating per-head copy/scatter overhead. For GPT-2
/// with seq=32, this eliminates 884K float copies per forward pass.
pub fn multi_head_attention(
    input: &[f32],           // [batch*seq_len, hidden]
    wq: &[f32], bq: Option<&[f32]>,  // [hidden, hidden]
    wk: &[f32], bk: Option<&[f32]>,
    wv: &[f32], bv: Option<&[f32]>,
    wo: &[f32], bo: Option<&[f32]>,
    batch: usize,
    seq_len: usize,
    hidden: usize,
    n_heads: usize,
) -> Vec<f32> {
    let head_dim = hidden / n_heads;
    let total_rows = batch * seq_len;
    debug_assert_eq!(hidden % n_heads, 0);
    debug_assert_eq!(input.len(), total_rows * hidden);

    // Q, K, V projections: [total_rows, hidden] @ [hidden, hidden].T = [total_rows, hidden]
    let q = fused_linear(input, wq, bq, total_rows, hidden, hidden, false);
    let k = fused_linear(input, wk, bk, total_rows, hidden, hidden, false);
    let v = fused_linear(input, wv, bv, total_rows, hidden, hidden, false);

    let mut attn_output = vec![0.0f32; total_rows * hidden];
    let scale = 1.0 / (head_dim as f32).sqrt();

    if seq_len == 1 {
        // Single-token fast path: softmax([q.k/sqrt(d)]) = [1.0], so attn = v
        for b in 0..batch {
            attn_output[b * hidden..(b + 1) * hidden]
                .copy_from_slice(&v[b * hidden..(b + 1) * hidden]);
        }
    } else {
        // Full causal self-attention with strided micro-kernel GEMM.
        //
        // Q, K, V are [total_rows, hidden] with head h at column offset h*head_dim.
        // Uses matrixmultiply::sgemm for per-head attention GEMMs because:
        // 1. Zero-copy: works directly on interleaved data via stride parameters
        // 2. Low overhead: ~2μs vs MKL's ~5-10μs per call for tiny matrices
        // 3. No thread pool: avoids Rayon/MKL thread synchronization
        //
        // For 12 heads × 12 blocks = 288 GEMM calls, this saves ~1-2ms.
        let mut scores = vec![0.0f32; seq_len * seq_len];
        let hidden_stride = hidden as isize;
        let seq_stride = seq_len as isize;

        for b in 0..batch {
            let b_off = b * seq_len * hidden;

            for h in 0..n_heads {
                let h_off = h * head_dim;

                let q_ptr = unsafe { q.as_ptr().add(b_off + h_off) };
                let k_ptr = unsafe { k.as_ptr().add(b_off + h_off) };
                let v_ptr = unsafe { v.as_ptr().add(b_off + h_off) };
                let out_ptr = unsafe { attn_output.as_mut_ptr().add(b_off + h_off) };

                // scores = (Q_h @ K_h^T) * scale -> [seq_len, seq_len]
                // matrixmultiply handles C = alpha * A @ B + beta * C
                // A = Q_h: [seq, head_dim], row stride=hidden, col stride=1
                // B = K_h^T: we want Q @ K^T, so B is K with swapped strides
                //   K: [seq, head_dim] with rs=hidden, cs=1
                //   K^T: rs=1, cs=hidden (swap row/col strides)
                unsafe {
                    matrixmultiply::sgemm(
                        seq_len, head_dim, seq_len,
                        scale,  // alpha fuses the 1/sqrt(d) scaling
                        q_ptr, hidden_stride, 1,   // Q_h: row stride=hidden, col stride=1
                        k_ptr, 1, hidden_stride,   // K_h^T: row stride=1, col stride=hidden
                        0.0,
                        scores.as_mut_ptr(), seq_stride, 1,
                    );
                }

                // Causal mask + softmax (scale already applied by GEMM)
                for i in 0..seq_len {
                    let row = &mut scores[i * seq_len..(i + 1) * seq_len];
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..=i {
                        if row[j] > max_val { max_val = row[j]; }
                    }
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        let e = (row[j] - max_val).exp();
                        row[j] = e;
                        sum += e;
                    }
                    let inv_sum = 1.0 / sum;
                    for j in 0..=i {
                        row[j] *= inv_sum;
                    }
                    for j in (i + 1)..seq_len {
                        row[j] = 0.0;
                    }
                }

                // out_h = scores @ V_h -> write directly to interleaved output
                // scores: [seq, seq] contiguous
                // V_h: [seq, head_dim] with rs=hidden, cs=1
                // out_h: [seq, head_dim] with rs=hidden, cs=1
                unsafe {
                    matrixmultiply::sgemm(
                        seq_len, seq_len, head_dim,
                        1.0,
                        scores.as_ptr(), seq_stride, 1,    // scores contiguous
                        v_ptr, hidden_stride, 1,           // V_h strided
                        0.0,
                        out_ptr, hidden_stride, 1,         // output strided (direct write)
                    );
                }
            }
        }
    }

    // Output projection: [total_rows, hidden] @ [hidden, hidden].T + bo
    fused_linear(&attn_output, wo, bo, total_rows, hidden, hidden, false)
}

// ============================================================================
// Transformer Block Forward
// ============================================================================

/// Op types for transformer forward pass
pub enum TransformerOp<'a> {
    Linear {
        weight: &'a [f32],
        bias: Option<&'a [f32]>,
        in_feat: usize,
        out_feat: usize,
        activation: &'a str,
    },
    LayerNorm {
        gamma: &'a [f32],
        beta: &'a [f32],
        features: usize,
        eps: f32,
    },
    Attention {
        wq: &'a [f32], bq: Option<&'a [f32]>,
        wk: &'a [f32], bk: Option<&'a [f32]>,
        wv: &'a [f32], bv: Option<&'a [f32]>,
        wo: &'a [f32], bo: Option<&'a [f32]>,
        hidden: usize,
        n_heads: usize,
    },
    ResidualStart,
    ResidualEnd,
    GELUActivation,
}

/// Execute a sequence of transformer operations.
/// Supports residual connections via ResidualStart/ResidualEnd markers.
///
/// input: [total_rows, features] where total_rows = batch * seq_len
/// batch: number of independent sequences
/// seq_len: tokens per sequence
pub fn transformer_forward(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    features: usize,
    ops: &[TransformerOp],
) -> Vec<f32> {
    let total_rows = batch * seq_len;
    let mut current = input.to_vec();
    let mut current_feat = features;
    let mut residual_stack: Vec<Vec<f32>> = Vec::new();

    for op in ops {
        match op {
            TransformerOp::Linear { weight, bias, in_feat, out_feat, activation } => {
                let is_relu = *activation == "relu";
                current = fused_linear(&current, weight, *bias, total_rows, *in_feat, *out_feat, is_relu);
                current_feat = *out_feat;
            }
            TransformerOp::LayerNorm { gamma, beta, features, eps } => {
                current = layer_norm(&current, gamma, beta, total_rows, *features, *eps);
                current_feat = *features;
            }
            TransformerOp::Attention { wq, bq, wk, bk, wv, bv, wo, bo, hidden, n_heads } => {
                current = multi_head_attention(
                    &current, wq, *bq, wk, *bk, wv, *bv, wo, *bo,
                    batch, seq_len, *hidden, *n_heads,
                );
                current_feat = *hidden;
            }
            TransformerOp::ResidualStart => {
                residual_stack.push(current.clone());
            }
            TransformerOp::ResidualEnd => {
                if let Some(residual) = residual_stack.pop() {
                    for (c, r) in current.iter_mut().zip(residual.iter()) {
                        *c += r;
                    }
                }
            }
            TransformerOp::GELUActivation => {
                gelu(&mut current);
            }
        }
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_linear_no_activation() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]
        let bias = vec![0.1, 0.2];

        let output = fused_linear(&input, &weight, Some(&bias), 2, 3, 2, false);

        assert!((output[0] - 1.1).abs() < 1e-5);
        assert!((output[1] - 2.2).abs() < 1e-5);
        assert!((output[2] - 4.1).abs() < 1e-5);
        assert!((output[3] - 5.2).abs() < 1e-5);
    }

    #[test]
    fn test_mlp_forward() {
        let input = vec![1.0, 2.0]; // [1, 2]
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
        let b1 = vec![0.0, 0.0, 0.0];
        let w2 = vec![1.0, 1.0, 1.0]; // [1, 3]
        let b2 = vec![0.0];

        let layers: Vec<(&[f32], Option<&[f32]>, usize, bool)> = vec![
            (&w1, Some(&b1), 3, true),
            (&w2, Some(&b2), 1, false),
        ];

        let output = mlp_forward(&input, 1, 2, &layers);
        assert!((output[0] - 6.0).abs() < 1e-5);
    }
}
