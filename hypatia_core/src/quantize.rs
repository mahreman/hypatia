//! INT4 Block Quantization with AVX2 SIMD for Large Model Inference
//!
//! For large models (1B+ params), memory bandwidth is the bottleneck:
//! - LLaMA-7B FFN weights: 344MB per layer in f32
//! - CPU memory bandwidth: ~40 GB/s
//! - Reading weights alone: 8.6ms per layer
//!
//! INT4 quantization reduces this by 8x:
//! - Same weights in INT4: 43MB per layer
//! - Reading time: 1.1ms per layer
//!
//! AVX2 SIMD acceleration:
//! - Fused dequantize + dot product: no intermediate f32 buffer needed
//! - Process 16 INT4 values per iteration (2x __m256 FMA)
//! - Rayon parallelism across output rows (16 cores)
//!
//! Format: Block quantization with group_size elements per block
//! Each block stores: scale (f32) + zero_point (f32) + packed int4 values
//! Dequantization: value = scale * (int4_value - zero_point)

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Group size for block quantization (128 is standard, like GPTQ/AWQ)
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Quantized weight block: scale + zero_point + packed int4 values
/// Each u8 stores 2 int4 values (low nibble + high nibble)
#[derive(Clone)]
pub struct QuantizedTensor {
    /// Scale per group: value = scale * (q - zero_point)
    pub scales: Vec<f32>,
    /// Zero point per group
    pub zeros: Vec<f32>,
    /// Packed INT4 values: 2 values per byte (low nibble first)
    pub data: Vec<u8>,
    /// Original dimensions
    pub rows: usize,
    pub cols: usize,
    /// Group size
    pub group_size: usize,
}

impl QuantizedTensor {
    /// Quantize a f32 weight matrix [rows, cols] to INT4 block format.
    /// Groups along the column (in_features) dimension.
    pub fn quantize(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);

        let num_groups_per_row = (cols + group_size - 1) / group_size;
        let total_groups = rows * num_groups_per_row;

        let mut scales = vec![0.0f32; total_groups];
        let mut zeros = vec![0.0f32; total_groups];
        let packed_cols = (cols + 1) / 2;
        let mut data = vec![0u8; rows * packed_cols];

        for row in 0..rows {
            for g in 0..num_groups_per_row {
                let group_idx = row * num_groups_per_row + g;
                let col_start = g * group_size;
                let col_end = (col_start + group_size).min(cols);

                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for c in col_start..col_end {
                    let v = weights[row * cols + c];
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }

                let range = max_val - min_val;
                let scale = if range > 1e-10 { range / 15.0 } else { 1.0 };
                let zero = min_val / scale;

                scales[group_idx] = scale;
                zeros[group_idx] = -zero;

                for c in col_start..col_end {
                    let v = weights[row * cols + c];
                    let q = ((v - min_val) / scale).round().clamp(0.0, 15.0) as u8;
                    let byte_idx = row * packed_cols + c / 2;
                    if c % 2 == 0 {
                        data[byte_idx] = (data[byte_idx] & 0xF0) | (q & 0x0F);
                    } else {
                        data[byte_idx] = (data[byte_idx] & 0x0F) | (q << 4);
                    }
                }
            }
        }

        QuantizedTensor { scales, zeros, data, rows, cols, group_size }
    }

    /// Dequantize a single value
    #[inline(always)]
    fn dequant_value(&self, row: usize, col: usize) -> f32 {
        let packed_cols = (self.cols + 1) / 2;
        let byte_idx = row * packed_cols + col / 2;
        let q = if col % 2 == 0 {
            self.data[byte_idx] & 0x0F
        } else {
            self.data[byte_idx] >> 4
        } as f32;

        let num_groups_per_row = (self.cols + self.group_size - 1) / self.group_size;
        let group_idx = row * num_groups_per_row + col / self.group_size;

        self.scales[group_idx] * (q - self.zeros[group_idx])
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let packed_cols = (self.cols + 1) / 2;
        let num_groups = self.scales.len();
        self.rows * packed_cols + num_groups * 4 * 2
    }

    /// Original f32 memory usage
    pub fn original_bytes(&self) -> usize {
        self.rows * self.cols * 4
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.original_bytes() as f32 / self.memory_bytes() as f32
    }
}

// ============================================================================
// AVX2 SIMD Fused INT4 Dot Product
// ============================================================================

/// Horizontal sum of __m256 (8 x f32) -> f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    // [a0+a4, a1+a5, a2+a6, a3+a7]
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    // [s0+s2, s1+s3, ...]
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    // [s0+s1, ...]
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// AVX2 fused INT4 dequantize + dot product for one weight row.
///
/// Computes: sum_i( scale[g] * (q[i] - zero[g]) * input[i] )
/// where g = group index for element i.
///
/// Processes 16 INT4 values per iteration:
/// 1. Load 8 packed bytes = 16 INT4 values
/// 2. Extract low/high nibbles via AND + SHIFT
/// 3. Interleave to original order via UNPACKLO
/// 4. Zero-extend u8 -> i32 -> f32
/// 5. Dequantize: scale * (q - zero)
/// 6. FMA with input values
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_dot_avx2(
    input: &[f32],
    packed_row: &[u8],
    row_scales: &[f32],
    row_zeros: &[f32],
    group_size: usize,
    in_features: usize,
) -> f32 {
    let num_groups = (in_features + group_size - 1) / group_size;
    let mask_0f = _mm_set1_epi8(0x0F);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut scalar_acc = 0.0f32;

    for g in 0..num_groups {
        let col_start = g * group_size;
        let col_end = (col_start + group_size).min(in_features);
        let group_len = col_end - col_start;

        let scale_v = _mm256_set1_ps(row_scales[g]);
        let zero_v = _mm256_set1_ps(row_zeros[g]);

        let mut c = 0;

        // Process 16 values per iteration (8 packed bytes)
        while c + 16 <= group_len {
            let byte_offset = (col_start + c) / 2;

            // Load 8 bytes = 16 INT4 values
            let packed_bytes = _mm_loadl_epi64(
                packed_row.as_ptr().add(byte_offset) as *const __m128i,
            );

            // Extract nibbles
            let low = _mm_and_si128(packed_bytes, mask_0f);
            let high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), mask_0f);

            // Interleave: [val0, val1, val2, ..., val15]
            let interleaved = _mm_unpacklo_epi8(low, high);

            // Convert first 8 values to f32
            let vals_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
            // Convert next 8 values to f32
            let vals_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_srli_si128(interleaved, 8),
            ));

            // Dequantize: scale * (q - zero)
            let dq_lo = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals_lo, zero_v));
            let dq_hi = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals_hi, zero_v));

            // Load input values and FMA
            let inp_lo = _mm256_loadu_ps(input.as_ptr().add(col_start + c));
            let inp_hi = _mm256_loadu_ps(input.as_ptr().add(col_start + c + 8));

            acc0 = _mm256_fmadd_ps(dq_lo, inp_lo, acc0);
            acc1 = _mm256_fmadd_ps(dq_hi, inp_hi, acc1);

            c += 16;
        }

        // Handle remaining 8 values
        if c + 8 <= group_len {
            let byte_offset = (col_start + c) / 2;
            let raw = std::ptr::read_unaligned(
                packed_row.as_ptr().add(byte_offset) as *const u32,
            );
            let packed_bytes = _mm_cvtsi32_si128(raw as i32);

            let low = _mm_and_si128(packed_bytes, mask_0f);
            let high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), mask_0f);
            let interleaved = _mm_unpacklo_epi8(low, high);

            let vals = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
            let dq = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals, zero_v));
            let inp = _mm256_loadu_ps(input.as_ptr().add(col_start + c));

            acc0 = _mm256_fmadd_ps(dq, inp, acc0);
            c += 8;
        }

        // Scalar remainder
        while c < group_len {
            let abs_col = col_start + c;
            let byte_idx = abs_col / 2;
            let q = if abs_col % 2 == 0 {
                (packed_row[byte_idx] & 0x0F) as f32
            } else {
                (packed_row[byte_idx] >> 4) as f32
            };
            scalar_acc += row_scales[g] * (q - row_zeros[g]) * input[abs_col];
            c += 1;
        }
    }

    // Combine accumulators
    let combined = _mm256_add_ps(acc0, acc1);
    hsum256_ps(combined) + scalar_acc
}

/// Scalar fallback for non-x86_64 platforms
#[cfg(not(target_arch = "x86_64"))]
fn q4_dot_scalar(
    input: &[f32],
    packed_row: &[u8],
    row_scales: &[f32],
    row_zeros: &[f32],
    group_size: usize,
    in_features: usize,
) -> f32 {
    let num_groups = (in_features + group_size - 1) / group_size;
    let mut acc = 0.0f32;
    for g in 0..num_groups {
        let col_start = g * group_size;
        let col_end = (col_start + group_size).min(in_features);
        let scale = row_scales[g];
        let zero = row_zeros[g];
        for c in col_start..col_end {
            let byte_idx = c / 2;
            let q = if c % 2 == 0 {
                (packed_row[byte_idx] & 0x0F) as f32
            } else {
                (packed_row[byte_idx] >> 4) as f32
            };
            acc += scale * (q - zero) * input[c];
        }
    }
    acc
}

// ============================================================================
// SIMD-Accelerated Dequantization for Tiled BLAS Path (batch > 1)
// ============================================================================

/// AVX2 SIMD dequantization of a weight row group into f32 buffer.
/// Processes 16 INT4 values per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_group_avx2(
    packed_row: &[u8],
    byte_start: usize,
    scale: f32,
    zero: f32,
    output: &mut [f32],
    count: usize,
) {
    let mask_0f = _mm_set1_epi8(0x0F);
    let scale_v = _mm256_set1_ps(scale);
    let zero_v = _mm256_set1_ps(zero);

    let mut c = 0;

    while c + 16 <= count {
        let byte_offset = byte_start + c / 2;
        let packed_bytes = _mm_loadl_epi64(
            packed_row.as_ptr().add(byte_offset) as *const __m128i,
        );

        let low = _mm_and_si128(packed_bytes, mask_0f);
        let high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), mask_0f);
        let interleaved = _mm_unpacklo_epi8(low, high);

        let vals_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
        let vals_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm_srli_si128(interleaved, 8),
        ));

        let dq_lo = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals_lo, zero_v));
        let dq_hi = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals_hi, zero_v));

        _mm256_storeu_ps(output.as_mut_ptr().add(c), dq_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add(c + 8), dq_hi);

        c += 16;
    }

    // Handle remaining 8
    if c + 8 <= count {
        let byte_offset = byte_start + c / 2;
        let raw = std::ptr::read_unaligned(
            packed_row.as_ptr().add(byte_offset) as *const u32,
        );
        let packed_bytes = _mm_cvtsi32_si128(raw as i32);

        let low = _mm_and_si128(packed_bytes, mask_0f);
        let high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), mask_0f);
        let interleaved = _mm_unpacklo_epi8(low, high);

        let vals = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
        let dq = _mm256_mul_ps(scale_v, _mm256_sub_ps(vals, zero_v));

        _mm256_storeu_ps(output.as_mut_ptr().add(c), dq);
        c += 8;
    }

    // Scalar remainder
    while c < count {
        let actual_byte = byte_start + c / 2;
        let q = if c % 2 == 0 {
            (packed_row[actual_byte] & 0x0F) as f32
        } else {
            (packed_row[actual_byte] >> 4) as f32
        };
        output[c] = scale * (q - zero);
        c += 1;
    }
}

// ============================================================================
// OpenBLAS FFI
// ============================================================================

extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Tile size for batch > 1 path.
const TILE_ROWS: usize = 256;

// ============================================================================
// Main Entry Point: Quantized Linear Layer
// ============================================================================

/// Zero-copy version: takes raw slices instead of owned QuantizedTensor.
/// Avoids copying weight data from numpy arrays every inference call.
pub fn quantized_linear_ref(
    input: &[f32],
    packed_data: &[u8],
    scales: &[f32],
    zeros: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    let in_feat = cols;
    let out_feat = rows;
    debug_assert_eq!(input.len(), batch * in_feat);

    let packed_cols = (in_feat + 1) / 2;
    let num_groups_per_row = (in_feat + group_size - 1) / group_size;

    let mut output = vec![0.0f32; batch * out_feat];

    if batch <= 4 {
        // === FUSED SIMD DOT PRODUCT + RAYON PARALLELISM ===
        // Best for batch=1 (LLM inference): directly compute dot products
        // from INT4 packed data without intermediate dequantized buffer.

        // For each batch element, parallelize across output rows
        for b in 0..batch {
            let input_row = &input[b * in_feat..(b + 1) * in_feat];
            let output_row = &mut output[b * out_feat..(b + 1) * out_feat];

            output_row
                .par_iter_mut()
                .enumerate()
                .for_each(|(o, out_val)| {
                    let packed_start = o * packed_cols;
                    let packed_end = packed_start + packed_cols;
                    let packed_row = &packed_data[packed_start..packed_end];
                    let scales_start = o * num_groups_per_row;
                    let row_scales =
                        &scales[scales_start..scales_start + num_groups_per_row];
                    let row_zeros =
                        &zeros[scales_start..scales_start + num_groups_per_row];

                    #[cfg(target_arch = "x86_64")]
                    {
                        *out_val = unsafe {
                            q4_dot_avx2(
                                input_row,
                                packed_row,
                                row_scales,
                                row_zeros,
                                group_size,
                                in_feat,
                            )
                        };
                    }

                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        *out_val = q4_dot_scalar(
                            input_row,
                            packed_row,
                            row_scales,
                            row_zeros,
                            group_size,
                            in_feat,
                        );
                    }
                });
        }
    } else {
        // === SIMD DEQUANT + TILED BLAS GEMM ===
        // Better for batch > 4: BLAS amortizes dequant cost across batch dimension

        for g in 0..num_groups_per_row {
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(in_feat);
            let tile_cols = col_end - col_start;
            let is_first_group = g == 0;

            let mut row_start = 0;
            while row_start < out_feat {
                let row_end = (row_start + TILE_ROWS).min(out_feat);
                let tile_rows = row_end - row_start;

                // SIMD-accelerated dequantization
                let mut dequant = vec![0.0f32; tile_rows * tile_cols];

                for r in 0..tile_rows {
                    let weight_row = row_start + r;
                    let group_idx = weight_row * num_groups_per_row + g;
                    let scale = scales[group_idx];
                    let zero = zeros[group_idx];
                    let byte_start = weight_row * packed_cols + col_start / 2;
                    let dequant_offset = r * tile_cols;

                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        dequant_group_avx2(
                            packed_data,
                            byte_start,
                            scale,
                            zero,
                            &mut dequant[dequant_offset..dequant_offset + tile_cols],
                            tile_cols,
                        );
                    }

                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for c in 0..tile_cols {
                            let abs_col = col_start + c;
                            let byte_idx = weight_row * packed_cols + abs_col / 2;
                            let q = if abs_col % 2 == 0 {
                                (packed_data[byte_idx] & 0x0F) as f32
                            } else {
                                (packed_data[byte_idx] >> 4) as f32
                            };
                            dequant[dequant_offset + c] = scale * (q - zero);
                        }
                    }
                }

                // GEMM: output[:, row_start:row_end] += input[:, col_start:col_end] @ dequant.T
                let beta = if is_first_group { 0.0 } else { 1.0 };

                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_TRANS,
                        batch as i32,
                        tile_rows as i32,
                        tile_cols as i32,
                        1.0,
                        input.as_ptr().add(col_start),
                        in_feat as i32,
                        dequant.as_ptr(),
                        tile_cols as i32,
                        beta,
                        output.as_mut_ptr().add(row_start),
                        out_feat as i32,
                    );
                }

                row_start = row_end;
            }
        }
    }

    // Fused bias + activation
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
        (None, false) => {}
    }

    output
}

/// Convenience wrapper that delegates to quantized_linear_ref using QuantizedTensor fields.
pub fn quantized_linear(
    input: &[f32],
    weight: &QuantizedTensor,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    quantized_linear_ref(
        input,
        &weight.data,
        &weight.scales,
        &weight.zeros,
        weight.rows,
        weight.cols,
        weight.group_size,
        bias,
        batch,
        relu,
    )
}

/// Multi-layer quantized MLP forward pass
pub fn quantized_mlp_forward(
    input: &[f32],
    batch: usize,
    in_features: usize,
    layers: &[(&QuantizedTensor, Option<&[f32]>, bool)],
) -> Vec<f32> {
    let mut current = input.to_vec();
    let mut _current_feat = in_features;

    for (weight, bias, relu) in layers {
        current = quantized_linear(&current, weight, *bias, batch, *relu);
        _current_feat = weight.rows;
    }

    current
}

/// Quantization error (RMSE between original and dequantized values)
pub fn quantization_error(original: &[f32], quantized: &QuantizedTensor) -> f32 {
    let rows = quantized.rows;
    let cols = quantized.cols;
    let mut sum_sq = 0.0f64;

    for r in 0..rows {
        for c in 0..cols {
            let orig = original[r * cols + c] as f64;
            let deq = quantized.dequant_value(r, c) as f64;
            let diff = orig - deq;
            sum_sq += diff * diff;
        }
    }

    (sum_sq / (rows * cols) as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let weights = vec![
            0.0, 0.5, 1.0, -1.0,
            0.3, -0.7, 0.8, 0.1,
        ];

        let qt = QuantizedTensor::quantize(&weights, 2, 4, 4);
        let rmse = quantization_error(&weights, &qt);

        assert!(rmse < 0.15, "RMSE too high: {}", rmse);
        assert!(qt.compression_ratio() > 2.0, "Compression: {}", qt.compression_ratio());
    }

    #[test]
    fn test_quantized_linear() {
        let weights = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let qt = QuantizedTensor::quantize(&weights, 2, 2, 2);

        let input = vec![3.0, 4.0];
        let output = quantized_linear(&input, &qt, None, 1, false);

        assert!((output[0] - 3.0).abs() < 0.5, "out[0]={}", output[0]);
        assert!((output[1] - 4.0).abs() < 0.5, "out[1]={}", output[1]);
    }

    #[test]
    fn test_simd_dot_correctness() {
        // Test with a realistic size that exercises SIMD paths
        let cols = 256;
        let rows = 64;
        let group_size = 128;

        // Random-ish weights
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.1).sin() * 2.0))
            .collect();
        let qt = QuantizedTensor::quantize(&weights, rows, cols, group_size);

        // Random-ish input
        let input: Vec<f32> = (0..cols)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();

        // Compute via quantized_linear (uses SIMD path for batch=1)
        let simd_output = quantized_linear(&input, &qt, None, 1, false);

        // Compute reference via scalar dequant + dot
        let packed_cols = (cols + 1) / 2;
        let num_groups = (cols + group_size - 1) / group_size;
        for o in 0..rows {
            let mut ref_val = 0.0f32;
            for c in 0..cols {
                let dq = qt.dequant_value(o, c);
                ref_val += dq * input[c];
            }
            let diff = (simd_output[o] - ref_val).abs();
            assert!(diff < 0.01, "Row {}: SIMD={}, ref={}, diff={}", o, simd_output[o], ref_val, diff);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let n = 4096 * 11008;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
        let qt = QuantizedTensor::quantize(&weights, 4096, 11008, 128);

        let ratio = qt.compression_ratio();
        assert!(ratio > 6.0, "Expected >6x compression, got {:.2}x", ratio);

        let orig_mb = qt.original_bytes() as f64 / 1024.0 / 1024.0;
        let quant_mb = qt.memory_bytes() as f64 / 1024.0 / 1024.0;
        eprintln!("LLaMA-7B FFN layer: {:.1}MB -> {:.1}MB ({:.1}x)", orig_mb, quant_mb, ratio);
    }
}
