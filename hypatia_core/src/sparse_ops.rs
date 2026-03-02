//! Sparse Tensor IR: CSR format + sparse-dense GEMM kernels
//!
//! For pruned neural networks, weight matrices are often 50-90% zeros.
//! Storing and computing with sparse formats provides:
//! - Memory reduction: 2-8x depending on sparsity level
//! - Compute reduction: proportional to non-zero count
//! - Bandwidth savings: fewer cache misses on large models
//!
//! Format: CSR (Compressed Sparse Row)
//! - row_ptrs[i]..row_ptrs[i+1] = indices of non-zeros in row i
//! - col_indices[k] = column of k-th non-zero
//! - values[k] = value of k-th non-zero
//!
//! GEMM: output[b][i] = sum_j sparse_weight[i][j] * input[b][j] + bias[i]
//! Only iterates over non-zero elements → O(nnz * batch) vs O(rows * cols * batch)

use rayon::prelude::*;

// ============================================================================
// CSR Sparse Weight
// ============================================================================

/// Compressed Sparse Row weight matrix for efficient sparse-dense GEMM.
#[derive(Clone, Debug)]
pub struct SparseWeightCSR {
    /// Row pointers: row_ptrs[i]..row_ptrs[i+1] are non-zeros in row i.
    /// Length: rows + 1
    pub row_ptrs: Vec<usize>,
    /// Column index for each non-zero value. Length: nnz
    pub col_indices: Vec<usize>,
    /// Non-zero values. Length: nnz
    pub values: Vec<f32>,
    /// Number of rows (out_features)
    pub rows: usize,
    /// Number of columns (in_features)
    pub cols: usize,
}

impl SparseWeightCSR {
    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio: fraction of zeros (0.0 = fully dense, 1.0 = all zeros).
    pub fn sparsity(&self) -> f32 {
        let total = (self.rows * self.cols) as f32;
        if total == 0.0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f32 / total)
    }

    /// Memory size in bytes (CSR format).
    pub fn memory_bytes(&self) -> usize {
        // row_ptrs: (rows+1) * 8 bytes (usize)
        // col_indices: nnz * 8 bytes (usize)
        // values: nnz * 4 bytes (f32)
        (self.rows + 1) * std::mem::size_of::<usize>()
            + self.nnz() * std::mem::size_of::<usize>()
            + self.nnz() * std::mem::size_of::<f32>()
    }

    /// Dense memory size in bytes for comparison.
    pub fn dense_memory_bytes(&self) -> usize {
        self.rows * self.cols * std::mem::size_of::<f32>()
    }

    /// Compression ratio: dense_bytes / sparse_bytes.
    pub fn compression_ratio(&self) -> f32 {
        let sparse = self.memory_bytes();
        if sparse == 0 {
            return 0.0;
        }
        self.dense_memory_bytes() as f32 / sparse as f32
    }
}

// ============================================================================
// Dense → CSR Conversion
// ============================================================================

/// Convert a dense weight matrix to CSR format.
/// Values with absolute value below `threshold` are treated as zero.
///
/// # Arguments
/// - `weights`: row-major dense [rows, cols] f32 array
/// - `rows`: number of rows (out_features)
/// - `cols`: number of columns (in_features)
/// - `threshold`: minimum absolute value to keep (magnitude pruning)
pub fn to_sparse_csr(
    weights: &[f32],
    rows: usize,
    cols: usize,
    threshold: f32,
) -> SparseWeightCSR {
    debug_assert_eq!(weights.len(), rows * cols);

    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0);

    for i in 0..rows {
        for j in 0..cols {
            let val = weights[i * cols + j];
            if val.abs() >= threshold {
                col_indices.push(j);
                values.push(val);
            }
        }
        row_ptrs.push(values.len());
    }

    SparseWeightCSR {
        row_ptrs,
        col_indices,
        values,
        rows,
        cols,
    }
}

/// Convert a dense weight matrix to CSR using a boolean mask.
/// mask[i] = true means keep the value.
pub fn to_sparse_csr_masked(
    weights: &[f32],
    mask: &[bool],
    rows: usize,
    cols: usize,
) -> SparseWeightCSR {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(mask.len(), rows * cols);

    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0);

    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            if mask[idx] {
                col_indices.push(j);
                values.push(weights[idx]);
            }
        }
        row_ptrs.push(values.len());
    }

    SparseWeightCSR {
        row_ptrs,
        col_indices,
        values,
        rows,
        cols,
    }
}

/// Convert CSR back to dense format (for validation / debugging).
pub fn to_dense(csr: &SparseWeightCSR) -> Vec<f32> {
    let mut dense = vec![0.0f32; csr.rows * csr.cols];
    for i in 0..csr.rows {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            dense[i * csr.cols + j] = csr.values[k];
        }
    }
    dense
}

// ============================================================================
// Sparse-Dense GEMM: output = sparse_weight @ input.T (per batch row)
// ============================================================================

/// Fused sparse linear: output = activation(input @ sparse_weight.T + bias)
///
/// CSR format enables row-wise iteration over non-zeros:
/// for each output row i, accumulate only non-zero weight[i][j] * input[b][j]
///
/// # Arguments
/// - `input`: [batch, in_feat] dense row-major
/// - `weight_csr`: CSR sparse weight [out_feat, in_feat]
/// - `bias`: optional [out_feat]
/// - `batch`: number of batch rows
/// - `relu`: apply ReLU activation
///
/// # Returns
/// - [batch, out_feat] dense output
pub fn sparse_linear(
    input: &[f32],
    weight_csr: &SparseWeightCSR,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    let in_feat = weight_csr.cols;
    let out_feat = weight_csr.rows;
    debug_assert_eq!(input.len(), batch * in_feat);

    let mut output = vec![0.0f32; batch * out_feat];

    for b in 0..batch {
        let input_row = &input[b * in_feat..(b + 1) * in_feat];
        let output_row = &mut output[b * out_feat..(b + 1) * out_feat];

        for i in 0..out_feat {
            let mut acc = 0.0f32;
            let start = weight_csr.row_ptrs[i];
            let end = weight_csr.row_ptrs[i + 1];

            for k in start..end {
                let j = weight_csr.col_indices[k];
                acc += weight_csr.values[k] * input_row[j];
            }

            if let Some(b) = bias {
                acc += b[i];
            }

            if relu {
                acc = acc.max(0.0);
            }

            output_row[i] = acc;
        }
    }

    output
}

/// Parallel sparse linear using Rayon (for large batch sizes).
/// Each batch row is processed independently.
pub fn sparse_linear_parallel(
    input: &[f32],
    weight_csr: &SparseWeightCSR,
    bias: Option<&[f32]>,
    batch: usize,
    relu: bool,
) -> Vec<f32> {
    let in_feat = weight_csr.cols;
    let out_feat = weight_csr.rows;
    debug_assert_eq!(input.len(), batch * in_feat);

    // Process batch rows in parallel
    let results: Vec<Vec<f32>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let input_row = &input[b * in_feat..(b + 1) * in_feat];
            let mut row_output = vec![0.0f32; out_feat];

            for i in 0..out_feat {
                let mut acc = 0.0f32;
                let start = weight_csr.row_ptrs[i];
                let end = weight_csr.row_ptrs[i + 1];

                for k in start..end {
                    let j = weight_csr.col_indices[k];
                    acc += weight_csr.values[k] * input_row[j];
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

    // Flatten
    let mut output = Vec::with_capacity(batch * out_feat);
    for row in results {
        output.extend_from_slice(&row);
    }
    output
}

// ============================================================================
// Magnitude Pruning
// ============================================================================

/// Compute the pruning threshold for a given sparsity ratio.
/// Uses magnitude-based pruning: smallest |weight| values are zeroed.
///
/// # Arguments
/// - `weights`: flat weight array
/// - `sparsity`: target fraction of zeros (0.5 = 50% sparse)
///
/// # Returns
/// - threshold value: |weight| < threshold → pruned to zero
pub fn compute_pruning_threshold(weights: &[f32], sparsity: f32) -> f32 {
    if sparsity <= 0.0 {
        return 0.0;
    }
    if sparsity >= 1.0 {
        return f32::MAX;
    }

    let mut magnitudes: Vec<f32> = weights.iter().map(|v| v.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((magnitudes.len() as f32 * sparsity) as usize).min(magnitudes.len() - 1);
    magnitudes[idx]
}

/// Apply magnitude pruning: zero out weights below threshold.
/// Returns the pruned weight array.
pub fn magnitude_prune(weights: &[f32], threshold: f32) -> Vec<f32> {
    weights
        .iter()
        .map(|&v| if v.abs() >= threshold { v } else { 0.0 })
        .collect()
}

/// Compute sparsity stats for a weight matrix.
pub fn sparsity_stats(weights: &[f32]) -> SparsityStats {
    let total = weights.len();
    let nonzero = weights.iter().filter(|&&v| v != 0.0).count();
    let zero = total - nonzero;

    SparsityStats {
        total_elements: total,
        nonzero_elements: nonzero,
        zero_elements: zero,
        sparsity_ratio: zero as f32 / total as f32,
        dense_bytes: total * 4,
        sparse_bytes_estimate: nonzero * 12 + (total / 128 + 1) * 8, // CSR estimate
    }
}

/// Statistics about weight sparsity.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    pub total_elements: usize,
    pub nonzero_elements: usize,
    pub zero_elements: usize,
    pub sparsity_ratio: f32,
    pub dense_bytes: usize,
    pub sparse_bytes_estimate: usize,
}

// ============================================================================
// Multi-layer Sparse MLP
// ============================================================================

/// Sparse MLP forward: multiple sparse linear layers.
pub fn sparse_mlp_forward(
    input: &[f32],
    batch: usize,
    in_features: usize,
    layers: &[(&SparseWeightCSR, Option<&[f32]>, bool)],
) -> Vec<f32> {
    let mut current = input.to_vec();
    let mut current_feat = in_features;

    for (weight_csr, bias, is_relu) in layers {
        debug_assert_eq!(weight_csr.cols, current_feat);

        if batch >= 4 {
            current = sparse_linear_parallel(&current, weight_csr, *bias, batch, *is_relu);
        } else {
            current = sparse_linear(&current, weight_csr, *bias, batch, *is_relu);
        }
        current_feat = weight_csr.rows;
    }

    current
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_sparse_csr_basic() {
        // 2x3 matrix:
        // [1.0, 0.0, 2.0]
        // [0.0, 3.0, 0.0]
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let csr = to_sparse_csr(&dense, 2, 3, 0.01);

        assert_eq!(csr.rows, 2);
        assert_eq!(csr.cols, 3);
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.row_ptrs, vec![0, 2, 3]);
        assert_eq!(csr.col_indices, vec![0, 2, 1]);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let csr = to_sparse_csr(&dense, 2, 3, 0.01);
        let recovered = to_dense(&csr);
        assert_eq!(recovered, dense);
    }

    #[test]
    fn test_sparsity_ratio() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let csr = to_sparse_csr(&dense, 2, 3, 0.01);
        assert!((csr.sparsity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sparse_linear_basic() {
        // Weight: [[1, 0], [0, 2]] (identity-ish, 50% sparse)
        let weight = vec![1.0, 0.0, 0.0, 2.0];
        let csr = to_sparse_csr(&weight, 2, 2, 0.01);

        // Input: [[3, 4]]
        let input = vec![3.0, 4.0];
        let output = sparse_linear(&input, &csr, None, 1, false);

        // Expected: [3*1 + 4*0, 3*0 + 4*2] = [3, 8]
        assert_eq!(output, vec![3.0, 8.0]);
    }

    #[test]
    fn test_sparse_linear_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 2.0];
        let csr = to_sparse_csr(&weight, 2, 2, 0.01);
        let bias = vec![10.0, 20.0];

        let input = vec![3.0, 4.0];
        let output = sparse_linear(&input, &csr, Some(&bias), 1, false);

        assert_eq!(output, vec![13.0, 28.0]);
    }

    #[test]
    fn test_sparse_linear_relu() {
        let weight = vec![1.0, 0.0, 0.0, -2.0];
        let csr = to_sparse_csr(&weight, 2, 2, 0.01);
        let bias = vec![-5.0, 0.0];

        let input = vec![3.0, 4.0];
        let output = sparse_linear(&input, &csr, Some(&bias), 1, true);

        // [3-5, -8+0] = [-2, -8] → ReLU → [0, 0]
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn test_sparse_linear_batch() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let csr = to_sparse_csr(&weight, 2, 2, 0.01);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=2
        let output = sparse_linear(&input, &csr, None, 2, false);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sparse_vs_dense_equivalence() {
        // Random-ish dense weight with ~50% sparsity
        let weight = vec![
            0.5, 0.0, -0.3, 0.0,
            0.0, 0.7, 0.0, -0.2,
            0.1, 0.0, 0.0, 0.4,
        ];
        let rows = 3;
        let cols = 4;
        let csr = to_sparse_csr(&weight, rows, cols, 0.01);

        let input = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // batch=2
        let bias = vec![0.1, 0.2, 0.3];

        // Sparse forward
        let sparse_out = sparse_linear(&input, &csr, Some(&bias), 2, false);

        // Dense forward (manual)
        let mut dense_out = vec![0.0f32; 2 * 3];
        for b in 0..2 {
            for i in 0..rows {
                let mut acc = bias[i];
                for j in 0..cols {
                    acc += weight[i * cols + j] * input[b * cols + j];
                }
                dense_out[b * rows + i] = acc;
            }
        }

        for (s, d) in sparse_out.iter().zip(dense_out.iter()) {
            assert!((s - d).abs() < 1e-6, "sparse={} dense={}", s, d);
        }
    }

    #[test]
    fn test_magnitude_pruning() {
        let weights = vec![0.1, -0.5, 0.02, 0.8, -0.03, 0.3];
        let threshold = compute_pruning_threshold(&weights, 0.5);
        let pruned = magnitude_prune(&weights, threshold);

        // 50% sparsity: 3 smallest |values| (0.02, 0.03, 0.1) → zero
        let nonzero_count = pruned.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero_count == 3, "Expected 3 non-zeros, got {}", nonzero_count);
    }

    #[test]
    fn test_sparsity_stats() {
        let weights = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0];
        let stats = sparsity_stats(&weights);
        assert_eq!(stats.total_elements, 6);
        assert_eq!(stats.nonzero_elements, 3);
        assert_eq!(stats.zero_elements, 3);
        assert!((stats.sparsity_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sparse_linear_parallel_matches_serial() {
        let weight = vec![
            0.5, 0.0, -0.3, 0.0,
            0.0, 0.7, 0.0, -0.2,
            0.1, 0.0, 0.0, 0.4,
        ];
        let csr = to_sparse_csr(&weight, 3, 4, 0.01);
        let bias = vec![0.1, 0.2, 0.3];

        let input = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // batch=2

        let serial = sparse_linear(&input, &csr, Some(&bias), 2, true);
        let parallel = sparse_linear_parallel(&input, &csr, Some(&bias), 2, true);

        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert!((s - p).abs() < 1e-6, "serial={} parallel={}", s, p);
        }
    }

    #[test]
    fn test_empty_row() {
        // Row 1 is all zeros
        let weight = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0];
        let csr = to_sparse_csr(&weight, 3, 2, 0.01);

        let input = vec![1.0, 1.0];
        let output = sparse_linear(&input, &csr, None, 1, false);

        assert_eq!(output, vec![3.0, 0.0, 7.0]);
    }

    #[test]
    fn test_fully_sparse() {
        let weight = vec![0.0; 6];
        let csr = to_sparse_csr(&weight, 2, 3, 0.01);
        assert_eq!(csr.nnz(), 0);
        assert!((csr.sparsity() - 1.0).abs() < 0.01);

        let input = vec![1.0, 2.0, 3.0];
        let output = sparse_linear(&input, &csr, None, 1, false);
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn test_fully_dense() {
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let csr = to_sparse_csr(&weight, 2, 2, 0.01);
        assert_eq!(csr.nnz(), 4);
        assert!(csr.sparsity() < 0.01);
    }

    #[test]
    fn test_compression_ratio() {
        // 1000x1000 matrix, 90% sparse → ~100K non-zeros
        // Dense: 4MB, CSR: ~1.2MB → compression ~3.3x
        let n = 100;
        let mut weights = vec![0.0f32; n * n];
        // Set 10% non-zero
        for i in 0..n {
            for j in 0..n {
                if (i * 7 + j * 13) % 10 == 0 {
                    weights[i * n + j] = 1.0;
                }
            }
        }
        let csr = to_sparse_csr(&weights, n, n, 0.01);
        assert!(csr.compression_ratio() > 1.5, "Should compress: ratio={}", csr.compression_ratio());
    }

    #[test]
    fn test_masked_csr() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, false, true, false, false, true];
        let csr = to_sparse_csr_masked(&weights, &mask, 2, 3);

        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.values, vec![1.0, 3.0, 6.0]);
        assert_eq!(csr.col_indices, vec![0, 2, 2]);
    }
}
