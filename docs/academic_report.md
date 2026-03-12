# Hypatia: A Geometric Algebra-Aware Neural Network Compiler with E-Graph Equality Saturation

## Project Overview

Hypatia is a neural network compiler built as a Rust/Python hybrid system that uses e-graph equality saturation (via the `egg` crate) to discover and apply algebraic rewrites on computation graphs. The compiler operates on an S-expression-based intermediate representation (IR) called HypatiaLang, which supports 25+ operators including standard neural network operations, geometric algebra primitives, and fused kernel variants. The system integrates with PyTorch's `torch.compile` infrastructure as a custom backend, enabling transparent optimization of existing PyTorch models.

The core compilation pipeline works as follows: PyTorch FX graphs are converted into Hypatia's S-expression IR, the e-graph optimizer explores equivalent representations through algebraic rewrite rules, and the optimal variant (by a custom cost function minimizing node count and favoring fused operations) is extracted and reconstructed back into executable PyTorch code.

All benchmark results below were obtained on a Linux system running Python 3.11.14, PyTorch 2.10.0 (CPU), and NumPy 2.4.2.

---

## 1. E-Graph Equality Saturation Optimizer

The core of Hypatia is its e-graph-based optimizer, implemented in Rust using the `egg` (e-graphs good) library. The optimizer maintains a set of 37 rewrite rules for inference mode, covering operator fusions (Linear+ReLU → FusedLinearReLU, multi-head attention fusion), algebraic simplifications (dead code elimination, identity operations), and geometric algebra-specific rewrites.

### Benchmark Results

**Simple Linear+ReLU fusion:** The expression `(relu (linear w b x))` was rewritten to `(fused_linear_relu w b x)`. The fusion eliminates the separate ReLU activation by incorporating it directly into the linear operation's output computation, removing one intermediate tensor allocation and one kernel dispatch. Each optimization completed in 0.880 milliseconds on average, measured over 100 iterations (87.99 ms total).

**Two-layer MLP:** The expression `(relu (linear w2 b2 (relu (linear w1 b1 x))))` was rewritten to `(fused_linear_relu w2 b2 (fused_linear_relu w1 b1 x))`. Both Linear+ReLU pairs were independently fused. The optimizer recognized and applied the fusion pattern at each layer. Average time per optimization: 0.591 ms.

**Three-layer MLP:** The expression `(relu (linear w3 b3 (relu (linear w2 b2 (relu (linear w1 b1 x)))))))` was transformed to `(fused_linear_relu w3 b3 (fused_linear_relu w2 b2 (fused_linear_relu w1 b1 x)))`. All three Linear+ReLU pairs fused in a single e-graph saturation pass. Average time: 0.411 ms. The decreasing optimization time for deeper networks demonstrates that the e-graph saturates quickly once the fusion rules are known, and the search space does not grow proportionally with depth.

**Residual connection:** The expression `(add (relu (linear w b x)) x)` was rewritten to `(add x (fused_linear_relu w b x))`. The optimizer fused the Linear+ReLU pair while preserving the skip connection. This is significant for ResNet-style architectures where the residual path must remain intact. Average time: 0.364 ms.

**Full attention pattern:** The expression `(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))` was rewritten to `(fused_attention wq bq wk bk wv bv wo bo x x)`. This fusion collapses four separate linear projections (Q, K, V, output) and the attention computation into a single fused kernel, eliminating three intermediate tensor materializations. Average time: 0.381 ms.

---

## 2. torch.compile Backend Integration

Hypatia registers itself as a `torch.compile` backend, appearing alongside PyTorch's built-in backends (inductor, cudagraphs, etc.). When a model is compiled with `torch.compile(model, backend="hypatia")`, the FX graph is intercepted, converted to Hypatia's S-expression IR, optimized through the e-graph, and reconstructed into executable graph modules with fused operations.

### Benchmark Results

**Small MLP (128→64→32, 10,336 parameters):** Standard PyTorch completed 1000 forward passes in 14.72 ms, while Hypatia compiled completed the same in 37.19 ms, yielding a speed ratio of 0.40x. For small models, the Python-level dispatch overhead of the compiled backend dominates the actual computation time. The maximum output difference between the standard and compiled versions was 0.00 (bit-exact).

**Medium MLP (512→256→128→64, 172,480 parameters):** Standard PyTorch completed 1000 iterations in 1634.97 ms, while Hypatia compiled completed them in 1755.74 ms, for a speed ratio of 0.93x. As the model grows, the overhead becomes proportionally smaller. Output difference: 0.00 (bit-exact).

**Large MLP (1024→512→256→128→64, 697,280 parameters):** Standard PyTorch completed 1000 iterations in 1714.73 ms, while Hypatia compiled completed them in 1592.21 ms, for a speed ratio of 1.08x. At this scale, the fused Linear+ReLU operations begin to outweigh the compilation overhead, producing a net speedup. Output difference: 0.00 (bit-exact).

The key observation is that all compiled outputs were bit-exact identical to the original PyTorch outputs (max difference = 0.00), confirming that the optimization is semantically correct. The crossover point where Hypatia's fused operations begin to provide net speedup occurs at approximately 500K-700K parameters for CPU inference.

---

## 3. Mixed Precision (FP16/BF16)

Hypatia implements mixed-precision inference using two half-precision formats: IEEE 754 FP16 (16-bit floating point with 10-bit mantissa) and BF16 (Brain Float 16 with 7-bit mantissa, matching FP32's exponent range). Weights are stored in half-precision but computation is performed in FP32 (mixed-precision GEMM: weights are decompressed on-the-fly during matrix multiplication). The conversion uses F16C hardware instructions when available.

### Benchmark Results

**Small layer (128×64, 8,192 parameters):** FP32 storage requires 32,768 bytes, while both FP16 and BF16 require 16,384 bytes, a 50% memory saving. FP16 conversion introduces a maximum absolute error of 1.198×10⁻⁴ with an RMSE of 2.034×10⁻⁵. BF16 conversion has a higher maximum error of 9.743×10⁻⁴ and RMSE of 1.658×10⁻⁴, which is expected given BF16's reduced mantissa precision. The FP32 GEMM (1000 iterations) completed in 13.41 ms, while the FP16 mixed-precision GEMM took 18.07 ms.

**Medium layer (512×256, 131,072 parameters):** Memory savings remain at 50% (524,288 → 262,144 bytes). FP16 max error: 1.220×10⁻⁴, RMSE: 2.069×10⁻⁵. BF16 max error: 9.760×10⁻⁴, RMSE: 1.658×10⁻⁴. FP32 GEMM: 19.68 ms. FP16 mixed-precision GEMM: 331.51 ms.

**Large layer (1024×512, 524,288 parameters):** FP32: 2,097,152 bytes → FP16/BF16: 1,048,576 bytes (50% saving). FP16 max error: 1.221×10⁻⁴, RMSE: 2.074×10⁻⁵. BF16 max error: 9.765×10⁻⁴, RMSE: 1.655×10⁻⁴. FP32 GEMM: 17.99 ms. FP16 mixed-precision: 1345.40 ms.

**Very large layer (2048×1024, 2,097,152 parameters):** FP32: 8,388,608 bytes → FP16/BF16: 4,194,304 bytes. FP16 max error: 1.221×10⁻⁴, RMSE: 2.069×10⁻⁵. BF16 max error: 9.765×10⁻⁴, RMSE: 1.656×10⁻⁴.

The consistent FP16 maximum error of approximately 1.22×10⁻⁴ across all layer sizes confirms that the precision loss is bounded and predictable. The BF16 errors are approximately 8× larger than FP16 but still within acceptable bounds for inference. The memory savings are exactly 50% regardless of layer size. The mixed-precision GEMM is currently slower than native FP32 GEMM on CPU because the decompression overhead (half → float conversion per element) is not offset by reduced memory bandwidth on CPU; this approach is primarily beneficial for GPU inference where memory bandwidth is the bottleneck, and for deployment scenarios where model size reduction is the primary goal.

---

## 4. Sparse Tensor IR (CSR Format)

Hypatia implements Compressed Sparse Row (CSR) format for weight matrices and provides sparse-dense GEMM (SpMV/SpMM) operations. The system can analyze weight sparsity, convert dense matrices to CSR format, and perform sparse linear forward passes. This is particularly useful after weight pruning, where a significant fraction of weights are zero.

### Benchmark Results

**Small layer (128×64):**
At 0% sparsity (fully dense), dense PyTorch GEMM completed 1000 iterations in 15.32 ms while sparse Hypatia took 89.64 ms (0.17x). At 50% sparsity, dense: 13.92 ms, sparse: 46.14 ms (0.30x). At 80% sparsity, dense: 17.13 ms, sparse: 19.83 ms (0.86x). At 90% sparsity, dense: 13.48 ms, sparse: 11.35 ms (1.19x). At 95% sparsity, dense: 15.60 ms, sparse: 6.19 ms (2.52x). The crossover point where sparse computation becomes faster than dense occurs at approximately 85-90% sparsity for this layer size. All outputs maintained numerical agreement with the dense computation (max difference < 10⁻⁷).

**Medium layer (512×256):**
At 0% sparsity: dense 21.37 ms, sparse 1386.20 ms (0.02x). At 95% sparsity: dense 32.63 ms, sparse 74.31 ms (0.44x). For medium layers, the sparse format overhead is larger due to the CSR indirection cost scaling with the number of rows. Even at 95% sparsity, the sparse path does not achieve a speedup for this size on CPU.

**Large layer (1024×512):**
At 0% sparsity: dense 22.17 ms, sparse 5684.89 ms. At 95% sparsity: dense 25.15 ms, sparse 291.77 ms (0.09x). For large layers, the dense PyTorch GEMM benefits from highly optimized BLAS routines (MKL/OpenBLAS), and the sparse format's per-element overhead is significant.

The key finding is that sparse computation provides genuine speedup only for small-to-medium layers at very high sparsity levels (>90%). The primary benefit of the sparse IR is memory reduction: at 95% sparsity, a 1024×512 layer's CSR representation uses 320,224 bytes versus 2,097,152 bytes for dense storage (6.5× compression). The numerical accuracy of sparse GEMM is excellent, with maximum differences below 10⁻⁶ across all configurations.

---

## 5. Fused Multi-Head Attention

Hypatia implements a native fused attention kernel in Rust that performs Q/K/V projections, scaled dot-product attention, and output projection in a single function call, eliminating intermediate tensor materializations. The kernel computes: `Output = (softmax(QK^T / √d_k) · V) · W_o + b_o` where Q, K, V are projected from the input using separate weight matrices.

### Benchmark Results

**Small configuration (d_model=64, 4 heads, seq_len=8, 16,384 parameters):** PyTorch's `nn.MultiheadAttention` completed 500 iterations in 81.51 ms. Hypatia's fused attention completed the same in 4.91 ms, yielding a 16.59× speedup. This dramatic improvement comes from eliminating PyTorch's per-operation dispatch overhead, which dominates at small sizes.

**Medium configuration (d_model=128, 8 heads, seq_len=16, 65,536 parameters):** PyTorch MHA: 120.80 ms for 500 iterations. Hypatia fused: 56.93 ms. Speed ratio: 2.12×. The speedup decreases as the actual computation time grows relative to the dispatch overhead.

**Large configuration (d_model=256, 8 heads, seq_len=32, 262,144 parameters):** PyTorch MHA: 143.79 ms. Hypatia fused: 90.12 ms. Speed ratio: 1.60×. Even at this scale, the fused kernel maintains a meaningful advantage by avoiding intermediate allocations for Q, K, V projections.

The fused attention kernel consistently outperforms PyTorch's modular implementation across all tested configurations, with speedups ranging from 1.60× to 16.59×. The advantage is most pronounced for smaller dimensions where dispatch overhead constitutes a larger fraction of total execution time.

---

## 6. Semantic Validation

Hypatia includes a semantic validation system that verifies optimization correctness at two levels: (1) structural validation of S-expression rewrites (checking that all variables are preserved and the expression structure is sound), and (2) model output equivalence testing (running random inputs through original and optimized models and comparing outputs using max absolute difference and cosine similarity).

### Benchmark Results

**Identical model (cloned):** A 3-layer MLP (256→128→64→32) was deep-copied and both versions were validated against each other using 10 random test inputs. The maximum output difference was 0.00 (bit-exact), mean difference was 0.00, and cosine similarity was 1.0000000000. Validation completed in 2.35 ms.

**Slightly perturbed model (noise=10⁻⁶):** Gaussian noise with standard deviation 10⁻⁶ was added to all parameters of the cloned model. The maximum output difference was 9.00×10⁻⁶, mean difference was 2.73×10⁻⁶, and cosine similarity was 0.9999999995. The validation correctly passed with tolerance 10⁻⁴, confirming that tiny weight perturbations (as might occur during quantization or pruning) produce proportionally tiny output changes.

**Completely different model:** A model with identical architecture but different random weights was compared to the original. The maximum output difference was 5.20×10⁻¹ and cosine similarity was -0.0129. The validation correctly reported failure, demonstrating that the system can distinguish between semantically equivalent and non-equivalent models.

**E-graph optimization validation:** The expression `(relu (linear w b x))` was optimized to `(fused_linear_relu w b x)`. Structural validation confirmed that all variables (w, b, x) were preserved in the optimized expression, the node count was reduced by 1, and the FusedLinearReLU fusion was correctly identified.

---

## 7. Real Model Analysis (GPT-2, DistilBERT)

Hypatia was tested on real transformer architectures from HuggingFace to verify that its analysis tools work on production model structures.

### GPT-2 (2-layer, 128-dimensional embedding)

The model contains 655,872 parameters across 10 weight layers. In FP32 format, the model occupies 2.50 MB of memory; converting to FP16 reduces this to 1.25 MB, saving 1,281 KB (50%). The attention layers use HuggingFace's Conv1D format (transposed weight matrices) rather than standard nn.Linear, and Hypatia correctly handles both formats.

Layer-by-layer analysis showed that the natural sparsity of randomly initialized GPT-2 weights is approximately 0.0000 (all weights are nonzero after initialization). FP16 conversion error across all layers was consistently around 3.0×10⁻⁵, confirming that half-precision is safe for this architecture.

### DistilBERT (2-layer, 128-dimensional)

The model contains 409,600 parameters across 14 weight layers. FP32 memory: 1.56 MB, FP16 memory: 0.78 MB, saving 800 KB. DistilBERT uses standard nn.Linear layers. The word embedding layer showed a natural sparsity of 0.0010 (0.1% of values are exactly zero), while all other layers had zero natural sparsity. FP16 conversion errors were consistent at approximately 3.0×10⁻⁵ across all layers.

---

## 8. INT4 Block Quantization

Hypatia implements INT4 (4-bit integer) block quantization with configurable group size for aggressive model compression. Weights are quantized to 4-bit integers with per-group scale factors and zero points, then dequantized to FP32 during inference. This is implemented using Rayon parallelism and SIMD instructions for the dequantization step.

### Benchmark Results

**Small layer (256×128, 32,768 parameters):** FP32 storage: 131,072 bytes. INT4 storage: 24,576 bytes (5.3× compression). FP32 GEMM (500 iterations): 8.93 ms. INT4 quantized forward: 45.32 ms (0.20× speed ratio). Maximum output error: 2.574×10⁻¹. Cosine similarity between FP32 and INT4 outputs: 0.99653.

**Medium layer (1024×512, 524,288 parameters):** FP32: 2,097,152 bytes. INT4: 393,216 bytes (5.3× compression). FP32 GEMM: 14.75 ms. INT4 forward: 65.96 ms (0.22×). Max error: 5.443×10⁻¹. Cosine similarity: 0.99683.

**Large layer (2048×1024, 2,097,152 parameters):** FP32: 8,388,608 bytes. INT4: 1,572,864 bytes (5.3× compression). FP32 GEMM: 20.76 ms. INT4 forward: 96.37 ms (0.22×). Max error: 8.214×10⁻¹. Cosine similarity: 0.99707.

INT4 quantization achieves consistent 5.3× memory compression across all layer sizes. The cosine similarity above 0.996 indicates that while individual values may differ significantly (max errors in the 10⁻¹ range, which is expected for 4-bit quantization), the overall output direction is well preserved. The current CPU implementation of dequantize-then-GEMM is slower than native FP32 GEMM; the primary benefit is the 5.3× reduction in model storage and memory footprint, which is critical for deploying large models on memory-constrained devices.

---

## Additional Features

### Geometric Algebra Operations
Hypatia includes native Rust implementations of 2D and 3D geometric algebra operations (geometric product, rotors, batch rotations) with NumPy integration. These operations support both numeric and symbolic computation modes, enabling algebraic simplification of geometric transformations in the e-graph optimizer.

### Neuromorphic Computing
The system supports ANN-to-SNN (Artificial Neural Network to Spiking Neural Network) conversion using Leaky Integrate-and-Fire (LIF) neuron models, with energy estimation for neuromorphic hardware deployment.

### Visualization
Hypatia can export computation graphs as DOT format for Graphviz rendering, generate ASCII tree representations of expressions, and produce HTML optimization reports comparing original and optimized expression structures.

---

## Architecture Summary

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| E-graph optimizer | Rust (`egg` crate) | Equality saturation-based rewrite optimization |
| HypatiaLang IR | Rust (25+ operators) | S-expression intermediate representation |
| torch.compile backend | Python + Rust (PyO3) | FX Graph → S-expr → optimize → reconstruct |
| Sparse IR | Rust (CSR format) | Weight pruning, sparse GEMM |
| Mixed precision | Rust (F16C instructions) | FP16/BF16 storage with FP32 compute |
| INT4 quantization | Rust (Rayon + SIMD) | 4-bit block quantization |
| Fused attention | Rust (native kernel) | Q/K/V projection + attention in single call |
| Semantic validation | Rust + Python | Output equivalence verification |
| Geometric algebra | Rust (2D/3D) | Rotor-based transformations |
| Python bindings | PyO3 | Seamless Python/NumPy integration |

---

## Conclusion

Hypatia demonstrates that e-graph equality saturation is a viable approach for neural network compilation, providing provably correct operator fusions with sub-millisecond optimization times. The fused attention kernel achieves 1.6-16.6× speedup over PyTorch's modular implementation. Mixed-precision and INT4 quantization provide 50% and 81% memory reductions respectively, with high cosine similarity (>0.996) to full-precision outputs. The semantic validation system ensures that all optimizations preserve model correctness within configurable tolerances.
