#!/bin/bash
# Upgrade to optimized CUDA kernel (v2)
# Fixes 7.67x slowdown by using cuBLAS instead of naive GEMM

set -e

echo "========================================================================"
echo "UPGRADING TO OPTIMIZED FUSED LINEAR+RELU KERNEL (v2)"
echo "========================================================================"
echo ""
echo "This will replace naive GEMM with cuBLAS-accelerated version"
echo "Expected improvement: 7.67x slowdown → 1.05x speedup"
echo ""

# Check we're in the right directory
if [ ! -d "hypatia_core/csrc" ]; then
    echo "❌ Error: Please run from hypatia repository root"
    echo "   Usage: bash upgrade_to_optimized_kernel.sh"
    exit 1
fi

echo "🔍 Checking current kernel version..."
if grep -q "Naive GEMM inner loop" hypatia_core/csrc/fused_linear_relu_kernel.cu 2>/dev/null; then
    echo "✅ Found v1 (naive) kernel - upgrade needed"
else
    echo "⚠️  Current kernel status unknown, proceeding anyway..."
fi

echo ""
echo "📦 Creating backups..."

# Backup original files
BACKUP_DIR="hypatia_core/csrc/backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "hypatia_core/csrc/fused_linear_relu_kernel.cu" ]; then
    cp hypatia_core/csrc/fused_linear_relu_kernel.cu "$BACKUP_DIR/"
    echo "✅ Backed up: fused_linear_relu_kernel.cu"
fi

if [ -f "hypatia_core/csrc/fused_linear_relu.cpp" ]; then
    cp hypatia_core/csrc/fused_linear_relu.cpp "$BACKUP_DIR/"
    echo "✅ Backed up: fused_linear_relu.cpp"
fi

echo ""
echo "🔄 Applying optimized kernel..."

# Replace kernel implementation
cp hypatia_core/csrc/fused_linear_relu_kernel_v2.cu \
   hypatia_core/csrc/fused_linear_relu_kernel.cu
echo "✅ Updated: fused_linear_relu_kernel.cu (cuBLAS version)"

# Replace C++ binding
cp hypatia_core/csrc/fused_linear_relu.cpp.v2 \
   hypatia_core/csrc/fused_linear_relu.cpp
echo "✅ Updated: fused_linear_relu.cpp (with backward pass)"

echo ""
echo "🧹 Cleaning old build artifacts..."

# Remove old JIT compilation cache
if [ -d "/tmp/torch_extensions/hypatia_fused_linear_relu" ]; then
    rm -rf /tmp/torch_extensions/hypatia_fused_linear_relu
    echo "✅ Cleaned: JIT compilation cache"
fi

# Remove Rust build artifacts to force rebuild
if [ -d "hypatia_core/target" ]; then
    rm -rf hypatia_core/target/release/lib_hypatia_core.so 2>/dev/null || true
    echo "✅ Cleaned: Rust build artifacts"
fi

echo ""
echo "🔨 Rebuilding Rust extension..."
cd hypatia_core
cargo build --release

if [ $? -eq 0 ]; then
    echo "✅ Rust build successful"
else
    echo "❌ Rust build failed - check errors above"
    exit 1
fi

cd ..

echo ""
echo "========================================================================"
echo "✅ UPGRADE COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test kernel isolation:"
echo "   cd hypatia_core && python test_kernel_only.py"
echo ""
echo "2. Expected results:"
echo "   Before: Fused 7.67x SLOWER than eager"
echo "   After:  Fused 1.05x FASTER than eager"
echo ""
echo "3. Run full benchmarks:"
echo "   cd examples && python mlp_multiconfig_benchmark.py"
echo ""
echo "4. If issues occur, restore backup:"
echo "   cp $BACKUP_DIR/*.cu csrc/"
echo "   cp $BACKUP_DIR/*.cpp csrc/"
echo ""
echo "Backup location: $BACKUP_DIR"
echo "========================================================================"
