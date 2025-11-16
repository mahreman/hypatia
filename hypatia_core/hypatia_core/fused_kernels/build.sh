#!/bin/bash
# Build script for Hypatia CUDA fused kernels

set -e  # Exit on error

echo "========================================="
echo "Building Hypatia CUDA Fused Kernels"
echo "========================================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: CUDA (nvcc) not found!"
    echo "Please install CUDA toolkit first."
    exit 1
fi

# Check if PyTorch is available
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ ERROR: PyTorch not found!"
    echo "Please install PyTorch with CUDA support:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "📦 Building CUDA extension..."
echo ""

# Build the extension
python3 setup.py install

echo ""
echo "✅ Build complete!"
echo ""
echo "To test the extension, run:"
echo "  cd ../../examples"
echo "  python3 test_fused_linear_relu.py"
