#!/bin/bash
# Run ResNet18 face classifier inference

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path>"
    echo ""
    echo "Example:"
    echo "  bash scripts/run_resnet18.sh photo.jpg"
    exit 1
fi

# Set LibTorch environment variables
export LIBTORCH=$HOME/libtorch
export DYLD_LIBRARY_PATH=$HOME/libtorch/lib

# Run inference
cargo run --release --bin infer_resnet18 -- "$1"