#!/usr/bin/env python3
"""
Hypatia Python binding build script
"""

import subprocess
import sys
import os
import shutil
import glob

def build_hypatia():
    """Build Hypatia Python bindings"""
    print("Building Hypatia Python bindings...")
    
    # Rust kütüphanesini build et
    result = subprocess.run([
        "cargo", "build", "--release"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        return False
    
    # Shared library'yi bul
    lib_patterns = [
        "target/release/libhypatia_core*.so",
        "target/release/hypatia_core*.dll",
        "target/release/libhypatia_core*.dylib"
    ]
    
    lib_path = None
    for pattern in lib_patterns:
        matches = glob.glob(pattern)
        if matches:
            lib_path = matches[0]
            break
    
    if not lib_path:
        print("Shared library not found! Searched for:")
        for pattern in lib_patterns:
            print(f"  {pattern}")
        return False
    
    print(f"Found library: {lib_path}")
    
    # Python modülünü oluştur
    ext = os.path.splitext(lib_path)[1]
    module_name = "hypatia_core" + ext
    
    # Mevcut dizine kopyala
    shutil.copy(lib_path, module_name)
    print(f"Copied to: {module_name}")
    
    # Examples dizinine de kopyala (demo için)
    examples_module = os.path.join("examples", module_name)
    shutil.copy(lib_path, examples_module)
    print(f"Copied to: {examples_module}")
    
    print("Build successful! Now you can use:")
    print("  python3 examples/python_demo.py")
    
    return True

if __name__ == "__main__":
    if build_hypatia():
        print("\n✅ Hypatia successfully built!")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)