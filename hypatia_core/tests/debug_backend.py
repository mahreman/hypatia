#!/usr/bin/env python3
"""
Hypatia Backend Registration Debug Script
Bu script backend registration sorununu tespit eder
"""

import sys
import os

print("="*80)
print("HYPATIA BACKEND REGISTRATION DEBUG")
print("="*80)

# 1. Python path kontrolü
print("\n1. Python Path:")
for path in sys.path[:5]:
    print(f"   - {path}")

# 2. hypatia_core modülünü import etmeyi dene
print("\n2. Importing hypatia_core...")
try:
    import hypatia_core
    print("   ✅ hypatia_core imported successfully")

    # Module path
    if hasattr(hypatia_core, '__file__'):
        print(f"   Module location: {hypatia_core.__file__}")

    # Version
    if hasattr(hypatia_core, '__version__'):
        print(f"   Version: {hypatia_core.__version__}")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# 3. torch._dynamo kontrolü
print("\n3. Checking torch._dynamo...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    if hasattr(torch, '_dynamo'):
        print("   ✅ torch._dynamo exists")
        
        if hasattr(torch._dynamo, 'register_backend'):
            print("   ✅ torch._dynamo.register_backend exists")
        else:
            print("   ❌ torch._dynamo.register_backend NOT FOUND")
            print("   → Upgrade to PyTorch 2.0+")
            
        if hasattr(torch._dynamo, 'list_backends'):
            backends = torch._dynamo.list_backends()
            print(f"   Registered backends: {backends}")
            
            if 'hypatia' in backends:
                print("   ✅ 'hypatia' backend is registered!")
            else:
                print("   ❌ 'hypatia' backend NOT registered")
        else:
            print("   ❌ torch._dynamo.list_backends NOT FOUND")
    else:
        print("   ❌ torch._dynamo module not found")
        
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")
    sys.exit(1)

# 4. hypatia_backend fonksiyonunu kontrol et
print("\n4. Checking hypatia_backend function...")
if hasattr(hypatia_core, 'hypatia_backend'):
    print("   ✅ hypatia_backend function exists")
else:
    print("   ❌ hypatia_backend function NOT FOUND")
    print("   → Check __init__.py implementation")

# 5. Manuel registration denemesi
print("\n5. Attempting manual registration...")
try:
    if hasattr(hypatia_core, 'register_backend'):
        print("   Calling hypatia_core.register_backend()...")
        hypatia_core.register_backend()
    elif hasattr(hypatia_core, 'hypatia_backend'):
        print("   Calling torch._dynamo.register_backend manually...")
        torch._dynamo.register_backend(name="hypatia", compiler_fn=hypatia_core.hypatia_backend)
        print("   ✅ Manual registration successful")
    else:
        print("   ❌ Cannot register - no backend function found")
        
    # Re-check backends
    backends = torch._dynamo.list_backends()
    if 'hypatia' in backends:
        print(f"   ✅ SUCCESS! 'hypatia' now in backends: {backends}")
    else:
        print(f"   ❌ FAILED! 'hypatia' still not in backends: {backends}")
        
except Exception as e:
    print(f"   ❌ Registration failed: {e}")
    import traceback
    traceback.print_exc()

# 6. __init__.py dosyasını bul
print("\n6. Locating __init__.py...")
hypatia_init = None
if hasattr(hypatia_core, '__file__'):
    module_dir = os.path.dirname(hypatia_core.__file__)
    init_path = os.path.join(module_dir, '__init__.py')
    if os.path.exists(init_path):
        print(f"   ✅ Found: {init_path}")
        hypatia_init = init_path

        # İlk 20 satırı göster
        print("\n   First 20 lines:")
        with open(init_path, 'r') as f:
            for i, line in enumerate(f, 1):
                if i <= 20:
                    print(f"   {i:2d}: {line.rstrip()}")
    else:
        print(f"   ❌ Not found at: {init_path}")
else:
    print("   ❌ Cannot determine module location")

print("\n" + "="*80)
print("DIAGNOSIS:")

if 'hypatia' in torch._dynamo.list_backends():
    print("✅ Backend is registered - everything OK!")
else:
    print("❌ Backend NOT registered. Possible causes:")
    print("   1. __init__.py missing torch._dynamo.register_backend() call")
    print("   2. __init__.py has import errors (silent failure)")
    print("   3. __init__.py in wrong location")
    print("\nRECOMMENDED FIX:")
    print("   Run this to manually register:")
    print("   >>> import hypatia_core")
    print("   >>> import torch")
    print("   >>> torch._dynamo.register_backend('hypatia', hypatia_core.hypatia_backend)")

print("="*80)