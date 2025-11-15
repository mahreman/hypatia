# Hypatia Kurulum Rehberi

Bu rehber, Hypatia projesinin modül yapısını ve kurulum adımlarını açıklar.

## 1. Modül Yapısı

Hypatia'nın iki katmanlı bir yapısı vardır:

### Rust Binary Modül: `_hypatia_core` (underscore'lı)
- PyO3 ile derlenmiş Rust kodu
- Maturin ile build ediliyor
- `.so` dosyası olarak derleniyor

### Python Paketi: `hypatia_core` (underscore'sız)
- Python tarafında kullanılan ana paket
- `_hypatia_core`'dan semboller import ediyor
- Import edilince otomatik olarak `register_backend()` çağrılıyor

### Nasıl Çalışır?

```python
# hypatia_core/__init__.py

# 1) Rust modülünden semboller alınıyor
from _hypatia_core import (
    compile_fx_graph,
    optimize_ast,
    parse_expr,
    is_equivalent,
    Symbol,
    PyMultiVector2D,
    PyMultiVector3D,
    HypatiaError,
    set_log_level,
)

import torch
import warnings

def hypatia_backend(gm, example_inputs):
    """Hypatia compiler backend for torch.compile()"""
    module_info_map = {}
    print(f"[Hypatia] Compiling graph with {len(list(gm.graph.nodes))} nodes")

    try:
        optimized_gm = compile_fx_graph(gm, example_inputs, module_info_map)
        return optimized_gm
    except Exception as e:
        warnings.warn(...)
        return gm

def register_backend():
    """Register Hypatia backend with PyTorch"""
    torch._dynamo.register_backend(name="hypatia", compiler_fn=hypatia_backend)
    ...

# 2) Import edilince backend otomatik kaydediliyor
register_backend()
```

**Özet:**
- Rust'ın ürettiği .so dosyası `_hypatia_core.so` olmalı
- Python kodu her zaman `import hypatia_core` kullanmalı, doğrudan `_hypatia_core` değil
- `hypatia_core` import edildiğinde otomatik olarak `register_backend()` çalışıp torch._dynamo içine "hypatia" backend'ini kaydediyor

## 2. Proje Layout

```
hypatia/
  pyproject.toml         # Python paket tanımı
  hypatia_core/          # Rust crate + Python paket
    Cargo.toml
    src/...
    hypatia_core/        # Python package dir
      __init__.py
    verify_setup.py      # Kurulum doğrulama scripti
    tests/
      mlp_safety_test.py
      mlp_perf_test.py
      test_backend_registration.py
      debug_backend.py
```

## 3. Geliştirici Ortamı Kurulumu

### Minimum Kurulum Adımları

1. **Sanal ortam oluştur**
   ```bash
   cd hypatia/hypatia_core
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # veya
   .venv\Scripts\activate  # Windows
   ```

2. **Rust tarafını derle**

   **Yöntem 1: Maturin ile (önerilen)**
   ```bash
   pip install maturin
   maturin develop --release
   ```

   **Yöntem 2: Manuel cargo build**
   ```bash
   cargo build --release
   # Üretilen .so dosyasını Python'ın göreceği yere kopyala
   cp target/release/lib_hypatia_core.so hypatia_core/_hypatia_core.so
   ```

3. **Python paketini editable olarak kur**
   ```bash
   cd ..  # Repo kök dizinine dön
   pip install -e .
   ```

4. **Kurulumu doğrula**
   ```bash
   python hypatia_core/verify_setup.py
   ```

### Doğrulama Test Script

```python
import torch
import hypatia_core

print("hypatia_core module:", hypatia_core.__file__)
print("Backends:", torch._dynamo.list_backends())
print("'hypatia' in backends:", "hypatia" in torch._dynamo.list_backends())
```

**Beklenen çıktı:**
```
✅ Hypatia backend registered successfully
   Usage: torch.compile(model, backend='hypatia')
   ✓ Backend confirmed in: [..., 'hypatia', ...]
```

## 4. Sorun Giderme

### Sorun 1: `ModuleNotFoundError: hypatia_core`

**Sebep:** Python'ın `hypatia_core` paketini bulamaması

**Çözüm:**
1. `pip install -e .` komutunu çalıştırdığınızdan emin olun
2. `PYTHONPATH` ayarlarını kontrol edin
3. Doğru çalışma dizininde olduğunuzdan emin olun

```bash
# Kurulumu kontrol et
pip list | grep hypatia

# Eğer listede yoksa:
pip install -e .
```

### Sorun 2: `Invalid backend: 'hypatia'`

**Sebep:** Backend kayıt olmuyor veya `register_backend()` çağrılmıyor

**Debug adımları:**

1. **Backend'in kayıtlı olup olmadığını kontrol et:**
   ```python
   import hypatia_core
   import torch

   print("FILE:", hypatia_core.__file__)
   print("BACKENDS:", torch._dynamo.list_backends())
   ```

2. **Manuel kayıt dene:**
   ```python
   import hypatia_core
   hypatia_core.register_backend()
   print(torch._dynamo.list_backends())
   ```

3. **Debug script çalıştır:**
   ```bash
   python hypatia_core/tests/debug_backend.py
   ```

**Güvenli kullanım (test dosyalarında):**
```python
import hypatia_core  # imports Rust bindings
hypatia_core.register_backend()  # ensure backend is registered

# Artık güvenle kullanabilirsiniz:
model = torch.compile(your_model, backend="hypatia")
```

### Sorun 3: `ImportError: _hypatia_core`

**Sebep:** Rust binary modülü derlenmemiş veya yanlış konumda

**Çözüm:**
```bash
cd hypatia_core
maturin develop --release

# veya manuel:
cargo build --release
cp target/release/lib_hypatia_core.so hypatia_core/_hypatia_core.so
```

## 5. Test Çalıştırma

### Safety Tests (Doğruluk Testleri)
```bash
python hypatia_core/tests/mlp_safety_test.py
```

Testler:
- Backend registration check
- Output correctness (Eager vs Hypatia)
- Multiple forward pass consistency
- Batch size variations

### Performance Tests (Performans Testleri)
```bash
python hypatia_core/tests/mlp_perf_test.py
```

Testler:
- Eager vs Hypatia performance comparison
- Batch size scaling
- Memory usage (CUDA only)

### Backend Registration Test
```bash
python hypatia_core/tests/test_backend_registration.py
```

## 6. Kullanım

### Temel Kullanım

```python
import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend

# Model tanımı
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).eval()

# Hypatia ile compile et
compiled_model = torch.compile(model, backend="hypatia")

# Kullan
x = torch.randn(128, 256)
output = compiled_model(x)
```

### Güvenli Kullanım (Backend Garantili)

```python
import torch
import hypatia_core

# Backend'i açıkça kaydet
hypatia_core.register_backend()

# Kayıtlı olduğunu doğrula
assert "hypatia" in torch._dynamo.list_backends()

# Compile et
compiled_model = torch.compile(model, backend="hypatia")
```

## 7. Geliştirme İpuçları

### Rust kodunu değiştirdikten sonra
```bash
maturin develop --release
# veya
cargo build --release && cp target/release/lib_hypatia_core.so hypatia_core/_hypatia_core.so
```

### Python kodunu değiştirdikten sonra
- `hypatia_core/__init__.py` değişiklikleri için: Python runtime'ı yeniden başlatın
- Test dosyaları için: Sadece tekrar çalıştırın

### Log seviyesini ayarlama
```python
import hypatia_core

hypatia_core.set_log_level("DEBUG")  # veya "INFO", "WARN", "ERROR"
```

## 8. Sık Sorulan Sorular

**S: Neden `_hypatia_core` ve `hypatia_core` ikisi de var?**
- `_hypatia_core`: Rust binary modülü (low-level)
- `hypatia_core`: Python wrapper paketi (high-level, kullanıcı dostu)
- Kullanıcılar her zaman `hypatia_core` import etmeli

**S: Backend neden otomatik kayıt olmuyor?**
- `hypatia_core/__init__.py` import edilmemiş olabilir
- `__init__.py` içinde exception oluşmuş olabilir (sessizce düşer)
- Manuel `hypatia_core.register_backend()` çağrısı güvenli bir çözümdür

**S: Maturin mi yoksa cargo build mi kullanmalıyım?**
- Geliştirme için: `maturin develop` (daha kolay)
- Production build için: `cargo build --release` (daha fazla kontrol)

**S: Test dosyaları nerede?**
- Safety tests: `hypatia_core/tests/mlp_safety_test.py`
- Performance tests: `hypatia_core/tests/mlp_perf_test.py`
- Debug tools: `hypatia_core/tests/debug_backend.py`

## 9. Destek

Sorun yaşarsanız:
1. `python hypatia_core/verify_setup.py` çalıştırın
2. `python hypatia_core/tests/debug_backend.py` çalıştırın
3. Hata mesajlarını ve çıktıları paylaşın
