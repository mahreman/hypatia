# Hypatia: E-Graph Esitlik Doygunlugu ile Donanim-Farkindali Derin Ogrenme Derleyicisi

**Ender Eryol**

*Bagimsiz Arastirma*

---

## Ozet

Bu calismada, sinir agi hesaplama grafiklerinde optimal operator kaynastirma kaliplarini kesfetmek icin E-Graph esitlik doygunlugu (equality saturation) kullanan, donanim-farkindali bir PyTorch sembolik derleyicisi olan Hypatia'yi sunuyoruz. Geleneksel kalip esleme derleyicilerinin (TVM, XLA) acgozlu ve sabit sirali yeniden yazma kurallarinin aksine, Hypatia cebirsel olarak esdeger tum temsilleri es zamanli olarak arastirir ve ozel maliyet fonksiyonu ile en dusuk maliyetli varyanti cikarir.

Sistem, `torch.compile` arka ucu olarak entegre olur ve sembolik optimizasyonu Triton cekirdek olusturma ile zincirleyerek iki fazli bir derleme hatti olusturur.

Hypatia'yi Qwen2.5-0.5B (494M parametre) dahil uretim modelleri uzerinde degerlendiriyoruz ve sunlari gosteriyoruz: (1) E-graph doygunlugu 50 operatore kadar grafiklerde <1ms'de tamamlanir; (2) kaynastirilmis dikkat cekirdekleri PyTorch'un moduler uygulamasina gore 1.6-16.6x hizlanma saglar; (3) INT4 blok nicemlemesi %99.6 kosinüs benzerligi ile 5.3-6.4x sikistirma saglar; (4) tam GPU hatti (FP16 + torch.compile) RTX 4070 Laptop GPU'da CPU FP32 taban cizgisine gore 164x hizlanma elde eder.

**Anahtar Kelimeler:** E-graph, esitlik doygunlugu, sinir agi derleyicisi, operator kaynastirma, nicemleme, PyTorch, torch.compile

---

## 1. Giris

Buyuk dil modellerinin konuslandirilmasi iki temel darbogazla karsi karsiyadir: hesaplama maliyeti ve bellek kapasitesi. 7B parametreli bir model FP32'de 28 GB gerektirir ve tek bir ileri gecis icin yaklasik 14 TFLOP/s'ye ihtiyac duyar. Operator kaynastirma, nicemleme ve donanima ozgu cekirdek uretimi bu darbogazlari gidermenin temel teknikleridir, ancak mevcut cozumler ya manuel mudahale (elle yazilmis CUDA cekirdekleri), farkli bir cerceve'ye model aktarimi (TVM, ONNX Runtime) ya da acgozlu kalip esleme nedeniyle suboptimal kaynastirma (TorchInductor) gerektirir.

**E-Graph Esitlik Doygunlugu** [Willsey ve dig., 2021] temel olarak farkli bir yaklasim sunar: yeniden yazma kurallarini sabit bir sirada uygulamak ve her karara bagli kalmak yerine, bir E-graph tum esdeger ifadeleri ayni anda kompakt bir sekilde temsil eder. Bir doygunluk asamasi, yeni esdegerlik bulunmayana kadar tum yeniden yazma kurallarini uygular, ardindan bir cikartma asamasi maliyet fonksiyonuna gore optimal temsili secer.

Hypatia'nin katkilari:

1. **E-Graph Sinir Agi IR'si**: Standart sinir agi islemleri, geometrik cebir primitifleri ve kaynastirilmis cekirdek varyantlarini destekleyen 25+ operatorlu S-ifadesi dili (HypatiaLang).

2. **Iki Fazli Derleme Hatti**: E-graph yapisal optimizasyonu (Rust) ardindan torch.compile cekirdek optimizasyonu (Triton), tek bir `torch.compile(model, backend='hypatia')` cagrisinda cebirsel ve donanim duzeyi kaynastirmayi birlestirerek.

3. **Donanim-Farkindali Oto-Ayarlama**: GPU hesaplama yetenegine, Tensor Core nesline, bellek bant genisligine ve model ozelliklerine (parametre sayisi, mimari tipi, FLOPs) dayali otomatik strateji secimi.

4. **Kapsamli Degerlendirme**: Gercek uretim modelleri (Qwen2.5-0.5B) uzerinde CPU ve GPU'da karsilastirmali degerler.

---

## 2. Sistem Mimarisi

### 2.1 Genel Bakis

```
                    PyTorch Modeli
                         |
                    torch.compile(backend="hypatia")
                         |
                  +------+------+
                  | FX Graf     |
                  | Yakalama    |
                  +------+------+
                         |
              +----------+----------+
              |  Faz 1: E-Graph     |
              |  (Rust / egg)       |
              |                     |
              |  FX -> S-ifadesi    |
              |  37 yeniden yazma   |
              |  Maliyet cikarma    |
              |  S-ifadesi -> FX    |
              +----------+----------+
                         |
              +----------+----------+
              |  Faz 2: Triton      |
              |  (torch.compile)    |
              |                     |
              |  Cekirdek kaynast.  |
              |  Bellek birlestirme |
              |  GPU kod uretimi    |
              +----------+----------+
                         |
                  Optimize Model
```

### 2.2 Yeniden Yazma Kurallari

Hypatia, cikarim modu icin 37 yeniden yazma kurali tanimlar:

**Operator Kaynastirma (12 kural)**
```
(relu (linear ?w ?b ?x))          => (fused_linear_relu ?w ?b ?x)
(gelu (linear ?w ?b ?x))          => (fused_gelu_mlp ?w ?b ?x)
(linear ?w2 ?b2 (mish (linear ?w1 ?b1 ?x)))
                                   => (fused_mish_mlp ?w1 ?b1 ?w2 ?b2 ?x)
(attention ?q ?k ?v)              => (sdpa ?q ?k ?v)
```

**Cebirsel Sadelestirme (15 kural)**
```
(add ?x 0)       => ?x            ; toplama birim elemani
(mul ?x 1)       => ?x            ; carpma birim elemani
(relu (relu ?x)) => (relu ?x)     ; idempotent ReLU
```

### 2.3 Maliyet Fonksiyonu

| Dugum Tipi | Maliyet |
|-----------|---------|
| Degisken, sabit | 1 |
| Basit operator (relu, add) | 10 |
| Karmasik operator (linear, conv2d) | 100 |
| Kaynastirilmis operator (fused_linear_relu) | 80 |

Bu, kaynastirmayi tesvik eder: `fused_linear_relu` (80) vs `linear + relu` (110).

### 2.4 GPU Gonderim Zinciri

4 katmanli geri donus zinciri:

1. **Ozel CUDA Uzantilari**: cuBLAS destekli kaynastirilmis cekirdekler
2. **torch.compile + Triton**: GPU oto-ayarli cekirdek uretimi
3. **Rust Yerel**: CPU optimize GEMM, sifir Python yuku
4. **Eager PyTorch**: Degistirilmemis geri donus

---

## 3. Oto-Ayarlama Sistemi

### 3.1 Hizli Ayarlama (Quick Tune, <200ms)

Karar agaci:
- **Model boyutu**: Kucuk (<1M) -> yalnizca kaynastirma; Orta (1-50M) -> yerel Rust; Buyuk (>50M) -> nicemleme
- **Mimari**: Transformer tespiti -> SDPA kaynastirma ile transformer modu
- **Donanim**: GPU mevcut -> karma hassasiyet etkinlestir (Ampere+ ise BF16, degilse FP16)

### 3.2 Olcum Tabanli Ayarlama (Benchmark Tune, 5-30s)

Hizli ayarlama analizine dayali aday strateji listesi olusturur, her aday icin gercek cikarim suresi olcer. En dusuk gecikmeli stratejiyi secer.

### 3.3 Sonuclar

Qwen2.5-0.5B (494M parametre, transformer mimarisi) icin:
- **Karar suresi**: 170ms
- **Secilen strateji**: Transformer (Rust-yerel blok)
- **Hassasiyet**: BF16 (Ada Lovelace tensor cekirdekleri tespit edildi)

---

## 4. Deneysel Degerlendirme

### 4.1 Donanim

| Bilesen | Ozellik |
|---------|---------|
| CPU | Intel Core i7-12700H (16C/16T) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU Mimarisi | Ada Lovelace (SM 8.9) |
| VRAM | 8 GB GDDR6 |
| SM Sayisi | 36 |
| GPU Saat | 3105 MHz (boost) |
| CUDA Cekirdek | 4608 (128/SM x 36 SM) |
| Tensor Cekirdek | 4. Nesil (Ada Lovelace) |
| Tepe FP32 | 28.6 TFLOPS |
| Tepe FP16 | 57.2 TFLOPS (Tensor Core ile) |
| Bellek Bant Genisligi | 504 GB/s |
| PyTorch | 2.6.0+cu124 |

### 4.2 E-Graph Doygunluk Performansi

| Ifade | Onceki Dugum | Sonraki Dugum | Kaynastirma | Doygunluk Suresi |
|-------|-------------|--------------|-------------|-----------------|
| `(relu (linear w b x))` | 5 | 4 | 1 (LinearReLU) | 0.880 ms |
| 2 katmanli MLP | 9 | 7 | 2 (LinearReLU) | 0.591 ms |
| 3 katmanli MLP | 13 | 10 | 3 (LinearReLU) | 0.411 ms |
| Artik blok | 7 | 6 | 1 (LinearReLU) | 0.364 ms |
| Tam dikkat | 12 | 4 | 1 (FusedAttention) | 0.381 ms |
| Mish MLP | 8 | 4 | 1 (FusedMishMLP) | ~0.5 ms |
| attention -> SDPA | 4 | 2 | 1 (SDPA) | ~0.4 ms |

**Kilit gozlem**: Daha derin aglarda doygunluk suresi *azalir* (1 katman: 0.88ms -> 3 katman: 0.41ms). E-graph'in birlesim-bul (union-find) yapisi, benzer alt ifadeler arasinda tekrarlanan kalip eslesme maliyetini amorte eder.

### 4.3 Kaynastirilmis Dikkat Cekirdegi Performansi (CPU)

| Konfigürasyon | Parametre | PyTorch MHA | Hypatia Fused | Hizlanma |
|--------------|----------|-------------|--------------|---------|
| d=64, 4 bas, seq=8 | 16K | 81.51 ms | 4.91 ms | **16.6x** |
| d=128, 8 bas, seq=16 | 65K | 120.80 ms | 56.93 ms | **2.1x** |
| d=256, 8 bas, seq=32 | 262K | 143.79 ms | 90.12 ms | **1.6x** |

### 4.4 Uctan Uca: Qwen2.5-0.5B Karsilastirmasi

**Model**: Qwen2.5-0.5B (494M parametre, 24 katman, gizli=896, 14 dikkat basi)
**Giris**: batch=1, seq_len=128
**Tahmini FLOPs**: 126.5 GFLOPs / cikarim

| Strateji | Gecikme | CPU FP32'ye Gore | Kategori |
|----------|---------|-----------------|----------|
| CPU FP32 (vanilya PyTorch) | 1449 ms | 1.0x | Taban |
| CPU INT8 Dinamik Nicemlenmis | 793 ms | 1.8x | Nicemlenmis |
| GPU FP32 | 33.8 ms | 42.9x | GPU |
| GPU FP16 (Tensor Core) | 31.4 ms | 46.2x | GPU |
| GPU BF16 | 49.7 ms | 29.1x | GPU |
| **GPU FP16 + torch.compile** | **8.9 ms** | **163.7x** | Derlenmis |
| MLP Blok FP32 (Rust yerel) | 0.98 ms | 1477.8x | Rust |
| **MLP Blok INT4 (Hypatia)** | **0.63 ms** | **2289.4x** | Nicemlenmis |

**Sonuc Analizi:**

1. **GPU FP16 + torch.compile (8.9ms, 164x)**: En iyi tam-model stratejisi. torch.compile'in max-autotune modu, epilog kaynastirma ve bellek birlestirme ile optimize Triton cekirdekleri uretir. Vanilya GPU FP32'ye gore 3.8x iyilesme onemli cekirdek duzeyi optimizasyonunu gosterir.

2. **GPU BF16'nin FP32'den yavas olmasi (49.7ms vs 33.8ms)**: Beklentiye aykiri ancak aciklanabilir. Kucuk batch boyutlari ve kisa dizilerde, BF16 hassasiyet donusum yuku (FP32 -> BF16 donusum, BF16 -> FP32 birikimi) bant genisligi tasarrufunu aser. Bu etki literaturde iyi belgelenmistir ve buyuk batch boyutlarinda ortadan kalkar.

3. **MLP Blok INT4 (0.63ms, 2289x)**: Bu, tek bir MLP blogunu olcer (tam modeli degil), ayni blokta FP32'ye gore 1.55x hizlanma ile 6.4x sikistirma basarir.

4. **CPU INT8 Dinamik (793ms, 1.8x)**: PyTorch'un yerlesik dinamik nicemleme'si, INT8 GEMM icin VNNI talimatlari kullanarak CPU'da orta duzeyde hizlanma saglar.

**Token uretim testi:**
- Bilgi istemi: "The future of AI is"
- Cikti: "The future of AI is in the hands of the people..."
- Optimizasyon hattindan sonra modelin tutarli metin urettigi dogrulanmistir.

### 4.5 INT4 Blok Nicemleme

| Katman Boyutu | FP32 Boyut | INT4 Boyut | Sikistirma | Kosinüs Benzerligi |
|--------------|-----------|-----------|-----------|-------------------|
| 256x128 | 131 KB | 24 KB | 5.3x | 0.9965 |
| 1024x512 | 2.0 MB | 384 KB | 5.3x | 0.9968 |
| 2048x1024 | 8.0 MB | 1.5 MB | 5.3x | 0.9971 |
| Qwen MLP blok | 4.5 MB | 0.7 MB | 6.4x | >0.995 |

---

## 5. Sayisal Kararlilik Garantileri

### 5.1 Kayan Noktanin Birlesme Ozelliginin Olmamasi

Kayan nokta aritmetigi birlesme ozelligine sahip degildir: `(a + b) + c != a + (b + c)` ULP (Son Birim Yerinde) duzeyinde. Operator kaynastirma dogasi geregi islemleri yeniden siralar, bu nedenle **orijinal ve kaynastirilmis ciktilar arasinda bit duzeyinde kimlk garanti edilemez ve beklenmemelidir**.

### 5.2 Dogrulama Cercevesi

| Mod | Maks Mutlak Fark | Kosinüs Esik | Kullanim |
|-----|-----------------|-------------|---------|
| Siki (Strict) | 1e-5 | 0.99999 | Gelistirme, hata ayiklama |
| Yumusak (Soft) | 1e-3 | 0.999 | Uretim cikarimi |
| Kapali (Off) | - | - | Maksimum performans |

### 5.3 Ampirik Gozlemler

Nicemlenmemis optimizasyonlar (yalnizca kaynastirma) icin tutarli olarak:
- Maks mutlak fark: < 10^-6 (FP32)
- Kosinüs benzerligi: > 0.999999

Nicemlenmis optimizasyonlar icin:
- INT8: Maks fark < 10^-2, kosinüs > 0.999
- INT4: Maks fark < 10^-1, kosinüs > 0.995

---

## 6. Sinirliliklar ve Gelecek Calisma

1. **E-Graph Bellek Olceklenmesi**: Cok buyuk grafiklerde (>1000 dugum) bellek tuketimi onemli olcude artar. Artan doygunluk stratejilerinin arastirilmasi gerekmektedir.

2. **CPU Seyrek GEMM**: CSR seyrek GEMM, yalnizca kucuk katmanlarda >%90 seyreklikte rekabetcidir. GPU seyrek GEMM (cuSPARSE) entegrasyonu planlanmaktadir.

3. **Dinamik Sekiller**: Mevcut E-graph optimizasyonu statik sekilleri varsayar. Dinamik dizi uzunluklari veya batch boyutlarina sahip modeller yeniden optimizasyon gerektirir.

4. **Gelecek Yonelimler**: Triton ozel cekirdekleri, dagitik derleme, nicemleme farkindali egitim (QAT), Apple Silicon / ARM destegi.

---

## 7. Sonuc

Hypatia, E-Graph esitlik doygunlugunun PyTorch ekosistemi icinde sinir agi derlemesi icin pratik ve etkili bir teknik oldugunu gostermektedir. Iki fazli hat (E-graph yapisal optimizasyonu + Triton cekirdek uretimi) cesitli model boyutlari ve donanim konfigürasyonlarinda onemli hizlanmalar elde eder.

Mevcut yaklasimlara gore temel avantajlar:
1. **Tamlik**: E-graph doygunlugu, kural seti icindeki tum esdeger temsilleri arastirir
2. **Sifir surtumus entegrasyon**: Dogrudan `torch.compile` arka ucu, model kodu degisikligi gerektirmez
3. **Donanim farkindalilik**: Oto-ayarlayici, stratejileri tespit edilen GPU yeteneklerine uyarlar
4. **Dogruluk garantileri**: Acik sayisal tolerans sinirlarıyla yapilandirilabilir semantik dogrulama

Sistem acik kaynaktir: [github.com/mahreman/hypatia](https://github.com/mahreman/hypatia)

---

## Referanslar

[1] Willsey, M. ve dig. (2021). egg: Fast and extensible equality saturation. *POPL*.

[2] Chen, T. ve dig. (2018). TVM: An automated end-to-end optimizing compiler for deep learning. *OSDI*.

[3] Jia, Z. ve dig. (2019). TASO: Optimizing deep learning computation with automatic generation of graph substitutions. *SOSP*.

[4] Yang, Y. ve dig. (2021). Equality saturation for tensor graph superoptimization. *MLSys*.

[5] Ansel, J. ve dig. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. *ASPLOS*.

[6] Williams, S. ve dig. (2009). Roofline: An insightful visual performance model for multicore architectures. *CACM*.

[7] Dettmers, T. ve dig. (2022). GPT3.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS*.

[8] Jacob, B. ve dig. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.

---

## Ek A: Test Suiti Ozeti

| Bilesen | Test | Kapsam |
|---------|------|--------|
| E-graph optimizer (Rust) | 155 | Yeniden yazma kurallari, maliyet cikarma, kaynastirma kaliplari |
| Profiler (Python) | 23 | Donanim tespiti, FLOPs tahmini, roofline |
| Oto-ayarlayici (Python) | 18 | Hizli ayarlama, karsilastirma ayarlama, konfigürasyon |
| Dashboard (Python) | 20 | HTML uretimi, kacis, grafik olusturma |
| Semantik dogrulama (Python) | 15 | Ifade dogrulama, model esdegers |
| Gorsellestime (Python) | 10 | DOT aktarimi, ASCII agac, HTML rapor |
| Arka uc entegrasyonu (Python) | 12 | torch.compile kaydi, FX koprusu |
| Kaynastirilmis moduller (Python) | 8 | CUDA uzantilari, GPU gonderim |
| **Toplam** | **~260** | |

## Ek B: Karsilastirmalarin Yeniden Uretilmesi

```bash
# 1. Bagimliliklari yukle
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install maturin transformers

# 2. Hypatia'yi derle
cd hypatia_core
maturin develop --release

# 3. Qwen2.5-0.5B karsilastirmasini calistir
python demos/demo_qwen.py

# 4. Test suitini calistir
cargo test                    # Rust (155 test)
python -m pytest tests/ -v    # Python (100+ test)
```
