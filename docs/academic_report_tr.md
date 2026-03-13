# Hypatia: Geometrik Cebir Destekli, E-Graph Eşitlik Doygunluğu Tabanlı Sinir Ağı Derleyicisi

## Proje Özeti

Hypatia, e-graph eşitlik doygunluğu (equality saturation) tekniğini kullanarak hesaplama grafikleri üzerinde cebirsel yeniden yazma kuralları keşfeden ve uygulayan, Rust/Python hibrit bir sinir ağı derleyicisidir. Derleyici, Rust'ın `egg` kütüphanesi üzerine inşa edilmiştir. Sistem, HypatiaLang adlı 25'ten fazla operatör içeren bir S-ifadesi (S-expression) tabanlı ara temsil (IR) kullanır. Bu operatörler arasında standart sinir ağı işlemleri, geometrik cebir primitifleri ve kaynaştırılmış (fused) çekirdek varyantları bulunmaktadır.

Hypatia, PyTorch'un `torch.compile` altyapısına özel bir arka uç (backend) olarak entegre olur ve mevcut PyTorch modellerinin şeffaf bir şekilde optimize edilmesini sağlar.

Temel derleme hattı şu şekilde çalışır: PyTorch FX grafikleri Hypatia'nın S-ifadesi IR'sine dönüştürülür, e-graph optimizer cebirsel yeniden yazma kuralları aracılığıyla eşdeğer temsilleri araştırır ve özel maliyet fonksiyonuna göre (düğüm sayısını minimize eden, kaynaştırılmış operasyonları tercih eden) optimal varyant çıkarılarak yeniden çalıştırılabilir PyTorch koduna dönüştürülür.

Aşağıdaki tüm benchmark sonuçları Linux sisteminde, Python 3.11.14, PyTorch 2.10.0 (CPU) ve NumPy 2.4.2 ile elde edilmiştir.

---

## 1. E-Graph Eşitlik Doygunluğu Optimizer

Hypatia'nın çekirdeği, Rust'ta `egg` (e-graphs good) kütüphanesi kullanılarak gerçeklenen e-graph tabanlı optimizer'dır. Optimizer, çıkarım modu için 37 yeniden yazma kuralı içerir. Bu kurallar operatör kaynaştırmalarını (Linear+ReLU → FusedLinearReLU, çok-başlı dikkat kaynaştırması), cebirsel sadeleştirmeleri (ölü kod eleme, birim işlemler) ve geometrik cebire özgü yeniden yazmaları kapsar.

### Benchmark Sonuçları

**Basit Linear+ReLU kaynaştırması:** `(relu (linear w b x))` ifadesi `(fused_linear_relu w b x)` olarak yeniden yazıldı. Bu kaynaştırma, ayrı ReLU aktivasyonunu doğrudan lineer işlemin çıktı hesaplamasına dahil ederek bir ara tensör tahsisini ve bir çekirdek gönderimini ortadan kaldırır. Standart PyTorch'ta bu işlem iki ayrı operasyon olarak yürütülürken (önce matris çarpımı + bias, sonra ReLU), Hypatia bunları tek bir birleşik çekirdeğe indirger. Her optimizasyon ortalama 0.880 milisaniyede tamamlandı (100 iterasyon üzerinden ölçüm: toplam 87.99 ms).

**İki katmanlı MLP:** `(relu (linear w2 b2 (relu (linear w1 b1 x))))` ifadesi `(fused_linear_relu w2 b2 (fused_linear_relu w1 b1 x))` olarak dönüştürüldü. Her iki Linear+ReLU çifti bağımsız olarak kaynaştırıldı. Standart PyTorch'ta 4 ayrı operasyon (2 lineer + 2 ReLU) olarak çalışan bu yapı, Hypatia ile 2 kaynaştırılmış operasyona indirildi. Optimizasyon başına ortalama süre: 0.591 ms.

**Üç katmanlı MLP:** `(relu (linear w3 b3 (relu (linear w2 b2 (relu (linear w1 b1 x)))))))` ifadesi `(fused_linear_relu w3 b3 (fused_linear_relu w2 b2 (fused_linear_relu w1 b1 x)))` olarak dönüştürüldü. Üç Linear+ReLU çiftinin tamamı tek bir e-graph doygunluk geçişinde kaynaştırıldı. Standart PyTorch'ta 6 ayrı operasyon, Hypatia ile 3 kaynaştırılmış operasyona indi. Ortalama süre: 0.411 ms. Daha derin ağlarda optimizasyon süresinin azalması, e-graph'ın kaynaştırma kuralları bilindikten sonra hızla doygunluğa ulaştığını ve arama uzayının derinlikle orantılı büyümediğini göstermektedir.

**Artık bağlantı (Residual connection):** `(add (relu (linear w b x)) x)` ifadesi `(add x (fused_linear_relu w b x))` olarak yeniden yazıldı. Optimizer, Linear+ReLU çiftini kaynaştırırken atlama bağlantısını (skip connection) korudu. Bu, ResNet tipi mimariler için önemlidir çünkü artık yol dokunulmadan kalmalıdır. Ortalama süre: 0.364 ms.

**Tam dikkat (attention) kalıbı:** `(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))` ifadesi `(fused_attention wq bq wk bk wv bv wo bo x x)` olarak yeniden yazıldı. Bu kaynaştırma, dört ayrı lineer projeksiyonu (Q, K, V, çıktı) ve dikkat hesaplamasını tek bir kaynaştırılmış çekirdeğe daraltarak üç ara tensör oluşturmayı ortadan kaldırır. Standart PyTorch'ta bu işlem 4 ayrı matris çarpımı + dikkat hesaplaması olarak yürütülürken, Hypatia hepsini tek bir çağrıya indirger. Ortalama süre: 0.381 ms.

---

## 2. torch.compile Arka Uç Entegrasyonu

Hypatia, kendisini PyTorch'un yerleşik arka uçlarının (inductor, cudagraphs vb.) yanına bir `torch.compile` arka ucu olarak kaydeder. Bir model `torch.compile(model, backend="hypatia")` ile derlendiğinde, FX grafiği yakalanır, Hypatia'nın S-ifadesi IR'sine dönüştürülür, e-graph aracılığıyla optimize edilir ve kaynaştırılmış operasyonlarla çalıştırılabilir graf modüllerine yeniden yapılandırılır.

### Benchmark Sonuçları

**Küçük MLP (128→64→32, 10.336 parametre):** Standart PyTorch 1000 ileri geçişi 14.72 ms'de tamamlarken, Hypatia derlenmiş sürüm aynı işlemi 37.19 ms'de tamamladı; hız oranı 0.40x. Küçük modellerde, derlenmiş arka ucun Python seviyesindeki gönderim yükü (dispatch overhead) gerçek hesaplama süresini domine eder. Standart ve derlenmiş sürümler arasındaki maksimum çıktı farkı 0.00 idi (bit-tam eşdeğer).

**Orta MLP (512→256→128→64, 172.480 parametre):** Standart PyTorch 1000 iterasyonu 1634.97 ms'de, Hypatia derlenmiş sürüm 1755.74 ms'de tamamladı; hız oranı 0.93x. Model büyüdükçe sabit yük oransal olarak küçülür. Çıktı farkı: 0.00 (bit-tam).

**Büyük MLP (1024→512→256→128→64, 697.280 parametre):** Standart PyTorch 1000 iterasyonu 1714.73 ms'de, Hypatia 1592.21 ms'de tamamladı; hız oranı 1.08x. Bu ölçekte, kaynaştırılmış Linear+ReLU operasyonları derleme yükünü aşmaya başlayarak net bir hızlanma sağlar. Çıktı farkı: 0.00 (bit-tam).

Kilit gözlem: Tüm derlenmiş çıktılar orijinal PyTorch çıktılarıyla bit düzeyinde aynıydı (maksimum fark = 0.00), bu da optimizasyonun semantik olarak doğru olduğunu onaylar. Hypatia'nın kaynaştırılmış operasyonlarının net hızlanma sağlamaya başladığı kesişim noktası, CPU çıkarımı için yaklaşık 500K-700K parametrede gerçekleşir.

---

## 3. Karma Hassasiyet (Mixed Precision) - FP16/BF16

Hypatia, iki yarı-hassasiyet formatı kullanarak karma hassasiyetli çıkarım uygular: IEEE 754 FP16 (10-bit mantis ile 16-bit kayan nokta) ve BF16 (Brain Float 16, 7-bit mantis, FP32'nin üs aralığıyla eşleşir). Ağırlıklar yarı-hassasiyette saklanır ancak hesaplama FP32'de yapılır (karma hassasiyetli GEMM: ağırlıklar matris çarpımı sırasında anında açılır). Dönüşüm, mevcut olduğunda F16C donanım talimatlarını kullanır.

### Benchmark Sonuçları

**Küçük katman (128×64, 8.192 parametre):** FP32 depolama 32.768 bayt gerektirirken, hem FP16 hem de BF16 16.384 bayt gerektirir; %50 bellek tasarrufu. Standart PyTorch'ta bu katman her zaman FP32'de (32.768 bayt) saklanırken, Hypatia ile FP16'ya dönüştürüldüğünde yalnızca 16.384 bayt kullanılır. FP16 dönüşümü maksimum 1.198×10⁻⁴ mutlak hata ve 2.034×10⁻⁵ RMSE ile gerçekleşir. BF16 dönüşümü, azaltılmış mantis hassasiyeti nedeniyle beklenen şekilde daha yüksek maksimum hata (9.743×10⁻⁴) ve RMSE (1.658×10⁻⁴) üretir. FP32 GEMM (1000 iterasyon) 13.41 ms'de, FP16 karma hassasiyetli GEMM 18.07 ms'de tamamlandı.

**Orta katman (512×256, 131.072 parametre):** Bellek tasarrufu %50 olarak sabit kalır (524.288 → 262.144 bayt). FP16 maks hata: 1.220×10⁻⁴, RMSE: 2.069×10⁻⁵. BF16 maks hata: 9.760×10⁻⁴, RMSE: 1.658×10⁻⁴. FP32 GEMM: 19.68 ms. FP16 karma hassasiyetli GEMM: 331.51 ms.

**Büyük katman (1024×512, 524.288 parametre):** FP32: 2.097.152 bayt → FP16/BF16: 1.048.576 bayt (%50 tasarruf). FP16 maks hata: 1.221×10⁻⁴, RMSE: 2.074×10⁻⁵. BF16 maks hata: 9.765×10⁻⁴, RMSE: 1.655×10⁻⁴.

**Çok büyük katman (2048×1024, 2.097.152 parametre):** FP32: 8.388.608 bayt → FP16/BF16: 4.194.304 bayt (%50 tasarruf). FP16 maks hata: 1.221×10⁻⁴, RMSE: 2.069×10⁻⁵. BF16 maks hata: 9.765×10⁻⁴, RMSE: 1.656×10⁻⁴.

Tüm katman boyutlarında tutarlı olan yaklaşık 1.22×10⁻⁴ FP16 maksimum hatası, hassasiyet kaybının sınırlı ve öngörülebilir olduğunu doğrular. BF16 hataları FP16'dan yaklaşık 8 kat daha büyüktür ancak yine de çıkarım için kabul edilebilir sınırlar içindedir. Karma hassasiyetli GEMM şu anda CPU'da yerel FP32 GEMM'den yavaştır çünkü açma yükü (half → float dönüşümü) CPU'da azaltılmış bellek bant genişliği ile dengelenmemektedir; bu yaklaşım öncelikle bellek bant genişliğinin darboğaz olduğu GPU çıkarımı ve model boyutu azaltmanın birincil hedef olduğu dağıtım senaryoları için faydalıdır.

---

## 4. Seyrek Tensör IR (CSR Formatı)

Hypatia, ağırlık matrisleri için Sıkıştırılmış Seyrek Satır (Compressed Sparse Row - CSR) formatı uygular ve seyrek-yoğun GEMM (SpMV/SpMM) operasyonları sağlar. Sistem ağırlık seyrekliğini analiz edebilir, yoğun matrisleri CSR formatına dönüştürebilir ve seyrek lineer ileri geçişler gerçekleştirebilir. Bu özellik, ağırlık budama (pruning) sonrasında ağırlıkların önemli bir kısmının sıfır olduğu durumlarda özellikle faydalıdır.

### Benchmark Sonuçları

**Küçük katman (128×64):**
%0 seyreklikte (tamamen yoğun): Standart PyTorch yoğun GEMM 1000 iterasyonu 15.32 ms'de tamamlarken, Hypatia seyrek GEMM 89.64 ms sürdü (0.17x). Yoğun bir matriste CSR formatının endeksleme yükü avantaj sağlamaz. %50 seyreklikte: yoğun 13.92 ms, seyrek 46.14 ms (0.30x). %80 seyreklikte: yoğun 17.13 ms, seyrek 19.83 ms (0.86x) — neredeyse eşit. %90 seyreklikte: yoğun 13.48 ms, seyrek 11.35 ms (1.19x) — seyrek hesaplama yoğundan hızlı olmaya başladı. %95 seyreklikte: yoğun 15.60 ms, seyrek 6.19 ms (2.52x) — belirgin hızlanma. Seyrek hesaplamanın yoğundan hızlı olduğu kesişim noktası bu katman boyutu için yaklaşık %85-90 seyreklikte gerçekleşir. Tüm çıktılar yoğun hesaplamayla sayısal uyum içindeydi (maks fark < 10⁻⁷).

**Orta katman (512×256):**
%0 seyreklikte: yoğun 21.37 ms, seyrek 1386.20 ms (0.02x). %95 seyreklikte: yoğun 32.63 ms, seyrek 74.31 ms (0.44x). Orta katmanlarda, CSR yönlendirme maliyeti satır sayısıyla ölçeklenir. %95 seyreklikte bile seyrek yol bu boyut için CPU'da hızlanma sağlamamaktadır.

**Büyük katman (1024×512):**
%0 seyreklikte: yoğun 22.17 ms, seyrek 5684.89 ms. %95 seyreklikte: yoğun 25.15 ms, seyrek 291.77 ms (0.09x). Büyük katmanlarda yoğun PyTorch GEMM, yüksek düzeyde optimize edilmiş BLAS rutinlerinden (MKL/OpenBLAS) yararlanır.

Kilit bulgu: Seyrek hesaplama, yalnızca küçük-orta katmanlarda çok yüksek seyreklik seviyelerinde (>%90) gerçek hızlanma sağlar. Seyrek IR'nin birincil faydası bellek azaltmadır: %95 seyreklikte 1024×512 boyutundaki bir katmanın CSR temsili, yoğun depolama için 2.097.152 bayta karşı yalnızca 320.224 bayt kullanır (6.5x sıkıştırma). Seyrek GEMM'in sayısal doğruluğu mükemmeldir; tüm konfigürasyonlarda maksimum farklar 10⁻⁶'nın altındadır.

---

## 5. Kaynaştırılmış Çok-Başlı Dikkat (Fused Multi-Head Attention)

Hypatia, Q/K/V projeksiyonlarını, ölçeklenmiş nokta çarpım dikkatini ve çıktı projeksiyonunu tek bir fonksiyon çağrısında gerçekleştiren, Rust'ta yazılmış yerel bir kaynaştırılmış dikkat çekirdeği uygular. Bu, ara tensör oluşturmayı ortadan kaldırır. Çekirdek şu hesaplamayı yapar: `Çıktı = (softmax(QK^T / √d_k) · V) · W_o + b_o`, burada Q, K, V ayrı ağırlık matrisleri kullanılarak girdiden projeksiyon yapılır.

### Benchmark Sonuçları

**Küçük konfigürasyon (d_model=64, 4 baş, seq_len=8, 16.384 parametre):** Standart PyTorch'un `nn.MultiheadAttention` modülü 500 iterasyonu 81.51 ms'de tamamladı. Hypatia'nın kaynaştırılmış dikkati aynı işlemi 4.91 ms'de tamamladı; **16.59 kat hızlanma**. Bu dramatik iyileşme, PyTorch'un her operasyon için ayrı gönderim yükünü ortadan kaldırmaktan kaynaklanır; küçük boyutlarda bu yük toplam süreyi domine eder.

**Orta konfigürasyon (d_model=128, 8 baş, seq_len=16, 65.536 parametre):** Standart PyTorch MHA: 500 iterasyon için 120.80 ms. Hypatia kaynaştırılmış: 56.93 ms. Hız oranı: **2.12x**. Gerçek hesaplama süresi gönderim yüküne göre büyüdükçe hızlanma azalır.

**Büyük konfigürasyon (d_model=256, 8 baş, seq_len=32, 262.144 parametre):** Standart PyTorch MHA: 143.79 ms. Hypatia kaynaştırılmış: 90.12 ms. Hız oranı: **1.60x**. Bu ölçekte bile kaynaştırılmış çekirdek, Q, K, V projeksiyonları için ara tahsislerden kaçınarak anlamlı bir avantaj sürdürür.

Kaynaştırılmış dikkat çekirdeği, test edilen tüm konfigürasyonlarda PyTorch'un modüler uygulamasını tutarlı bir şekilde geride bırakır; hızlanmalar 1.60x ile 16.59x arasında değişir. Avantaj, gönderim yükünün toplam yürütme süresinin daha büyük bir bölümünü oluşturduğu küçük boyutlarda en belirgindir.

---

## 6. Semantik Doğrulama

Hypatia, optimizasyon doğruluğunu iki seviyede doğrulayan bir semantik doğrulama sistemi içerir: (1) S-ifadesi yeniden yazmalarının yapısal doğrulaması (tüm değişkenlerin korunup korunmadığını ve ifade yapısının sağlam olup olmadığını kontrol eder) ve (2) model çıktı eşdeğerlik testi (rastgele girdileri orijinal ve optimize edilmiş modeller üzerinden çalıştırarak çıktıları maksimum mutlak fark ve kosinüs benzerliği kullanarak karşılaştırır).

### Benchmark Sonuçları

**Aynı model (klonlanmış):** 3 katmanlı bir MLP (256→128→64→32) derin kopyalandı ve her iki sürüm 10 rastgele test girdisi kullanılarak birbirine karşı doğrulandı. Standart PyTorch'ta `copy.deepcopy` ile klonlanan modelin çıktısı orijinalle karşılaştırıldığında: maksimum çıktı farkı 0.00 (bit-tam), ortalama fark 0.00 ve kosinüs benzerliği 1.0000000000 idi. Doğrulama 2.35 ms'de tamamlandı.

**Hafif bozulmuş model (gürültü=10⁻⁶):** Klonlanan modelin tüm parametrelerine 10⁻⁶ standart sapmalı Gauss gürültüsü eklendi. Bu, kuantizasyon veya budama sırasında oluşabilecek küçük ağırlık değişikliklerini simüle eder. Maksimum çıktı farkı 9.00×10⁻⁶, ortalama fark 2.73×10⁻⁶ ve kosinüs benzerliği 0.9999999995 oldu. Doğrulama, 10⁻⁴ toleransla doğru şekilde geçti; bu, küçük ağırlık bozulmalarının orantılı olarak küçük çıktı değişiklikleri ürettiğini doğrular.

**Tamamen farklı model:** Aynı mimariye sahip ancak farklı rastgele ağırlıklara sahip bir model orijinalle karşılaştırıldı. Maksimum çıktı farkı 5.20×10⁻¹ ve kosinüs benzerliği -0.0129 idi. Doğrulama doğru şekilde başarısızlık bildirdi; bu, sistemin semantik olarak eşdeğer olan ve olmayan modelleri ayırt edebildiğini gösterir.

**E-graph optimizasyon doğrulaması:** `(relu (linear w b x))` ifadesi `(fused_linear_relu w b x)` olarak optimize edildi. Yapısal doğrulama, tüm değişkenlerin (w, b, x) optimize edilmiş ifadede korunduğunu, düğüm sayısının 1 azaldığını ve FusedLinearReLU kaynaştırmasının doğru şekilde tanımlandığını onayladı.

---

## 7. Gerçek Model Analizi (GPT-2, DistilBERT)

Hypatia, analiz araçlarının üretim model yapıları üzerinde çalıştığını doğrulamak için HuggingFace'ten gerçek transformer mimarileri üzerinde test edildi.

### GPT-2 (2 katman, 128 boyutlu gömme)

Model, 10 ağırlık katmanı boyunca 655.872 parametre içerir. Standart PyTorch'ta bu model FP32 formatında 2.50 MB bellek kaplar; Hypatia'nın FP16 dönüşümüyle bu 1.25 MB'ye düşer ve 1.281 KB tasarruf sağlanır (%50). Dikkat katmanları, standart nn.Linear yerine HuggingFace'in Conv1D formatını (devrik ağırlık matrisleri) kullanır ve Hypatia her iki formatı da doğru şekilde işler.

Katman bazlı analiz, rastgele başlatılmış GPT-2 ağırlıklarının doğal seyrekliğinin yaklaşık 0.0000 olduğunu (başlatma sonrası tüm ağırlıklar sıfırdan farklı) gösterdi. Tüm katmanlarda FP16 dönüşüm hatası tutarlı bir şekilde yaklaşık 3.0×10⁻⁵ civarındaydı; bu, yarı-hassasiyetin bu mimari için güvenli olduğunu doğrular.

### DistilBERT (2 katman, 128 boyutlu)

Model, 14 ağırlık katmanı boyunca 409.600 parametre içerir. Standart PyTorch'ta FP32 bellek: 1.56 MB, Hypatia ile FP16 bellek: 0.78 MB, tasarruf: 800 KB. DistilBERT standart nn.Linear katmanları kullanır. Kelime gömme (word embedding) katmanı %0.1 doğal seyreklik gösterirken (değerlerin %0.1'i tam olarak sıfır), diğer tüm katmanlar sıfır doğal seyrekliğe sahipti. FP16 dönüşüm hataları tüm katmanlarda tutarlı bir şekilde yaklaşık 3.0×10⁻⁵ idi.

---

## 8. INT4 Blok Kuantizasyonu

Hypatia, agresif model sıkıştırması için yapılandırılabilir grup boyutuna sahip INT4 (4-bit tam sayı) blok kuantizasyonu uygular. Ağırlıklar, grup başına ölçek faktörleri ve sıfır noktalarıyla 4-bit tam sayılara kuantize edilir, ardından çıkarım sırasında FP32'ye geri açılır. Bu işlem, açma adımı için Rayon paralelliği ve SIMD talimatları kullanılarak gerçeklenir.

### Benchmark Sonuçları

**Küçük katman (256×128, 32.768 parametre):** Standart PyTorch'ta FP32 depolama 131.072 bayt kullanırken, Hypatia'nın INT4 kuantizasyonuyla bu 24.576 bayta düşer (5.3x sıkıştırma). FP32 GEMM (500 iterasyon): 8.93 ms. INT4 kuantize ileri geçiş: 45.32 ms (0.20x hız oranı). Maksimum çıktı hatası: 2.574×10⁻¹. FP32 ve INT4 çıktıları arasındaki kosinüs benzerliği: 0.99653.

**Orta katman (1024×512, 524.288 parametre):** Standart FP32: 2.097.152 bayt, Hypatia INT4: 393.216 bayt (5.3x sıkıştırma). FP32 GEMM: 14.75 ms. INT4 ileri geçiş: 65.96 ms (0.22x). Maks hata: 5.443×10⁻¹. Kosinüs benzerliği: 0.99683.

**Büyük katman (2048×1024, 2.097.152 parametre):** Standart FP32: 8.388.608 bayt, Hypatia INT4: 1.572.864 bayt (5.3x sıkıştırma). FP32 GEMM: 20.76 ms. INT4 ileri geçiş: 96.37 ms (0.22x). Maks hata: 8.214×10⁻¹. Kosinüs benzerliği: 0.99707.

INT4 kuantizasyonu tüm katman boyutlarında tutarlı 5.3x bellek sıkıştırması sağlar. 0.996'nın üzerindeki kosinüs benzerliği, bireysel değerler önemli ölçüde farklılık gösterse de (4-bit kuantizasyon için beklenen 10⁻¹ mertebesinde maksimum hatalar), genel çıktı yönünün iyi korunduğunu gösterir. Mevcut CPU gerçeklemesinde açma-sonra-GEMM işlemi yerel FP32 GEMM'den yavaştır; birincil fayda model depolama ve bellek ayak izinde 5.3 kat azalmadır ve bu, büyük modellerin bellek kısıtlı cihazlarda dağıtımı için kritiktir.

---

## Ek Özellikler

### Geometrik Cebir İşlemleri
Hypatia, NumPy entegrasyonlu 2B ve 3B geometrik cebir işlemlerinin (geometrik çarpım, rotörler, toplu döndürmeler) yerel Rust gerçeklemelerini içerir. Bu işlemler hem sayısal hem de sembolik hesaplama modlarını destekleyerek e-graph optimizer'da geometrik dönüşümlerin cebirsel sadeleştirilmesini mümkün kılar.

### Nöromorfik Hesaplama
Sistem, nöromorfik donanım dağıtımı için enerji tahmini ile birlikte Sızıntılı Entegre Et-ve-Ateşle (Leaky Integrate-and-Fire - LIF) nöron modelleri kullanan YSA'dan ASA'ya (Yapay Sinir Ağı'ndan Atımlı Sinir Ağı'na) dönüşümü destekler.

### Görselleştirme
Hypatia, hesaplama grafiklerini Graphviz işleme için DOT formatında dışa aktarabilir, ifadelerin ASCII ağaç temsillerini oluşturabilir ve orijinal ile optimize edilmiş ifade yapılarını karşılaştıran HTML optimizasyon raporları üretebilir.

---

## Mimari Özeti

| Bileşen | Gerçekleme | Amaç |
|---------|-----------|------|
| E-graph optimizer | Rust (`egg` kütüphanesi) | Eşitlik doygunluğu tabanlı yeniden yazma optimizasyonu |
| HypatiaLang IR | Rust (25+ operatör) | S-ifadesi ara temsili |
| torch.compile arka ucu | Python + Rust (PyO3) | FX Grafik → S-ifade → optimize et → yeniden yapılandır |
| Seyrek IR | Rust (CSR formatı) | Ağırlık budama, seyrek GEMM |
| Karma hassasiyet | Rust (F16C talimatları) | FP16/BF16 depolama, FP32 hesaplama |
| INT4 kuantizasyon | Rust (Rayon + SIMD) | 4-bit blok kuantizasyonu |
| Kaynaştırılmış dikkat | Rust (yerel çekirdek) | Q/K/V projeksiyonu + dikkat tek çağrıda |
| Semantik doğrulama | Rust + Python | Çıktı eşdeğerlik doğrulaması |
| Geometrik cebir | Rust (2B/3B) | Rotör tabanlı dönüşümler |
| Python bağlamaları | PyO3 | Sorunsuz Python/NumPy entegrasyonu |

---

## Sonuç

Hypatia, e-graph eşitlik doygunluğunun sinir ağı derlemesi için uygulanabilir bir yaklaşım olduğunu göstermektedir; milisaniyenin altında optimizasyon süreleriyle kanıtlanabilir şekilde doğru operatör kaynaştırmaları sağlar. Kaynaştırılmış dikkat çekirdeği, PyTorch'un modüler uygulamasına göre 1.6-16.6x hızlanma sağlar. Karma hassasiyet ve INT4 kuantizasyonu sırasıyla %50 ve %81 bellek azaltması sağlarken, tam hassasiyetli çıktılara yüksek kosinüs benzerliği (>0.996) korur. Semantik doğrulama sistemi, tüm optimizasyonların yapılandırılabilir toleranslar dahilinde model doğruluğunu korumasını güvence altına alır.
