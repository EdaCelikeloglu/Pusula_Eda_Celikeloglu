# EDA ÇELİKELOĞLU
## edacelikeloglu@gmail.com

# README — Veri Temizleme, Özellik Mühendisliği ve Modellemeye Hazır Set Pipeline’ı

Bu repo; ham klinik kayıt tablosunu **temizleyen**, **normalize eden**, **eksikleri KNN ile dolduran**, **özellik türeten** ve sonuçta **modellemeye hazır** bir veri seti üreten uçtan uca bir Python pipeline’ı içerir. Süreç tek dosyada (`pipeline.py`) toplanmıştır; alan-özgü normalizasyon kuralları `rules/` klasöründedir.

---

## İçindekiler

- Genel Bakış  
- Dizin Yapısı  
- Nasıl Çalıştırılır  
- Veri Beklentileri  
- İşleme Adımları ve Gerekçeler  
  1) Keşif & Raporlama  
  2) Metinsel Sürelerin Sayısala Çevrilmesi  
  3) Düşük Kardinaliteli Kolonların Kategoriğe Çevrilmesi  
  4) Kural Tabanlı Normalizasyon (Çok-Değerli Metinler)  
  5) Uygulama Yerleri Üst Gruplama  
  6) Kimlik (ID) Türetimi  
  7) Özet ve Sayaç Özellikleri  
  8) KNN ile Eksik Doldurma  
  9) Yaşın Ölçeklenmesi  
  10) One-Hot & Multi-Hot Özellikler  
  11) Modellemeye Hazır Tablonun Şeması  
- Konsol Çıktısı  
- EDA Çıktıları  
- Üretilen Dosyalar  
- **Görsel Etiket Standardı**  
- **Çok-Etiketli Alanlarda Eksik Bayrakları**  
- **Multi-Hot “other” Sütunları**  
- **KNN İmputasyonu – Parametreler & Kapsam**  
- **ID Sıralama Varsayımı**  
- **Gereksinimler (önerilen sürümler)**  
- Kurallar (`rules/`) — Yapı ve Örnekler  
  - Kuralların Çalışma Mantığı  
  - Dosya Bazında Beklenen Şema  
  - Örnek Kural Dosyaları  
  - En İyi Uygulamalar ve İpuçları  
- Gizlilik / Görselleştirme Güvenliği  
- Bilinen Sınırlar  
- Parametreleştirme (opsiyonel)  
- Sorun Giderme  
- Genişletme İpuçları

---

## Genel Bakış

Pipeline, `data/Talent_Academy_Case_DT_2025.xlsx` dosyasını okuyarak aşağıdaki çıktıları üretir:

- **EDA görselleri** → `reports/figures/`  
- **Ara tablo (satırlar)** → `data/rows.xlsx`  
- **Tedavi düzeyi özet** → `data/by_treatment.xlsx`  
- **Modellemeye hazır tablo** → `data/model_ready.xlsx`

Temel ilkeler:

- Eksik değer doldurma **KNNImputer** ile yapılır.  
- Kategorikler KNN için **geçici koda** çevrilir, imputasyon sonrası **etikete geri döner**.  
- `Yas`, **KNN’den sonra** ölçeklenir.  
- Konsol raporu **yalın** tutulur: yalnızca gerçekten doldurulan kolonlar ve **kaç hücrenin** doldurulduğu yazılır; **nihai NA kontrolü** verilir.

---

## Dizin Yapısı

    project-root/
    ├─ data/
    │  ├─ Talent_Academy_Case_DT_2025.xlsx     # Girdi (ham veri)
    │  ├─ rows.xlsx                            # Çıktı 1: Satır düzeyi veri
    │  ├─ by_treatment.xlsx                    # Çıktı 2: Tedavi düzeyi özet
    │  └─ model_ready.xlsx                     # Çıktı 3: Modellemeye hazır tablo
    ├─ reports/
    │  └─ figures/                             # EDA görselleri (PNG)
    ├─ rules/                                  # Normalizasyon kuralları (Python modülleri)
    │  ├─ common.py
    │  ├─ tanilar.py
    │  ├─ alerji.py
    │  ├─ kronik.py
    │  ├─ uyg_yer.py
    │  ├─ tedavi.py
    │  ├─ noise.py
    │  └─ site_groups.py
    ├─ requirements.txt                        # Bağımlılıklar bu dosyada
    └─ pipeline.py                             # Tüm akışın tek dosyası

---

## Nasıl Çalıştırılır

1. Bağımlılıklar **requirements.txt** içindedir. Terminalde:
       
       pip install -r requirements.txt

2. Girdi dosyasını `data/Talent_Academy_Case_DT_2025.xlsx` konumuna koyun.  
3. Komut satırından çalıştırın:

       python pipeline.py

> Not: Proje Python 3.12.6 ile geliştirilmiştir (uyumlu 3.10+ önerilir).

---

## Veri Beklentileri

- **Kimlik & Demografi**: `HastaNo`, `Yas`, `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`  
- **Metin alanları (virgülle ayrılmış olabilir)**: `KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri`, `TedaviAdi`  
- **Süreler**:
  - `TedaviSuresi` (örn. “15 Seans”) → **`TedaviSuresi(Seans)`**  
  - `UygulamaSuresi` (örn. “20 Dakika”) → **`UygulamaSuresi(Dakika)`**

---

## İşleme Adımları ve Gerekçeler

### 1) Keşif & Raporlama
- `head()`, `info()`, **sayısal/kategorik `describe()`** konsola yazılır.  
- **Boş değer sayıları**: kolon bazında NA adetleri yazdırılır.  
- EDA görselleri **NA özetinden sonra** üretilir.

### 2) Metinsel Sürelerin Sayısala Çevrilmesi
- “5 Seans” → `TedaviSuresi(Seans)=5` (NA destekli `Int64`)  
- “20 Dakika” → `UygulamaSuresi(Dakika)=20`

### 3) Düşük Kardinaliteli Kolonların Kategoriğe Çevrilmesi
- `threshold=10` altındaki `object/string` kolonlar **category** tipe çevrilir.  
- Öncesinde `str.strip()` + boş/“nan/none/null/na” → **NaN** yapılır.

### 4) Kural Tabanlı Normalizasyon (Çok-Değerli Metinler)
- `KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri`, `TedaviAdi` üzerinde:  
  - Büyük/küçük, noktalama, tekrarlar normalize edilir.  
  - `rules/*.py` içindeki **regex** kanonikleştirme kuralları uygulanır.  
  - `noise.py`’daki **gürültü etiketleri** düşürülür.

### 5) Uygulama Yerleri Üst Gruplama
- `UygulamaYerleri_cleaned` token’ları, `site_groups.py`’daki desenlere göre **üst gruplara** (`UygulamaYerleri_grouped`) map edilir.

### 6) Kimlik (ID) Türetimi
- `HastaTedaviID`: `HastaNo::tedavi_slug`  
- `HastaTedaviSeansID`: `HastaTedaviID#NNN` (seans sıra numarası)

### 7) Özet ve Sayaç Özellikleri
- Çok-değerli sütunlar için `__primary` (ilk token) ve `__count` (tekil token sayısı).

### 8) KNN ile Eksik Doldurma
- Matris: **`TedaviSuresi(Seans)`, `UygulamaSuresi(Dakika)`, `Yas`, `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`**  
- Kategorikler **geçici koda** çevrilir → **KNNImputer** → **etikete dönüş**.  
- Parametre: `n_neighbors=5`, `weights="distance"`.  
- `KanGrubu` öncesi “0 Rh+” → “0Rh+” boşluk temizliği.  
- KNN sonrası **NA kalırsa hata**; placeholder kullanılmaz.  
- Konsol raporunda **yalnız** gerçekten doldurulan kolonlar ve **doldurulan hücre sayıları** yazılır; ayrıca **toplam doldurulan hücre** ve **nihai NA** kontrolü verilir.

### 9) Yaşın Ölçeklenmesi
- `Yas`, KNN’den **sonra** `StandardScaler` ile ölçeklenir.

### 10) One-Hot & Multi-Hot Özellikler
- **One-Hot**: `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum` → `drop_first=True`, `dummy_na=False`  
  - Konsolda **baz başına** kaç yeni sütun üretildiği yazılır.
- **Multi-Hot**:  
  - `Tanilar_cleaned` → `DX__` (Top 30)  
  - `KronikHastalik_cleaned` → `CHR__` (Top 20)  
  - `Alerji_cleaned` → `ALG__` (Top 10)  
  - `UygulamaYerleri_grouped` → `SITE__` (Tümü)  
  - `TedaviAdi_cleaned` → `TX__` (Top 15)  
- “`__n`” sayım kolonları **model_ready’ye eklenmez** (yalnız 1/0’lar eklenir).

### 11) Modellemeye Hazır Tablonun Şeması (Sıra)
1. `HastaTedaviSeansID`  
2. `TedaviSuresi(Seans)`  
3. `UygulamaSuresi(Dakika)`  
4. `Yas` (ölçeklenmiş)  
5. One-hot kolonları (Cinsiyet_*, KanGrubu_*, Uyruk_*, Bolum_*)  
6. Multi-hot 1/0 kolonları (DX__, CHR__, ALG__, SITE__, TX__)

---

## Konsol Çıktısı (Güncel Sözleşme)

- Başta `head()`, `info()`, `describe()` ve kolon bazında NA sayıları.  
- “EDA görselleri kaydedildi (reports/figures).”  
- Rapor:
  - “Kategoriğe çevrilen sütunlar: …”  
  - “Eksik değer doldurma (KNNImputer):  
    — Doldurulan sayısal kolonlar: …  
    — Doldurulan kategorik kolonlar: …  
    — Doldurulan hücre sayıları: {Kolon=Adet,…}  
    — Toplam doldurulan hücre: N”  
  - “Dönüşümler:  
    — Yas ölçeklendi: Evet/Hayır  
    — One-hot (drop_first=True): {Cinsiyet=…, KanGrubu=…, Uyruk=…, Bolum=…}  
    — Multi-hot: {DX__=30, CHR__=20, ALG__=10, SITE__=tümü, TX__=15}”  
  - “Modellemeye hazır tabloda toplam NA: 0” → “Boş değer kontrolü: PAS”

---

## EDA Çıktıları

`reports/figures/` klasöründe:

- **Yaş**: Histogram (ort./medyan çizgili) — Matplotlib  
- **TedaviSuresi(Seans)**: Tam sayıya hizalı histogram — Matplotlib  
- **UygulamaSuresi**: Donut (Top6 + “Diğer”) — Matplotlib  
- **UygulamaYerleri**: Yatay bar (token’lanmış, Top30) — Matplotlib  
- **Uyruk**: Yatay bar (Top10 + “Diğer”) — Matplotlib  
- **Cinsiyet**: Yüzde bar (NaN gösterimi yalnız grafikte) — Matplotlib  
- **KanGrubu**: ABO × Rh gruplu bar — Matplotlib  
- **Korelasyon Isı Haritası** (hedef ilk sırada): `TedaviSuresi(Seans)`, `UygulamaSuresi(Dakika)`, `Yas` — **Seaborn**  
- **Scatter + Trend**: `Yas` ↔ `TedaviSuresi(Seans)` — **Seaborn**

> Gizlilik: **HastaNo / HastaTedaviID / HastaTedaviSeansID** görselleştirilmez.

---

## Üretilen Dosyalar

- `data/rows.xlsx` — Temizlenmiş **satır düzeyi** veri  
- `data/by_treatment.xlsx` — **Tedavi başına** özet metrikler  
- `data/model_ready.xlsx` — **Eksikleri KNN ile kapanmış**, **Yas ölçeklenmiş**, **one-hot + multi-hot** genişletilmiş nihai set

---

## Görsel Etiket Standardı

- Grafiklerde **tüm başlıklar, eksen etiketleri, lejant başlıkları ve kategori yazıları** **BÜYÜK HARF**’e normalize edilir (Türkçe `i/ı` düzeltmeli).  
- Bu standart `pipeline.py` içindeki `UC()` yardımcı fonksiyonuyla otomatik uygulanır.

---

## Çok-Etiketli Alanlarda Eksik Bayrakları

Aşağıdaki alanlar için ek bayrak sütunları üretilir (hücre tamamen boşsa 1):

- `DX__missing` (Tanilar_cleaned), `CHR__missing`, `ALG__missing`, `SITE__missing`, `TX__missing`  
- Çok-etiketli NA’larda **imputasyon yapılmaz**; eksiklik bu bayraklarla temsil edilir.

---

## Multi-Hot “other” Sütunları

- `DX__`, `CHR__`, `ALG__`, `TX__` için Top-N dışındaki etiketler **`__other`** kolonu ile yakalanır.  
- `SITE__`’ta **other yoktur** (tüm gruplar üretilir).

---

## KNN İmputasyonu – Parametreler & Kapsam

- Matris: `TedaviSuresi(Seans)`, `UygulamaSuresi(Dakika)`, `Yas`, `Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`  
- Parametreler: `n_neighbors=5`, `weights="distance"`  
- Kategorikler KNN için **koda** çevrilir, oluşan değerler **etikete döndürülür**.  
- KNN sonrası NA kalırsa **hata fırlatılır** (placeholder yok).  
- Not: Sayısal kolonlar modelleme için numerik (float) formatta bırakılır.

---

## ID Sıralama Varsayımı

- `SeansIndex` satır sırasına göre atanır. Elinizde tarih/saat varsa, pipeline’ı **çalıştırmadan önce** veri setini kronolojik sıralamanız önerilir.

---

## Gereksinimler (önerilen sürümler)

- Bağımlılıklar **`requirements.txt`** dosyasında listelidir. Örnek sürümler:

      numpy==2.3.2
      pandas==2.3.2
      matplotlib==3.10.6
      seaborn==0.13.2
      scikit-learn==1.7.1
      openpyxl==3.1.5

---

## Kurallar (`rules/`) — Yapı ve Örnekler

### Kuralların Çalışma Mantığı

- `pipeline.py`, `rules/` içindeki Python modüllerini **dinamik** yükler.  
- İlgili sütun için belirtilen kural listeleri **sırasıyla** uygulanır.  
- Normalizasyon, metni **BÜYÜK HARF** ve sade karakter kümesine çeker; ardından regex kurallarınız devreye girer. Bu nedenle pattern’ları **büyük harf beklentisi** ile yazmak güvenlidir.

### Dosya Bazında Beklenen Şema

1) **Kanonikleştirme listesi** (regex, replacement) çiftleri:

       # patterns = [(r"REGEX1", "REPL1"), (r"REGEX2", "REPL2"), ...]

2) **Gürültü setleri** veya **gruplama tabloları**:
- `noise.py` → `tedavi_drop_set`, `tanilar_drop_set` (tam eşleşen token’ları düşürmek için)  
- `site_groups.py` → `{"pattern": "...", "label": "..."}` biçiminde liste (üst gruplama)

> Tüm regex’leri Python ham string (`r"..."`) olarak yazın.

### Örnek Kural Dosyaları

**`rules/common.py`**

       patterns = [
           (r"\bSERVIKOTORAS(I|İ)K\b", "SERVİKOTORASİK"),
           (r"\bDORSALJ(I|İ)\b", "DORSALJİ"),
       ]

**`rules/tanilar.py`**

       patterns = [
           (r"\bOMUZUN DARBE SENDROMU\b", "OMUZ DARBE SENDROMU"),
           # veri içi varyantlarınızı ekleyin
       ]

**`rules/alerji.py`**

       patterns = [
           (r"\bTOZ ALERJI(S|Ş)I\b", "TOZ ALERJİSİ"),
           (r"\bPOLEN ALERJ(I|İ)S(I|İ)\b", "POLEN ALERJİSİ"),
       ]

**`rules/kronik.py`**

       patterns = [
           (r"\bHTN\b", "HİPERTANSİYON"),
           (r"\bDM\b", "DİYABET MELLİTUS"),
           (r"\bKBY\b", "KRONİK BÖBREK YETMEZLİĞİ"),
       ]

**`rules/uyg_yer.py`**

       patterns = [
           (r"\bAYAK\s*B(I|İ)LE(G|Ğ)I\b", "AYAK BİLEĞİ"),
           (r"\bSKAPULA(R)?\b", "SKAPULA"),
           (r"\bTRAPEZ(\s*KAS(I|İ))?\b", "TRAPEZ"),
       ]

**`rules/tedavi.py`**

       patterns = [
           (r"\bTENS\b", "TENS"),
           (r"\bULTRASON(OGRAF(I|İ))?\b", "ULTRASON"),
           (r"\bESWT\b", "ESWT"),
       ]

**`rules/noise.py`**

       tedavi_drop_set = {"TEDAVİ", "UYGULAMA", "SEANS", "GENEL", "DİĞER"}
       tanilar_drop_set = {"DİĞER", "KONTROL", "BİLGİ", "HATA"}

**`rules/site_groups.py`**

       groups = [
           {"pattern": r"\bBOYUN\b", "label": "BOYUN"},
           {"pattern": r"\bBEL|LOMBER\b", "label": "BEL/LOMBER"},
           {"pattern": r"\bOMUZ\b", "label": "OMUZ"},
           {"pattern": r"\bS(K|C)APULA\b", "label": "SKAPULA"},
           {"pattern": r"\bAYAK\s*B(I|İ)LE(G|Ğ)I\b", "label": "AYAK BİLEĞİ"},
       ]

### En İyi Uygulamalar ve İpuçları

- **BÜYÜK HARF** yazın (normalize zaten büyük harfe çeviriyor).  
- Pattern’lar **kelime tabanlı** olsun: `\b...\b`.  
- Önce **genel** (`common.py`), sonra **alan-özgü** kurallar (örn. `tanilar.py`).  
- `noise.py` set’lerine gerçekten anlamsız/gürültü token’ları koyun; aşırı genişletmeyin.  
- `site_groups.py` sıra duyarlıdır; en spesifik pattern’lar **önce** gelsin.

---

## Gizlilik / Görselleştirme Güvenliği

- Kişisel kimlik unsurları (`HastaNo`, `HastaTedaviID`, `HastaTedaviSeansID`) **grafiklerde kullanılmaz**.  
- EDA çıktıları yalnızca toplulaştırılmış/anonimleştirilmiş görseller üretir.

---

## Bilinen Sınırlar

- KNN yalnızca **baz kategorik + sayısal** matriste çalışır; çok-etiketli metin alanlarında **imputasyon yapılmaz**.  
- `SeansIndex` veri sırasına bağlıdır; zaman damgası yoksa gerçek klinik sıra ile **birebir** örtüşmeyebilir.  
- Aşırı seyrek kategoriler OHE sonrası yüksek boyut oluşturabilir (özellikle `SITE__`).

---

## Parametreleştirme (opsiyonel)

- Multi-hot eşikleri: `DX`=30, `CHR`=20, `ALG`=10, `TX`=15; `SITE`=tümü.  
- KNN: `n_neighbors=5`, `weights="distance"`.  
- Görsel tema: seaborn `whitegrid`, etiketler **tamamı BÜYÜK** (`UC()` ile).  
- OHE: `drop_first=True`, `dummy_na=False`.

---

## Sorun Giderme

- **Girdi dosyası yok** → `data/Talent_Academy_Case_DT_2025.xlsx` yolunu kontrol edin.  
- **Kurallar eksik/hatalı** → `rules/` altındaki modül adları ve `patterns`/`*_set`/`groups` isimleri doğru olmalı.  
- **Excel’e yazma hatası** → Bağımlılıkları `requirements.txt` ile kurduğunuzdan emin olun.  
- **KNN sonrası NA kaldı** → KNN matrisi kolonlarını ve veri kapsamını kontrol edin (tamamen boş bir kategorik varsa NA kalabilir).  

---

## Genişletme İpuçları

- Multi-hot Top-N eşikleri iş ihtiyacına göre ayarlanabilir.  
- `rules/` kütüphanesi yaşayan bir sözlük gibi genişletilebilir.  
- `model_ready` şemasını sabitlemek için whitelist ile kolon sırası kilitlenebilir.  
- Büyük veri için KNN maliyetini azaltmak adına matrise dahil edilen kolon sayısı optimize edilebilir.
