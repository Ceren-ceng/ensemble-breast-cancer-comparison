# 🔬 Ensemble Learning ile Meme Kanseri Sınıflandırması (Random Forest, AdaBoost, XGBoost)

Bu projede, Breast Cancer Wisconsin (Diagnostic) veri seti ile üç farklı ensemble öğrenme algoritmasının (Random Forest, AdaBoost ve XGBoost) performansını aynı veri seti üzerinde karşılaştırıyoruz. Tüm aşamalar adım adım anlatılmıştır. Kod örnekleri ve çıktılarla tam bir rehberdir.

---

## 📢 Projenin Amacı

- Ensemble (topluluk) öğrenme yöntemlerini pratikte kıyaslamak
- Hangi model hangi tür veri ve problemde daha güçlü, avantaj ve dezavantajları nelerdir görmek
- Sıfırdan çalışma akışını görmek isteyenler için açıklamalı ve gerçek çıktılı örnek sunmak

---

## 📚 Ensemble Learning Nedir?

Ensemble Learning, birden fazla makine öğrenmesi modelinin birleştirilerek (çoğunluk oyu, ağırlıklı ortalama vs.) daha sağlam, genellenebilir ve güvenilir sonuçlar alınmasını amaçlar. Tek başına zayıf kalabilecek modeller bir araya gelince çoğu zaman daha başarılı tahminler yapar.

**İki temel yaklaşım vardır:**
- **Bagging (Bootstrap Aggregating):** Aynı algoritmadan birçok farklı model rastgele örneklerle eğitilir (ör: Random Forest).
- **Boosting:** Modeller sıralı şekilde eğitilir, her biri öncekinin hatalarını düzeltmeye odaklanır (ör: AdaBoost, XGBoost).

---

## 🏷️ Kullanılan Algoritmaların Temel Farkları

### 🌲 Random Forest (Bagging)
- Çok sayıda karar ağacı oluşturur
- Her bir ağaç farklı örneklerle ve/veya farklı özelliklerle eğitilir
- Sonuç, çoğunluk oyuyla verilir (klasik “demokrasi”)
- Aşırı öğrenmeye karşı dirençlidir

### 🚀 AdaBoost (Boosting)
- Zayıf öğrenicileri (genellikle tek katmanlı karar ağaçları) sıralı eğitir
- Her yeni model, bir öncekinin yanlışlarını düzeltmeye çalışır
- Yanlış tahmin edilen örneklere daha çok ağırlık verir

### ⚡ XGBoost (Boosting – Advanced)
- Boosting yaklaşımının en gelişmiş, hızlı ve regularize edilmiş hali
- Eksik veriyle başa çıkabilir, paralel çalışır, overfitting’i azaltır
- Genellikle en iyi doğruluk oranlarına ulaşır

---

## 🗂️ Veri Seti Özeti

- **Kaynak:** Breast Cancer Wisconsin (Diagnostic)
- **Toplam Gözlem:** 569
- **Özellik Sayısı:** 30 + 1 hedef sütunu (`diagnosis`)
- **Hedef Değişken:** `diagnosis` (‘M’ = Malignant/Kötü Huylu, ‘B’ = Benign/İyi Huylu)

### Örnek Sütunlar:
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean` ...
- “Unnamed: 32” gibi alakasız/eksik sütunlar silindi

---

## 🧹 Veri Hazırlama Aşamaları

1. **Yükleme ve Temizlik:**
   - Gereksiz sütunlar (örn. id, Unnamed: 32) silindi.
   - `diagnosis` etiketi `0` (B) ve `1` (M) olarak encode edildi.
2. **Öznitelik ve Hedef Ayrımı:**
   - X = tüm özellikler, y = diagnosis etiketi
3. **Eğitim/Test Bölmesi:**
   - %80 eğitim, %20 test (stratify ile)
4. **Ölçekleme yapılmadı** çünkü ağaç tabanlı modeller buna ihtiyaç duymaz.

---

## 🧠 Model Eğitimi ve Değerlendirme

Her model için:

- Model oluşturuldu ve `.fit()` ile eğitildi
- Test setinde `.predict()` ile tahminler yapıldı
- Hata metrikleri (`accuracy`, `precision`, `recall`, `f1-score`) alındı
- Confusion matrix (karışıklık matrisi) görseli çizildi

---

## 🧾 Kullanılan Kütüphaneler

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


## 📊 Sonuç Tablosu

| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 0.9649   | 0.9722    | 0.9444 | 0.9581   |
| AdaBoost      | 0.9561   | 0.9722    | 0.9306 | 0.9510   |
| XGBoost       | 0.9737   | 0.9722    | 0.9583 | 0.9652   |

---

### 📈 Classification Reports

#### Random Forest

markdown
Kopyala
Düzenle
          precision    recall  f1-score   support

       0      0.97      0.98      0.98        71
       1      0.97      0.94      0.96        43

accuracy                          0.96       114
macro avg 0.97 0.96 0.96 114
weighted avg 0.97 0.96 0.96 114

shell
Kopyala
Düzenle

#### AdaBoost

markdown
Kopyala
Düzenle
          precision    recall  f1-score   support

       0      0.97      0.97      0.97        71
       1      0.97      0.93      0.95        43

accuracy                          0.96       114
macro avg 0.97 0.95 0.96 114
weighted avg 0.97 0.96 0.96 114

shell
Kopyala
Düzenle

#### XGBoost

markdown
Kopyala
Düzenle
          precision    recall  f1-score   support

       0      0.97      0.99      0.98        71
       1      0.97      0.96      0.97        43

accuracy                          0.97       114
macro avg 0.97 0.97 0.97 114
weighted avg 0.97 0.97 0.97 114

yaml
Kopyala
Düzenle

---

### 🧑‍🔬 Sonuçların Yorumlanması ve F1-Score Değerlendirmesi

- **XGBoost** en yüksek accuracy ve en dengeli f1-score'u verdi.  
- **Random Forest** ve **AdaBoost** da çok yakın ve güvenilir sonuçlar verdi.
- F1-score değerleri hem yanlış pozitifleri hem de yanlış negatifleri cezalandırdığı için sağlık alanı gibi kritik uygulamalarda en önemli metriktir.
- Precision değerleri yüksek: "kanser" dediği hastaların neredeyse tamamı gerçekten hasta.
- Recall yüksek: Model, neredeyse tüm kanserli hastaları doğru tespit ediyor.
- Sonuç: Bu veri seti için en yüksek genel başarı XGBoost ile sağlandı. Random Forest ve AdaBoost da neredeyse aynı başarıda; çok büyük farklar yok.

---

🧑‍🔬 Sonuçların Teknik Değerlendirmesi
Sadece accuracy’ye bakmak yetersizdir. F1-score, recall ve precision sağlık gibi kritik alanlarda çok daha önemlidir.

F1-score yüksekse, model hem yanlış “hasta” diyerek hem de “hastayı gözden kaçırarak” hata yapmıyor demektir.

Precision düşükse, model sağlıklı insana “kanser” deme eğiliminde (gereksiz korku, gereksiz test)

Recall düşükse, model kanser hastasını gözden kaçırıyor (hayati risk!)

Klinik, sağlık ve riskli alanlarda F1-score ve recall birincil metrik olarak izlenmelidir. En yüksek f1-score ve recall, “en az ölümcül hata” anlamına gelir.

🏁 Son Söz: Hangi Model Ne Zaman?
Random Forest: Hızlı, dengeli ve genellikle default ayarlarıyla bile iyi sonuç

AdaBoost: Basit veride ve outlier azsa iyi çalışır, noise fazlaysa düşer

XGBoost: Karmaşık veri, büyük set, yüksek başarı isteniyorsa önerilir; ayar yapmak önemlidir

🤝 Katkı ve Geliştirme
Bu repo öğrenmek ve kendi veri setlerinizi test etmek için uygundur. PR atabilirsiniz, parametrelerle oynayabilir, yeni metrik veya grafik ekleyebilirsiniz.



Keywords / Anahtar Kelimeler:

ensemble learning

random forest

adaboost

xgboost

breast cancer

cancer detection

classification

machine learning

boosting

bagging

sklearn

python

medical data

f1-score

accuracy

recall

precision

binary classification

data science

medical AI

diagnostic prediction

model comparison

confusion matrix

healthcare AI

-----------------------------------------

topluluk öğrenmesi

rastgele orman

adaboost

xgboost

meme kanseri

kanser tespiti

sınıflandırma

makine öğrenmesi

hata metrikleri

doğruluk

sağlık verisi

ikili sınıflandırma

model karşılaştırma

yapay zeka
