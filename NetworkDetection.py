import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Veriyi yükle
df = pd.read_parquet('UNSW_NB15_training-set.parquet')

# 'Normal' etiketli 1000 örnek al
df_normal = df[df['attack_cat'] == 'Normal'].sample(1000, random_state=42)

# Diğer saldırı türlerinden 1000'er tane seçildi, toplamda 7000 örnek
categories = ['Reconnaissance', 'Fuzzers', 'DoS', 'Shellcode', 'Analysis', 'Exploits', 'Generic']
df_samples = [df[df['attack_cat'] == category].sample(1000, random_state=42) for category in categories]

# Tüm verileri birleştir
df_combined = pd.concat([df_normal] + df_samples)
df_combined.to_csv("ornek_data.csv", index=False)

# CSV dosyasını oku
ornek = pd.read_csv("ornek_data.csv")

# Gereksiz kolonları belirle
gereksizKolonlar = [
    'state', 'stcpb', 'dtcpb', 'swin', 'dwin', 'trans_depth',
    'response_body_len', 'is_ftp_login', 'ct_ftp_cmd', 'is_sm_ips_ports'
]

# Bu kolonları çıkararak yeni bir veri seti oluştur
temizVeri = ornek.drop(columns=gereksizKolonlar)

# Temiz veri setini CSV olarak kaydet
temizVeri.to_csv("temiz_veri.csv", index=False)

# Ekstra özellikler ekleyelim

# Paket başına düşen ortalama byte
temizVeri['ort_byte'] = temizVeri['sbytes'] / (temizVeri['spkts'] + 1)

# Giden ve gelen byte oranı
temizVeri['giden_gelen_byte'] = temizVeri['sbytes'] / (temizVeri['dbytes'] + 1)

# İstek-cevap oranı
temizVeri['istek_cevap_orani'] = temizVeri['spkts'] / (temizVeri['dpkts'] + 1)

# Veriyi tekrar oku
temizVeri = pd.read_csv("temiz_veri.csv")

# Bağımsız ve bağımlı değişkenleri ayır
X = temizVeri.drop(columns=['attack_cat', 'label'])
y = temizVeri['attack_cat']

# Kategorik veriler için One-Hot Encoding uygula
X = pd.get_dummies(X, columns=['proto', 'service'], drop_first=True)

# True/False değerlerini 0 ve 1'e çevir
X = X.astype({col: 'float' for col in X.select_dtypes(include=['bool']).columns})

# Veriyi normalleştir
scaler = MinMaxScaler()
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Etiketleri sayısallaştır
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost modelini tanımla
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False
)

# XGBoost modelini eğit
xgb_model.fit(X_train, y_train)

# Test setinde tahmin yap
y_pred_xgb = xgb_model.predict(X_test)


# Sınıf etiketlerini sıralı şekilde tanımla
class_labels =['Analysis', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode']

# XGBoost modelinin performans değerlendirmesi
print("XGBoost Model Doğruluk:", accuracy_score(y_test, y_pred_xgb))
print("\nXGBoost Model Analiz Raporu:\n", classification_report(y_test, y_pred_xgb,target_names=class_labels))


# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred_xgb)

# LabelEncoder ile dönüştürülen sınıf etiketlerini tersine çevirerek orijinal sınıf isimlerini al
class_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

# Sınıf bazlı doğruluk oranlarını hesapla
class_accuracy = []
for i, label in enumerate(class_labels):
    true_positives = cm[i, i]  # Doğru tahmin edilen sınıf örnekleri
    total_samples = cm[i, :].sum()  # Sınıfa ait toplam örnek sayısı
    accuracy = true_positives / total_samples if total_samples > 0 else 0  # Doğruluk oranı
    class_accuracy.append(accuracy)

# Grafik oluştur: Sınıf bazlı doğruluk oranları
plt.figure(figsize=(10, 6))
colors = ['green' if label == 'Normal' else 'red' for label in class_labels]  # Normal sınıfını yeşil yap
plt.bar(class_labels, class_accuracy, color=colors)

# Grafik ayarları
plt.xlabel('Ağ Trafiği Sınıfları')  # X ekseni etiketi
plt.ylabel('Doğruluk Oranı')  # Y ekseni etiketi
plt.title('Her Sınıf için Doğru Tahmin Oranları')  # Başlık
plt.ylim(0, 1.1)  # Doğruluk oranı aralığı
plt.xticks(rotation=45)  # Sınıf isimlerini döndür
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Y eksenine kılavuz çizgileri

# Görselleştirme
plt.tight_layout()
plt.show()
