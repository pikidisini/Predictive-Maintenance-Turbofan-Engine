import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Membaca data training
col_names = ['unit_number', 'time_in_cycles', 
             'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
            [f'sensor_{i}' for i in range(1, 22)]

df_train = pd.read_csv("NASA_Turbofan_Engine_Degradation_Simulation/CMaps/train_FD001.txt", sep=" ", header=None)
df_train.drop(columns=[26, 27], inplace=True)  # Buang kolom kosong di akhir baris
df_train.columns = col_names

# Lihat 5 baris pertama
print(df_train.head())

# Inforamasi Data
df_train.info()

# Cek Missing Value dan Duplicate
print("\nMissing values:\n", df_train.isnull().sum())
print("\nDuplicated rows:", df_train.duplicated().sum())

# Cek Distribusi Siklus Mesin
lifespan = df_train.groupby('unit_number')['time_in_cycles'].max()
lifespan.hist(bins=20)
plt.title("Distribusi Waktu Hidup Mesin (Training)")
plt.xlabel("Jumlah Siklus Sampai Gagal")
plt.ylabel("Jumlah Mesin")
plt.show()

lifespan.describe()

# Plot Sensor terhadap Waktu Siklus
unit_id = 10  # contoh unit
df_unit = df_train[df_train['unit_number'] == unit_id]

plt.plot(df_unit['time_in_cycles'], df_unit['sensor_2'])
plt.title(f"Sensor 2 - Mesin {unit_id}")
plt.xlabel("Siklus")
plt.ylabel("Nilai Sensor 2")
plt.grid(True)
plt.show()

# Menghitung Remaining Useful Life (RUL)
# Hitung RUL
rul_df = df_train.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul_df.columns = ['unit_number', 'max_cycle']

# Gabungkan ke dataset asli
df_train = df_train.merge(rul_df, on='unit_number')
df_train['RUL'] = df_train['max_cycle'] - df_train['time_in_cycles']

# Simpan hasil ke file CSV
df_train.to_csv("train_FD001_with_RUL.csv", index=False)
print("File berhasil disimpan sebagai 'train_FD001_with_RUL.csv'")


# Distribusi RUL
df_train['RUL'].hist(bins=50)
plt.title("Distribusi RUL")
plt.xlabel("Remaining Useful Life (RUL)")
plt.ylabel("Jumlah Sampel")
plt.show()

df_train['RUL'].describe()

# Korelasi Antar Sensor
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[[f'sensor_{i}' for i in range(1, 22)]].corr(), annot=False, cmap='coolwarm')
plt.title("Heatmap Korelasi Antar Sensor")
plt.show()

for sensor in ['sensor_9', 'sensor_14']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_train[sensor], y=df_train['RUL'], alpha=0.3)
    plt.title(f'{sensor} vs RUL')
    plt.xlabel(sensor)
    plt.ylabel('RUL')
    plt.show()
    

# Korelasi Sensor terhadap RUL
correlations = df_train[[f'sensor_{i}' for i in range(1, 22)] + ['RUL']].corr()['RUL'].drop('RUL')
print(correlations.sort_values(ascending=False))

# Distribusi Nilai Sensor yang tidak Berkorelasi
df_train[['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']].describe()

# Korelasi RUL dengan Senosr 11 dan 12
for sensor in ['sensor_11', 'sensor_12']:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=df_train[sensor], y=df_train['RUL'])
    plt.title(f'{sensor} vs RUL')
    plt.xlabel(sensor)
    plt.ylabel('RUL')
    plt.show()
    
    
# Normaliasasi  MinMaxScaller
# List of sensor columns
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 to sensor_21

# Inisialisasi scaler
scaler = MinMaxScaler()

# Terapkan normalisasi pada kolom sensor saja
df_train[sensor_cols] = scaler.fit_transform(df_train[sensor_cols])

# Cek hasil normalisasi
print(df_train[sensor_cols].head())

# Inisialisasi scaler untuk target (RUL)
target_scaler = MinMaxScaler()

# Terapkan normalisasi pada RUL
df_train['RUL'] = target_scaler.fit_transform(df_train[['RUL']])

# Drop Data yang tidak Berkorelasi
df_train.drop(columns=['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'], inplace=True)

# Menentukan Kolom Fitur
feature_cols = [col for col in df_train.columns if col.startswith('sensor_') or col.startswith('operational_setting')]
label_col = 'RUL'
window_size = 30

# Truncantion/Capping RUL
upper_limit = 200
lower_limit = 0
df_train['RUL'] = df_train['RUL'].clip(lower=lower_limit, upper=upper_limit)
df_train['RUL'].describe()


# Sequence Data untuk LSTM (windowing)
sequences = []
targets = []

for unit in df_train['unit_number'].unique():
    unit_data = df_train[df_train['unit_number'] == unit]
    for i in range(len(unit_data) - window_size + 1):
        window = unit_data.iloc[i:i + window_size]
        sequence = window[feature_cols].values
        target = window[label_col].values[-1]  # RUL pada akhir window
        sequences.append(sequence)
        targets.append(target)

X = np.array(sequences)
y = np.array(targets)

print("X shape:", X.shape)  # (samples, window_size, features)
print("y shape:", y.shape)  # (samples,)


# Pemisahan Data (Train-Test-Split)
test_size = 0.2  # 20% data untuk pengujian

# Membagi data secara time-series, dengan memastikan urutan waktu tetap terjaga
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# Menampilkan bentuk data pelatihan dan pengujian
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Modelling
# Menyiapkan Model LSTM
# Membangun model LSTM
model = Sequential()

# Lapisan LSTM pertama
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout untuk regularisasi

# Lapisan LSTM kedua
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))  # Dropout untuk regularisasi

# Lapisan Dense untuk output
model.add(Dense(units=1))  # 1 unit karena kita memprediksi nilai kontinu (RUL)

# Mengompilasi model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Menampilkan ringkasan model
model.summary()

# ReduceLROnPlateau: Kurangi learning rate jika validation loss tidak membaik
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Memantau loss pada data validasi
                              factor=0.5,         # Mengurangi learning rate menjadi setengahnya
                              patience=5,         # Menunggu 5 epoch jika loss tidak membaik
                              verbose=1,          # Menampilkan informasi saat learning rate dikurangi
                              min_lr=1e-6)        # Batas bawah untuk learning rate

# EarlyStopping: Hentikan pelatihan jika validation loss tidak membaik
early_stop = EarlyStopping(monitor='val_loss',    # Memantau loss pada data validasi
                           patience=10,           # Menunggu 10 epoch jika loss tidak membaik
                           verbose=1,             # Menampilkan informasi saat training berhenti
                           restore_best_weights=True)  # Mengembalikan bobot terbaik

# ModelCheckpoint: Simpan model dengan bobot terbaik berdasarkan validation loss
model_checkpoint = ModelCheckpoint('best_model.h5',   # Nama file untuk menyimpan model
                                   monitor='val_loss', # Memantau validation loss
                                   save_best_only=True, # Simpan hanya model terbaik
                                   verbose=1)           # Menampilkan informasi saat model disimpan

# Melatih Model
# Fit model dengan callbacks
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, early_stop, model_checkpoint])  # Menambahkan callbacks

# Evaluasi Pelatihan Model
# Visualisasi loss training dan validasi
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Simpan Model
# Menyimpan model
model.save('lstm_model_1.h5')

# Load Model
# Load model terbaik dari pelatihan
best_model = load_model('best_model.h5')

# Prediksi pada data test
y_pred = best_model.predict(X_test).flatten()
y_true = y_test.flatten()


# Hitung metrik evaluasi
mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# Visualisasi hasil prediksi vs nilai aktual RUL
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='Actual RUL', color='blue')
plt.plot(y_pred, label='Predicted RUL', color='red')
plt.title('Actual vs Predicted RUL')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()