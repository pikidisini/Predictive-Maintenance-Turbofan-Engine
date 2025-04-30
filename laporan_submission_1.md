# Laporan Proyek Machine Learning - Fiqih Hidayaturrahman

## Predictive Maintenance menggunakan NASA Turbofan Engine Degradation Dataset.

**Latar Belakang**:
Dalam dunia industri, terutama di sektor penerbangan, efisiensi operasional dan keselamatan menjadi prioritas utama. Setiap mesin pesawat, khususnya mesin jet, memiliki umur pakai yang terbatas dan membutuhkan pemeliharaan rutin untuk memastikan kinerjanya tetap optimal. Namun, pemeliharaan yang dilakukan terlalu sering atau tidak tepat waktu dapat menyebabkan pemborosan biaya, sementara pemeliharaan yang terlambat bisa berisiko terhadap keselamatan penerbangan. Oleh karena itu, Predictive Maintenance (Pemeliharaan Prediktif) menjadi solusi yang sangat penting untuk memitigasi masalah ini.

Pemeliharaan prediktif adalah pendekatan yang menggunakan data dan analitik untuk memprediksi kapan suatu mesin atau komponen akan mengalami kerusakan, sehingga tindakan pemeliharaan dapat dilakukan tepat waktu, sebelum kerusakan terjadi. Pendekatan ini sangat bergantung pada analisis data yang dihasilkan selama operasional mesin. Oleh karena itu, untuk meningkatkan efisiensi dan menurunkan biaya pemeliharaan, penting untuk mengembangkan model prediktif yang akurat.

Dataset NASA Turbofan Engine Degradation Simulation (CMAPSS) adalah salah satu dataset yang banyak digunakan dalam penelitian mengenai pemeliharaan prediktif. Dataset ini berisi data simulasi kerusakan yang terjadi pada mesin turbofan pesawat. Data ini mencakup berbagai parameter sensor, seperti suhu, tekanan, dan kecepatan aliran udara yang merekam kondisi operasional mesin seiring waktu, dan bagaimana perubahan ini terkait dengan degradasi mesin.

Penelitian yang mengarah pada predictive maintenance berpotensi memberikan manfaat besar bagi industri penerbangan, karena dapat membantu memprediksi kerusakan mesin sebelum terjadi, mengurangi downtime pesawat, dan mengoptimalkan biaya pemeliharaan. Oleh karena itu, pemanfaatan teknologi machine learning dan deep learning untuk memproses data dari dataset CMAPSS menjadi langkah penting untuk mewujudkan sistem pemeliharaan prediktif yang lebih canggih dan efisien.

**Mengapa Masalah Ini Harus Diselesaikan**:
Masalah yang dihadapi oleh industri penerbangan terkait pemeliharaan mesin sangat krusial, baik dari segi biaya maupun keselamatan. Tanpa pemeliharaan yang optimal, ada risiko besar terjadinya kerusakan mesin secara tak terduga, yang dapat berujung pada kecelakaan atau kerugian finansial yang signifikan. Mengingat tingginya biaya yang terkait dengan pemeliharaan dan perbaikan mesin pesawat, serta pentingnya keselamatan dalam penerbangan, maka penerapan predictive maintenance melalui pemanfaatan data sensor dan analitik menjadi hal yang mendesak untuk dilakukan.

Dengan adanya model prediktif, pihak operator pesawat dan perusahaan penerbangan dapat melakukan pemeliharaan hanya pada waktu yang diperlukan dan lebih tepat, yang pada gilirannya akan meningkatkan efisiensi operasional dan menurunkan biaya pemeliharaan.

**Riset Terkait**:

- Salah satu studi penting dalam pengembangan sistem predictive maintenance dilakukan oleh NASA Ames Research Center, melalui simulasi degradasi mesin turbofan. Dataset Turbofan Engine Degradation Simulation yang dikembangkan oleh Saxena dan Goebel (2008) memberikan data sensor yang merekam kondisi mesin selama siklus operasionalnya hingga terjadi kegagalan. Dataset ini banyak digunakan sebagai benchmark dalam pengembangan model Remaining Useful Life (RUL) prediction, yaitu model yang bertujuan untuk memperkirakan sisa umur pakai mesin sebelum mencapai titik kegagalan.
    
     Format Referensi: [Saxena, A., & Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository. Retrieved from](https://scholar.google.com/)
- Model prediktif yang akurat terhadap RUL memiliki nilai strategis dalam sistem industri modern, khususnya dalam sistem pemeliharaan mesin pesawat dan alat berat lainnya. McMillan dan Ault (2017) menegaskan bahwa penerapan PdM tidak hanya meningkatkan efisiensi operasional, tetapi juga mendukung keselamatan dan ketahanan sistem secara keseluruhan.
 
     Format Referensi: [McMillan, D., & Ault, G. (2017). Condition Monitoring on Rotating Electrical Machines. IEE Colloquium on Instrumentation of Rotating Electrical Machines, 32(2), 593–601.](https://ieeexplore.ieee.org/document/181056)

## Business Understanding

Pada bagian ini, kita akan mengklarifikasi masalah yang dihadapi dan tujuan yang ingin dicapai melalui proyek predictive maintenance menggunakan dataset NASA Turbofan Engine Degradation Simulation (CMAPSS). Bagian ini bertujuan untuk mendefinisikan masalah yang akan diselesaikan, tujuan yang ingin dicapai, serta cara-cara yang dapat digunakan untuk mencapai tujuan tersebut.

### Problem Statements

Pernyataan masalah latar belakang:
- Bagaimana cara memprediksi kerusakan mesin turbofan pesawat secara akurat menggunakan data sensor yang tersedia, sehingga pemeliharaan dapat dilakukan secara tepat waktu untuk menghindari kerusakan tak terduga?

- Algortima Model Machine Learning apa yang paling optimal untuk melakukan prediksi pemeliharaan berdasarkan data sensor?

### Goals

Tujuan dari pernyataan masalah:
- Mengembangkan model machine learning yang dapat memprediksi kapan suatu mesin akan mengalami kerusakan berdasarkan data sensor yang tersedia.

- Mendapatkan hasil terbaik dari perbandingan performa algoritma yang digunakan antara Gated Recurrent Unit (GRU) dan Long Short-Term Memory (LSTM).

    ### Solution statements
    - Menggunakan dataset CMAPSS untuk membangun model yang dapat memprediksi waktu ke depan sampai terjadinya kerusakan, dengan akurasi prediksi yang tinggi, sehingga pemeliharaan mesin bisa dilakukan tepat waktu.
    
    - Mengidentifikasi dan memilih algoritma yang memberikan performa terbaik dalam memprediksi degradasi mesin, dan membangun model dengan nilai MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error) yang rendah dalam memprediksi siklus kerusakan mesin yang akan terjadi. Membandingkan performa Gated Recurrent Unit (GRU) dan Long Short-Term Memory (LSTM), untuk memprediksi kapan mesin akan gagal berdasarkan urutan waktu degradasi.
    
    - Nilai Metrik dalam memprediksi waktu kegagalan dan kerusakan mesin. Metrik yang digunakan berupa MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error) untuk prediksi kerusakan mesin.

## Data Understanding
NASA Turbofan Engine Degradation Simulation (CMAPSS) dataset dirancang untuk tujuan penelitian terkait pemeliharaan prediktif pada mesin turbofan pesawat terbang. Dataset ini dapat diunduh melalui situs resmi NASA Prognostics Center of Excellence atau melalui platform lainnya yang menyediakan akses ke dataset CMAPSS. Dataset CMAPSS terdiri dari data simulasi yang menggambarkan operasi mesin turbofan pada berbagai kondisi. Data ini mengandung informasi yang dihasilkan oleh berbagai sensor mesin selama masa operasional, yang mencatat degradasi mesin pada berbagai tingkat keparahan. 

- **Sumber dataset:** [NASA Turbofan Engine Degradation Simulation C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data).
- **Jumlah Dataset:** berisi 26 kolom dengan 20.630 baris data.
- **Missing Value:** Tidak terdapat missing value pada dataset ini
- **Duplicate Rows:** Tidak terdapat duplicate rows pada dataset ini.
- **File:** Dataset ini terdiri dari 4 data latih, 4 data test, dan 4 data RUL. Masing masing data tersebut mewakili kondisi yang berbeda. Namun pada proyek ini hanya akna digunakan 1 kondisi saja dengan nama file FD001.

Variabel-variabel pada NASA Turbofan Engine Degradation Simulation C-MAPSS adalah sebagai berikut:
- **unit_number** : Menunjukkan identitas unik untuk setiap mesin yang diuji.
- **time_in_cycles** : Urutan waktu/siklus terbang (1, 2, 3, … hingga rusak)
- **operational_setting_1-3** : Kondisi operasional (misalnya tekanan, suhu lingkungan, dll. Namun tidak dijelaskan secara detail oleh NASA)
- **sensor_1** s/d **sensor_21**: Data dari berbagai sensor internal mesin (getaran, suhu, tekanan, dll)

### Exploratory Data Analysis

**1. Melihat Distribusi Data Siklus Mesin.**

![Distirbusi waktu hidup mesin](https://github.com/user-attachments/assets/02c22faf-2f55-4c8c-b4c1-8f18aed8dc80)
![Distirbusi waktu hidup mesin describe](https://github.com/user-attachments/assets/ffe063ee-9a36-426a-b0fe-c279279fe77a)

- Rata-rata Waktu Hidup Mesin (mean): Rata-rata waktu hidup mesin adalah sekitar 206 siklus, dengan rentang nilai antara 128 hingga 362 siklus.
- Variasi Waktu Hidup Mesin (std): Ada variasi yang cukup besar (sekitar 46 siklus) antara mesin yang memiliki waktu hidup pendek dan panjang.
- Median (50%): Nilai median dari waktu hidup mesin adalah 199 siklus, yang sedikit lebih rendah dari rata-rata, menunjukkan bahwa sebagian besar mesin memiliki waktu hidup yang relatif lebih pendek daripada rata-rata.
- Distribusi: Dalam histogram, Anda dapat melihat bahwa sebagian besar mesin cenderung gagal dalam rentang 170-230 siklus, dengan beberapa mesin yang memiliki waktu hidup yang jauh lebih panjang, mendekati nilai maksimum 362 siklus.

**2. Distibusi RUL.**

![Distribusi RUL](https://github.com/user-attachments/assets/cf08fe79-b3d3-4e4f-b838-935ae1316e5e)
![Distribusi RUL desribe](https://github.com/user-attachments/assets/3bd0110d-eb9f-446c-80f6-13667be03ed9)

- Sebagian besar data memiliki nilai **RUL** rendah-menengah, dan hanya sedikit data yang memiliki RUL sangat tinggi (dekat 361).
- Banyak data dikumpulkan hingga mesin benar-benar gagal.
- Dengan 25% = 51 dan 75% = 155, sebagian besar nilai **RUL** terletak dalam rentang tersebut. Artinya model lebih baik dilatih pada rentang ini.

**3. Korelasi Sensor Terhadap RUL.**

![Korelasi sensor terhadap RUL](https://github.com/user-attachments/assets/66fb14b3-f22a-4b22-9fd0-21bc9f7af1c5)

- Mayoritas sensor memiliki korelasi yang cukup tinggi dan seimbang dengan **RUL**
- **Sensor 12** memiliki korelasi positive paling tinggi dan **Sensor 11** memiliki korelasi negative paling rendah terhadap **RUL**
- **Sensor 1,5,10,16,18,19** tidak memiliki korelasi sama sekali dengan **RUL**. kita akan melihat nilai distribusi sensor tersebut dibawah ini.

![Distribusi sensor yang tidak berkorelasi denagn RUL](https://github.com/user-attachments/assets/b5c02643-6eab-470c-83c7-dfe6c2d0c34f)

- sensor-sensor seperti **sensor_1, sensor_10, sensor_16, sensor_18, dan sensor_19** memiliki masalah distribusi yang signifikan, yang menyebabkan mereka tidak memiliki korelasi yang jelas dengan **RUL**.

## Data Preparation
Pada tahap Data Preparation, kita akan mempersiapkan data untuk analisis dan pembangunan model prediktif. Tahap ini sangat penting untuk memastikan bahwa data yang digunakan dalam pembangunan model telah diproses dengan baik dan siap untuk digunakan dalam algoritma machine learning. Berikut adalah langkah-langkah yang dilakukan dalam tahap ini, beserta penjelasan mengenai teknik yang digunakan:

**1. Normalisasi MinMaxScaller**: 
- **Deskripsi:** Data sensor dalam dataset memiliki rentang nilai yang sangat berbeda, sehingga normalisasi diperlukan untuk memastikan bahwa setiap fitur berada dalam skala yang sama. Pada tahap ini, kita akan menggunakan MinMaxScaler yang merubah data menjadi skala antara 0 dan 1. MinMaxScaler sangat berguna untuk data yang memiliki distribusi dengan rentang nilai yang berbeda.
- **Alasan:** Normalisasi penting untuk menghindari dominasi fitur dengan skala yang lebih besar dan memastikan bahwa model machine learning, khususnya LSTM, dapat bekerja lebih baik dengan data yang sudah terstandarisasi.

**2. Penghapusan Data yang Tidak Diperlukan**: 
- **Deskripsi:** Beberapa fitur dalam dataset mungkin tidak relevan untuk analisis atau tidak memberikan kontribusi signifikan terhadap model. Sebagai langkah pertama, akan dilakukan penghapusan kolom yang tidak diperlukan, seperti kolom Sensor 1,5,10,16,18,19 yang tidak memiliki korelasi terhadap RUL.
- **Alasan:** Penghapusan fitur yang tidak relevan dapat mengurangi kompleksitas model, meningkatkan kecepatan pelatihan, dan menghindari overfitting.

**3. Menentukan Kolom Fitur dan Label**: 
- **Deskripsi:** Setelah melakukan normalisasi dan penghapusan kolom yang tidak relevan, langkah selanjutnya adalah menentukan fitur (features) dan label untuk model. Fitur terdiri dari data sensor yang mencatat parameter operasional mesin, sedangkan label adalah RUL (Remaining Useful Life), yang merupakan target yang ingin diprediksi.
- **Alasan:** Memisahkan fitur dan label adalah langkah penting untuk melatih model machine learning, karena model perlu mengetahui input (fitur) dan output yang diinginkan (label) untuk belajar dari data.

**4. Truncation / Capping RUL**: 
- **Deskripsi:** Beberapa mesin dalam dataset dapat memiliki nilai RUL yang sangat tinggi, yang dapat mengarah pada distribusi yang sangat miring atau skewed. Untuk mengatasi hal ini, kita akan menerapkan teknik truncation atau capping pada nilai RUL untuk membatasi rentang nilai maksimum dan mengurangi bias akibat nilai yang sangat tinggi.
- **Alasan:** Truncation atau capping diperlukan untuk memastikan bahwa model tidak terbebani oleh nilai RUL yang ekstrem yang tidak realistis, serta untuk memperbaiki distribusi data agar lebih seimbang.

**5. Buat sequence data untuk LSTM dan GRU (windowing)**: 
- **Deskripsi:** Model LSTM dan GRU memerlukan data dalam bentuk urutan waktu (sequences). Oleh karena itu, kita perlu membuat data dalam format windowing atau sequence di mana setiap sequence terdiri dari sejumlah titik data sensor yang berurutan. Proses ini dikenal dengan istilah sliding window.
- **Alasan:** LSTM dan GRU adalah model yang sangat baik untuk memproses data time-series, dan membutuhkan data yang disusun dalam bentuk urutan. Dengan membuat window data, kita memberikan model informasi mengenai bagaimana data sensor berubah seiring waktu, yang membantu memprediksi kegagalan mesin dengan lebih akurat.

**6. Pemisahan Data (Train-Test Split)**: 
- **Deskripsi:** Setelah data diproses menjadi format sequence, langkah terakhir adalah membagi dataset menjadi dua bagian: data pelatihan (training data) dan data pengujian (testing data). Ini dilakukan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.
- **Alasan:** Pemisahan data penting untuk menghindari overfitting dan untuk menguji kemampuan model dalam memprediksi dengan data baru yang belum pernah dilihat sebelumnya.

## Modeling
Tahap Modeling merupakan inti dari proyek predictive analytics ini. Di tahap ini, kita membangun dan melatih model machine learning untuk memprediksi Remaining Useful Life (RUL) dari mesin turbofan berdasarkan data sensor. Karena data yang digunakan bersifat time-series dan berkorelasi temporal, kita memilih algoritma Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU).

### Model LSTM
#### Cara Kerja LSTM
LSTM adalah jenis RNN yang dapat mengatasi masalah vanishing gradient yang terjadi pada RNN tradisional saat memproses urutan panjang. LSTM memiliki sel memori (memory cell) yang mampu menyimpan informasi lebih lama dan secara selektif mengupdate atau melupakan informasi berdasarkan kondisi yang ada.

#### Parameter LSTM
- Input: data time-series sensor (dalam window) dari setiap unit mesin.
- Menggunakan jumlah unit/neuron 64.
- Arsitektur: 2 lapisan LSTM dengan dropout, lalu dense 1 output.
- Dropout sebesar 20% untuk mencegah overfitting dengan menghilangkan acak neuron saat training.
- Layer dense (fully connected) dengan 1 unit

#### Kelebihan LSTM
- LSTM Sangat cocok untuk memproses dan memprediksi data urutan seperti sensor mesin yang berubah secara berurutan dari waktu ke waktu.
- LSTM memiliki cell state yang memungkinkan model untuk menyimpan informasi lebih lama.

#### Kekurangan LSTM
- LSTM lebih kompleks dibandingkan dengan GRU karena memiliki lebih banyak parameter.
- Karena memiliki banyak parameter, LSTM cenderung lebih rentan terhadap overfitting

### Model GRU
#### Cara Kerja GRU
GRU adalah varian yang lebih sederhana dari LSTM, yang juga bertujuan untuk mengatasi masalah vanishing gradient dan menyimpan informasi dalam urutan data. GRU menggunakan dua gerbang utama yang lebih sederhana dibandingkan LSTM, namun tetap efektif dalam banyak aplikasi.

#### Parameter GRU
- Input: data time-series sensor (dalam window) dari setiap unit mesin.
- Menggunakan jumlah unit/neuron 64.
- Arsitektur: 2 lapisan GRU dengan dropout, lalu dense 1 output.
- Dropout sebesar 20% untuk mencegah overfitting dengan menghilangkan acak neuron saat training.
- Layer dense (fully connected) dengan 1 unit

#### Kelebihan GRU
- GRU sangat efektif dalam mempelajari hubungan antar waktu sehingga bisa mendeteksi degradasi mesin secara bertahap.
- Dengan lebih sedikit parameter, GRU lebih efisien dalam hal komputasi dan memori, membuatnya lebih cocok untuk aplikasi dengan sumber daya terbatas.

#### Kekurangan GRU
- Karena tidak memiliki cell state seperti LSTM, GRU mungkin kurang baik dalam menangani ketergantungan jangka panjang dalam data yang sangat panjang atau kompleks.
- GRU bisa lebih terbatas dalam hal penyaringan informasi dibandingkan dengan LSTM.

### Optimizer
Kedua algoritma menggunakan Optimizer Adam karena cepat, stabil, adaptif, dan mudah digunakan, sangat cocok untuk training model GRU/LSTM dalam predictive maintenance.

### Callbacks
#### 1. ReduceLROnPlateau:
- Menurunkan learning rate secara otomatis jika model mengalami stagnasi dalam validasi.
- monitor='val_loss': Pantau nilai loss pada data validasi.
- factor=0.5: Jika tidak ada perbaikan, kurangi learning rate jadi setengah dari nilai sebelumnya.
- patience=5: Tunggu 5 epoch tanpa perbaikan sebelum mengurangi.
- min_lr=1e-6: Batasi minimum learning rate agar tidak terlalu kecil.
#### 2. early_stop:
- Menghentikan pelatihan lebih awal jika model tidak membaik untuk menghindari overfitting dan pemborosan waktu.
- patience=10: Tunggu maksimal 10 epoch tanpa peningkatan sebelum berhenti.
- restore_best_weights=True: Kembalikan bobot terbaik model yang diperoleh selama pelatihan (bukan bobot di akhir epoch).
#### 3. model_checkpoint:
- Menyimpan model ke file setiap kali model mencapai performa validasi terbaik.
- save_best_only=True: Hanya menyimpan model jika val_loss membaik.

## Evaluation

### **Metrik Yang digunakan**
Pada proyek ini, model yang dibangun bertujuan untuk memprediksi nilai Remaining Useful Life (RUL) dari mesin turbofan. Karena target prediksi bersifat kontinu, maka pendekatan yang digunakan adalah regresi. Oleh karena itu, metrik evaluasi yang digunakan adalah metrik regresi, yaitu:
#### **a. Mean Absolute Error (MAE)**
MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual (target). MAE memberikan gambaran sederhana tentang seberapa besar kesalahan prediksi dalam satuan unit RUL. Semakin kecil nilai yang ditunjukan, semakin baik.
#### **b. Mean Squared Error (MSE)**
MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. MSE memberi penalti yang lebih besar terhadap kesalahan yang lebih besar, karena kesalahan dihitung dalam kuadrat. Semakin kecil nilai yang ditunjukan, semakin baik.
#### **c. Root Mean Squared Error (RMSE)**
RMSE adalah akar kuadrat dari MSE dan memberikan gambaran lebih intuitif tentang kesalahan model dalam satuan yang sama dengan data asli (RUL). RMSE cenderung memberi penalti lebih besar pada kesalahan besar. Semakin kecil nilai yang ditunjukan, semakin baik.

### **Interpretasikan Hasil Proyek**

1. **MAE** Dalam proyek ini, MAE untuk LSTM adalah 27.97, sementara untuk GRU adalah 26.01. Artinya, prediksi model GRU lebih mendekati nilai RUL yang sebenarnya dibandingkan dengan model LSTM, karena nilai MAE-nya lebih kecil.
2. **MSE** Dalam proyek ini, MSE model LSTM adalah 1572.89 dan MSE model GRU adalah 1506.88. Nilai MSE yang lebih rendah pada model GRU menunjukkan bahwa GRU memiliki prediksi yang lebih baik dan lebih konsisten dibandingkan LSTM.
3. **RMSE** LSTM adalah 39.66, sedangkan RMSE GRU adalah 38.82. Meskipun perbedaannya kecil, RMSE yang lebih rendah pada GRU menunjukkan bahwa model GRU sedikit lebih akurat dalam memprediksi RUL dibandingkan dengan LSTM.

Berdasarkan metrik evaluasi yang digunakan, model GRU menunjukkan performa yang lebih baik dibandingkan dengan LSTM dalam hal mengurangi kesalahan prediksi nilai RUL, baik itu dalam bentuk MAE, MSE, maupun RMSE. Model GRU mampu melakukan prediksi mendekati nilai asli dari RUL sehingga meminimalisir kesalahan dalam melakukan predictive maintenance.

### **Plot Prediksi vs Nilai Aktual (visualisasi kinerja model)**

![Actual vs Predicted RUL](https://github.com/user-attachments/assets/18d18a67-b173-4d11-bbe6-093e938df0cc)


- Grafik perbandingan Actual RUL dengan Predicted RUL oleh LSTM dan GRU menunjukan hasil prediksi model yang sangat baik. Predicted RUL berhasil mengikuti pola pada Actual RUL.
- Perbedaan pola antara model LSTM dan GRU tidak terlalu signifikan dan menunjukan pola yang mirip.

## Kesimpulan
Dalam pengembangan model predictive maintenance turbofan engine degradation diperlukan model machine learning yang dapat melakukan prediksi nilai RUL mendekati nilai aslinya. Hal ini dapat membantu keterlambatan proses maintenance pada turbofan engine yang menyebabkan kegagalan pada engine.

Hasil pengembangan model machine learning menggunakan LSTM dan GRU menunjukan performa yang sama bagusnya. Namun GRU mampu melakukan prediksi lebih baik dengan nilai kesalahan pada metrik MAE, MSE, maupun RMSE yang sedikit lebih baik dengan menunjukan nilai kesalahan yang lebih kecil.

