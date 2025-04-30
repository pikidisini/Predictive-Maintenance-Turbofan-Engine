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

- Bagaimana cara memilih dan menerapkan algoritma machine learning atau deep learning yang tepat untuk membangun model prediktif yang mampu mengenali pola degradasi mesin dengan tingkat akurasi yang tinggi?

### Goals

Tujuan dari pernyataan masalah:
- Masalah utama yang dihadapi industri penerbangan adalah ketidakpastian dalam penjadwalan pemeliharaan mesin turbofan. Pemeliharaan yang terlalu sering atau terlambat dapat menyebabkan pemborosan biaya atau bahkan kerusakan fatal pada mesin. Oleh karena itu, diperlukan sistem yang dapat memprediksi kapan suatu mesin akan mengalami kerusakan berdasarkan data sensor yang tersedia.

- Dalam analisis prediktif, pemilihan algoritma yang tepat sangat penting untuk memastikan model yang dihasilkan memiliki akurasi yang tinggi. Beberapa algoritma yang digunakan, seperti Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), dan Random Forest, memiliki karakteristik berbeda dan dapat menghasilkan performa yang bervariasi. Memilih algoritma yang tepat dan menyesuaikan dengan dataset sangat penting untuk mencapai hasil yang optimal.

    ### Solution statements
    - Menggunakan dataset CMAPSS untuk membangun model yang dapat memprediksi waktu ke depan sampai terjadinya kerusakan, dengan akurasi prediksi yang tinggi, sehingga pemeliharaan mesin bisa dilakukan tepat waktu.
    
    - Mengidentifikasi dan memilih algoritma yang memberikan performa terbaik dalam memprediksi degradasi mesin, dan membangun model dengan akurasi lebih dari 85% dalam mengklasifikasikan tingkat kerusakan mesin. Mengintegrasikan Long Short-Term Memory (LSTM), yang sangat baik dalam menangani data urutan waktu, untuk memprediksi kapan mesin akan gagal berdasarkan urutan waktu degradasi.
    
    - Akurasi model dalam memprediksi waktu kegagalan dan tingkat kerusakan mesin. Metrik yang digunakan bisa berupa Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan F1-Score untuk klasifikasi kerusakan mesin.

## Data Understanding
Pada tahap ini, kita akan membahas informasi mengenai data yang digunakan dalam proyek ini, yaitu NASA Turbofan Engine Degradation Simulation (CMAPSS) dataset. Dataset ini dirancang untuk tujuan penelitian terkait pemeliharaan prediktif pada mesin turbofan pesawat terbang. Dataset ini dapat diunduh melalui situs resmi NASA Prognostics Center of Excellence atau melalui platform lainnya yang menyediakan akses ke dataset CMAPSS.

Sumber: [NASA Turbofan Engine Degradation Simulation C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data).

**Deskripsi Umum Dataset:**
Dataset CMAPSS terdiri dari data simulasi yang menggambarkan operasi mesin turbofan pada berbagai kondisi. Data ini mengandung informasi yang dihasilkan oleh berbagai sensor mesin selama masa operasional, yang mencatat degradasi mesin pada berbagai tingkat keparahan. Data simulasi ini digunakan untuk membangun model prediksi yang mampu memprediksi kapan mesin akan mengalami kerusakan.

Dataset CMAPSS terbagi menjadi beberapa bagian, yang masing-masing mencakup data untuk tiga jenis skenario degradasi yang berbeda, serta dua jenis data (train dan test).

### Variabel-variabel pada NASA Turbofan Engine Degradation Simulation C-MAPSS adalah sebagai berikut:
- **unit_number** : Menunjukkan identitas unik untuk setiap mesin yang diuji.
- **time_in_cycles** : Urutan waktu/siklus terbang (1, 2, 3, … hingga rusak)
- **operational_setting_1-3** : Kondisi operasional (misalnya tekanan, suhu lingkungan, dll. Namun tidak dijelaskan secara detail oleh NASA)
- **sensor_1** s/d **sensor_21**: Data dari berbagai sensor internal mesin (getaran, suhu, tekanan, dll)

## Exploratory Data Analysis
Pada tahap ini, akan menggali lebih jauh mengenai informasi yang dapat diperoleh dari dataset.

**1. Ekstrak informasi Data.**
- Menampilkan informasi data, seperti nama kolom, tipe data setiap fitur, jumlah kolom dan jumlah baris.

**2. Melihat Distribusi Data Siklus Mesin.**
- ata-rata Waktu Hidup Mesin (mean): Rata-rata waktu hidup mesin adalah sekitar 206 siklus, dengan rentang nilai antara 128 hingga 362 siklus.
- Variasi Waktu Hidup Mesin (std): Ada variasi yang cukup besar (sekitar 46 siklus) antara mesin yang memiliki waktu hidup pendek dan panjang.
- Median (50%): Nilai median dari waktu hidup mesin adalah 199 siklus, yang sedikit lebih rendah dari rata-rata, menunjukkan bahwa sebagian besar mesin memiliki waktu hidup yang relatif lebih pendek daripada rata-rata.
- Distribusi: Dalam histogram, Anda dapat melihat bahwa sebagian besar mesin cenderung gagal dalam rentang 170-230 siklus, dengan beberapa mesin yang memiliki waktu hidup yang jauh lebih panjang, mendekati nilai maksimum 362 siklus.

**3. Plot Sensor Terhadap Waktu Siklus.**
- Grafik menunjukan bahwa semakin banyak siklus waktu pada mesin (mesin digunakan), nilai pada **Sensor 2** unit 10 meningkat.

**4. Menghitung Remaining Useful Life (RUL).**
- Karena Dataset **train_FD001** tidak memiliki label RUL (Remaining Useful Life), maka nilai RUL perlu dihitung.

**4. Distibusi RUL.**
- Sebagian besar data memiliki nilai **RUL** rendah-menengah, dan hanya sedikit data yang memiliki RUL sangat tinggi (dekat 361).
- Banyak data dikumpulkan hingga mesin benar-benar gagal.
- Dengan 25% = 51 dan 75% = 155, sebagian besar nilai **RUL** terletak dalam rentang tersebut. Artinya model lebih baik dilatih dan diuji secara cermat pada rentang ini.

**5. Korelasi Sensor Terhadap RUL.**
- Mayoritas sensor memiliki korelasi yang cukup tinggi dan seimbang dengan **RUL**
- **Sensor 12** memiliki korelasi positive paling tinggi dan **Sensor 11** memiliki korelasi negative paling rendah terhadap **RUL**
- **Sensor 1,5,10,16,18,19** tidak memiliki korelasi sama sekali dengan **RUL**. kita akan melihat nilai distribusi sensor tersebut dibawah ini.
- sensor-sensor seperti **sensor_1, sensor_10, sensor_16, sensor_18, dan sensor_19** memiliki masalah distribusi yang signifikan, yang menyebabkan mereka tidak memiliki korelasi yang jelas dengan **RUL**.
- Persebaran Unit dengan **RUL** tinggi memiliki nilai pada **Sensor 11** yang rendah. Persebaran Unit dengan **RUL** tinggi memiliki nilai pada **Sensor 11** yang tinggi.

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

**5. Buat sequence data untuk LSTM (windowing)**: 
- **Deskripsi:** Model LSTM (Long Short-Term Memory) memerlukan data dalam bentuk urutan waktu (sequences). Oleh karena itu, kita perlu membuat data dalam format windowing atau sequence di mana setiap sequence terdiri dari sejumlah titik data sensor yang berurutan. Proses ini dikenal dengan istilah sliding window.
- **Alasan:** LSTM adalah model yang sangat baik untuk memproses data time-series, dan membutuhkan data yang disusun dalam bentuk urutan. Dengan membuat window data, kita memberikan model informasi mengenai bagaimana data sensor berubah seiring waktu, yang membantu memprediksi kegagalan mesin dengan lebih akurat.

**6. Pemisahan Data (Train-Test Split)**: 
- **Deskripsi:** Setelah data diproses menjadi format sequence, langkah terakhir adalah membagi dataset menjadi dua bagian: data pelatihan (training data) dan data pengujian (testing data). Ini dilakukan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.
- **Alasan:** Pemisahan data penting untuk menghindari overfitting dan untuk menguji kemampuan model dalam memprediksi dengan data baru yang belum pernah dilihat sebelumnya.
- 
## Modeling
Tahap Modeling merupakan inti dari proyek predictive analytics ini. Di tahap ini, kita membangun dan melatih model machine learning untuk memprediksi Remaining Useful Life (RUL) dari mesin turbofan berdasarkan data sensor. Karena data yang digunakan bersifat time-series dan berkorelasi temporal, kita memilih algoritma Long Short-Term Memory (LSTM) sebagai pendekatan utama.

**1. Pemilihan Algoritma: Long Short-Term Memory (LSTM)**
- **Alasan:** LSTM adalah jenis dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient dan mampu menangkap dependensi jangka panjang dalam data time-series. Sangat cocok untuk memproses dan memprediksi data urutan seperti sensor mesin yang berubah secara berurutan dari waktu ke waktu.

**2. Arsitektur Model LSTM (Baseline)** 
Berikut arsitektur model baseline yang digunakan:
- Input layer: urutan data sensor dengan dimensi (window_size, jumlah_fitur)
- LSTM layer 1: 100 unit
- Dropout layer: 0.2 (untuk mencegah overfitting)
- Dense layer (fully connected): 50 neuron, aktivasi ReLU
- Output layer: 1 neuron (output berupa nilai prediksi RUL)

**3. Training Model Baseline** 
- Epochs: 200
- Batch size: 64
- Loss function: Mean Squared Error (MSE), karena kita ingin meminimalkan kesalahan prediksi kuadrat.
- Optimizer: Adam, karena mampu menyesuaikan learning rate secara dinamis dan bekerja baik dalam banyak kasus.

**4. Evaluasi Model Baseline** 
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Plot Prediksi vs Nilai Aktual (visualisasi kinerja model)

**5. Improvement Model: Hyperparameter Tuning** 
- Eksperimen dengan jumlah unit LSTM: 64, 128, 256
- Hasil: Jumlah unit LSTM 64 memberikan keseimbangan terbaik antara akurasi dan waktu pelatihan
- Eksperimen ukuran window: 20, 30, 50
- Hasil: Window size 30 menghasilkan kinerja terbaik (lebih banyak informasi tanpa overfitting)
- Mencoba optimizer lain seperti RMSprop dan SGD, namun Adam tetap memberikan performa terbaik dalam eksperimen ini.

## Evaluation
Pada proyek ini, model yang dibangun bertujuan untuk memprediksi nilai Remaining Useful Life (RUL) dari mesin turbofan. Karena target prediksi bersifat kontinu, maka pendekatan yang digunakan adalah regresi, bukan klasifikasi. Oleh karena itu, metrik evaluasi yang digunakan adalah metrik regresi, yaitu:

**1. Mean Absolute Error (MAE)**
MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini mudah diinterpretasikan dan menunjukkan seberapa besar kesalahan rata-rata dalam satuan asli (setelah normalisasi bisa dikembalikan ke satuan RUL nyata).
- **Mean Absolute Error (MAE): 0.07** menunjukkan bahwa rata-rata kesalahan prediksi model hanya sekitar 0.07 unit dari nilai RUL sebenarnya (dalam skala normalisasi). Ini menunjukkan model memiliki akurasi yang tinggi dan mampu memahami pola degradasi mesin dengan baik.

**2. Root Mean Squared Error (RMSE)**
RMSE memberikan penalti yang lebih besar untuk kesalahan besar dibandingkan MAE karena mengkuadratkan error. Metrik ini membantu mengidentifikasi apakah model menghasilkan prediksi yang konsisten atau ada prediksi yang meleset jauh (outlier).
- **Root Mean Squared Error (RMSE): 0.11** menunjukkan bahwa prediksi juga stabil dan konsisten, karena tidak terdapat kesalahan besar atau outlier yang signifikan.


