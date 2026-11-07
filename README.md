# Prediksi Penjualan Mingguan Supermarket Menggunakan Random Forest Regressor

Proyek ini memprediksi penjualan mingguan 45 toko Walmart menggunakan **Random Forest Regressor**, lalu membandingkannya dengan **XGBoost** dan **model naif** (penjualan minggu sebelumnya). Perbedaan kesalahan antar model diuji dengan **uji Wilcoxon signed-rank**.

Proyek ini dibuat sebagai tugas akhir mata kuliah Kecerdasan Buatan.

## Isi Folder

```
.
├── Dataset_Walmart.csv  # Dataset penjualan mingguan Walmart (Kaggle, lisensi CC0)
├── prediction.ipynb     # Notebook utama: preprocessing, pelatihan, evaluasi, dan uji Wilcoxon
├── visualisasi/         # Hasil gambar dari notebook (7 gambar)
└── README.md            # Dokumen ini
```

### Isi folder `Visualisasi/`

| No | Gambar | Keterangan |
|---|---|---|
| 1 | penjualan aktual vs prediksi tanpa sales lag.png | Penjualan aktual vs prediksi naif, Random Forest, dan XGBoost tanpa Sales_Lag |
| 2 | Perbandingan MAE, RMSE, dan R2 ketiga algoritma.png | Penjualan aktual vs prediksi naif, Random Forest, dan XGBoost dengan Sales_Lag |
| 3 | penjualan aktual vs prediksi dengan sales lag.png | Perbandingan tiga metrik untuk naif, Random Forest, dan XGBoost tanpa Sales_Lag |
| 4 | Perbandingan MAE, RMSE, dan R2 dengan sales lag.png | Perbandingan tiga metrik untuk naif, Random Forest + lag, dan XGBoost + lag |
| 5 | output hasil uji wilcoxon setiap algoritma | Progress bar p-value setiap pasangan model terhadap batas 0,005 |
| 6 | Boxplot tanpa lag.png | Sebaran error absolut dengan jumlah outlier |
| 7 | Boxplot dengan lag.png | Sebaran error absolut dengan jumlah outlier |

## Kebutuhan

Notebook dirancang untuk dijalankan di **Jupyter Notebook berbasis web**, seperti JupyterLab di browser atau Google Colab. Tidak perlu memasang Python di komputer lokal.

Library yang dipakai: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, dan `matplotlib`. Sebagian besar sudah tersedia di lingkungan Jupyter web, tetapi `xgboost` biasanya perlu dipasang terlebih dahulu (lihat langkah 3 di bawah).

## Cara Menjalankan di Jupyter Web

### Langkah 1: Siapkan lingkungan

Gunakan salah satu dari dua cara berikut:

- **JupyterLab di browser** (misalnya dari server kampus atau layanan Jupyter yang disediakan dosen). Buka alamatnya di browser, lalu login jika diminta.
- **Google Colab**. Buka [colab.research.google.com](https://colab.research.google.com), lalu pilih **File → Upload notebook** dan unggah `prediction.ipynb`.

### Langkah 2: Unggah file ke lingkungan Jupyter

Unggah `Dataset_Walmart.csv` dan `prediction.ipynb` ke folder yang sama.

- **JupyterLab:** klik tombol **Upload** (ikon panah ke atas) di panel File Browser, lalu pilih kedua file.
- **Google Colab:** klik ikon **folder** di panel kiri, lalu klik **Upload to session storage** dan pilih `Dataset_Walmart.csv`. Panel ini adalah tempat notebook membaca file. Jika sesi Colab dimulai ulang, file harus diunggah lagi.

Notebook membaca dataset dengan nama `Dataset_Walmart.csv` tanpa path tambahan, sehingga file harus berada di folder kerja yang sama dengan notebook.

### Langkah 3: Pasang library yang kurang

Jalankan sel berikut di bagian paling atas notebook, lalu tunggu sampai selesai:

```python
%pip install xgboost scipy matplotlib scikit-learn
```

Jika library sudah terpasang, sel ini tetap aman dijalankan. Setelah instalasi, restart kernel dengan menu **Kernel → Restart Kernel** (atau **Runtime → Restart session** di Colab) sebelum melanjutkan.

### Langkah 4: Jalankan seluruh notebook

Gunakan **Kernel → Restart Kernel and Run All Cells** (atau **Runtime → Run all** di Colab). Jika menjalankan sel satu per satu, ikuti urutan dari atas ke bawah, karena setiap sel memakai variabel dari sel sebelumnya.

Pastikan tidak ada sel yang menampilkan error `FileNotFoundError` untuk `Dataset_Walmart.csv`. Jika muncul, periksa kembali bahwa file berada di folder yang sama dengan notebook.

### Langkah 5: Lihat hasil

Hasil akan tampil langsung di bawah setiap sel, berupa:

- Ringkasan pembagian data latih dan uji
- MAE, RMSE, dan R² untuk setiap model
- Hasil uji Wilcoxon untuk setiap pasangan model
- Visualisasi prediksi, metrik, p-value, dan boxplot

Runtime biasanya selesai dalam beberapa menit. Sel yang melatih Random Forest dan XGBoost berulang kali menjadi bagian yang paling lama.

## Alur Program

Notebook dibagi menjadi enam langkah berurutan:

| Langkah | Isi |
|---|---|
| 1 | Membaca `Dataset_Walmart.csv`, mengubah format tanggal, membuat fitur `Month`, `WeekOfYear`, dan `Sales_Lag1`, lalu membagi data 80% latih dan 20% uji secara kronologis |
| 2 | Melatih Random Forest tanpa lag dan mengevaluasinya |
| 3 | Melatih XGBoost tanpa lag dan mengevaluasinya |
| 4 | Mengevaluasi model naif (penjualan minggu sebelumnya) |
| 5 | Melatih Random Forest dan XGBoost dengan `Sales_Lag1`, lalu membandingkannya dengan naif |
| 6 | Menguji perbedaan error setiap pasangan model dengan uji Wilcoxon (batas signifikansi 0,005) |

## Hasil Utama (rasio 80/20)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naif (minggu sebelumnya) | 54.343,62 | 82.736,58 | 0,9760 |
| Random Forest tanpa lag | 62.754,72 | 92.488,51 | 0,9700 |
| XGBoost tanpa lag | 65.403,34 | 95.240,62 | 0,9681 |
| Random Forest + lag | 49.817,85 | 75.420,61 | 0,9800 |
| XGBoost + lag | 45.364,60 | 66.887,58 | 0,9843 |

Uji Wilcoxon pada rasio 80/20 menunjukkan bahwa ketiga perbedaan antar pasangan model signifikan pada batas 0,005.

## Catatan Penting

- **Model naif** adalah baseline deret waktu, bukan algoritma Naive Bayes.
- **Uji Wilcoxon** membandingkan perbedaan error antar pasangan model. Uji ini tidak menyatakan bahwa satu algoritma akurat secara terpisah.
- **Nilai z negatif** terjadi karena W diambil sebagai nilai terkecil dari W+ dan W−. Tanda ini tidak berarti model yang dibandingkan kalah.
- **Indikator ekonomi** (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`) tidak dipakai dalam model utama, karena menurunkan akurasi pada data uji ini.
- **Hasil berlaku untuk periode uji yang diteliti** (2012) dan belum tentu berlaku untuk periode atau negara lain.
- Angka kecil bisa sedikit berbeda antar lingkungan karena perbedaan versi library, terutama `xgboost` dan `scikit-learn`.

## Sumber Data

Dataset: *Walmart Recruiting - Store Sales Forecasting*, tersedia di Kaggle dengan lisensi CC0 (Public Domain).