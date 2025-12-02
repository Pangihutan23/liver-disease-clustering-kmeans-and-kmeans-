# Analisis Cluster Pasien HCV K-Means VS K-Means++  

Analisis clustering data HCV ini menggunakan dua algoritma K-Means. Proyek ini berfokus pada perbandingan performa antara K-Means (inisialisasi random) dan K-Means++ dalam mengelompokkan pasien berdasarkan parameter klinis.  

## Tujuan
- Membandingkan efektivitas K-Means VS K-Means++ pada data medis
- Menentukan jumlah klaster optimal dengan elbow method dan silhouette score
- Menganalisis karakteristik setiap cluster
- Membust Visualisasi hasil cluster dengan PCA  

# Tahapan Implementasi  
1. Load data: Membaca dan inspeksi dataset awal
2. Preprocesing:   
- encoding variabel
- menemukan missing value
- standarisasi fitur numerik  
3. Analisis cluster:  
- Elbow Method  
- Silhouette score  
4. clustering:  
- K-means dengan random initialization  
- K-Means++ dengan smart initialization  
5. Evaluasi: MSE (Mean Squared Error)  
6. Visualisasi: PCA  

# Fitur Utama
- Perbandingan lengkap dua metode K-Means dalam satu eksekusi
- Validasi ganda menggunakan Elbow method dan Silhouette Score
- Output terstruktur dengan tabel dan statistik
- PCA yang terpisah untuk setiap metode  

# Catatan  
**Kode ini dibuat hanya untuk pembelajaran**

