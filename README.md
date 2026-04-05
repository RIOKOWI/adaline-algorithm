# Dokumentasi Logika Program Adaline (Streamlit)

## 1. Inisialisasi Sistem
- Sistem menyiapkan:
  - Learning rate (α) dari input user (slider)
  - Threshold error = 0.001 (batas berhenti)
  - Data input (X)
  - Target (T)
- Bobot (w) dan bias (b) diinisialisasi secara acak (random)

**Makna:**
Model mulai dari kondisi awal tanpa pengetahuan.

---

## 2. Proses Training (Loop Epoch)
- Training dilakukan maksimal 100 epoch
- Setiap epoch:
  - `total_error` di-reset ke 0

**Makna:**
Satu epoch = satu siklus belajar dari seluruh data.

---

## 3. Iterasi Setiap Data

### a. Hitung Output (yin)
yin ​= b + ( x ⋅ w )

**Makna:**
Model menghasilkan prediksi berdasarkan bobot saat ini.

---

### b. Hitung Error
e = t − yin​

**Makna:**
Mengukur selisih antara target dan hasil prediksi.

---

### c. Update Bobot
w = w + α ⋅ e ⋅ x

**Makna:**
Bobot disesuaikan untuk mengurangi error.

---

### d. Update Bias
b = b + α ⋅ e

**Makna:**
Bias ikut dikoreksi agar prediksi lebih akurat.

---

### e. Akumulasi Error
total_error += e2

**Makna:**
Mengumpulkan total kesalahan dalam satu epoch.

---

## 4. Hitung MSE (Mean Squared Error)
MSE= jumlah_data / total_error​

**Makna:**
Rata-rata error dalam satu epoch.

---

## 5. Simpan Error
- Nilai MSE disimpan ke dalam `error_list`

**Makna:**
Digunakan untuk analisis performa model (grafik).

---

## 6. Kondisi Berhenti
Jika:
\[
MSE < 0.001
\]

→ Training dihentikan

**Makna:**
Model dianggap sudah cukup akurat.

---

## 7. Output Hasil
- Menampilkan:
  - Bobot akhir (w)
  - Bias akhir (b)

**Makna:**
Parameter hasil pembelajaran model.

---

## 8. Visualisasi
- Grafik:
  - Sumbu X = Epoch
  - Sumbu Y = MSE

**Makna:**
- Grafik turun → model belajar dengan baik  
- Grafik datar → model tidak belajar  
- Grafik tidak stabil → parameter tidak tepat  

---

## Ringkasan Logika
1. Hitung prediksi (yin)  
2. Hitung error  
3. Update bobot dan bias  
4. Ulangi sampai error kecil  

---

## Catatan Penting
- Learning rate terlalu besar → training tidak stabil  
- Learning rate terlalu kecil → training lambat  
- Data tidak sesuai → model sulit konvergen  


## Installation

```bash
git clone https://github.com/RIOKOWI/adaline-algorithm.git

cd adaline-algorithm

pip install -r requirements.txt

streamlit run app.py
```