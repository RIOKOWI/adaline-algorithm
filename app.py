import streamlit as st  # library untuk membuat web app sederhana
import numpy as np      # library untuk operasi numerik (array, dot product, dll)
import matplotlib.pyplot as plt  # library untuk membuat grafik

# Judul aplikasi di halaman web
st.title("Adaline Training")

# Slider untuk mengatur learning rate (alpha)
# User bisa geser dari 0.01 sampai 1.0, default 0.2
alpha = st.slider("Learning Rate", 0.01, 1.0, 0.2)

# Threshold untuk menghentikan training jika error sudah kecil
threshold = 0.001

# Data input (fitur)
# Setiap baris = 1 data, setiap kolom = fitur
X = np.array([[1, 0.5],
              [1, 1],
              [0, 0.5]])

# Target output
T = np.array([1, 1, 0])

# Inisialisasi bobot secara random (2 fitur → 2 bobot)
w = np.random.rand(2)

# Inisialisasi bias secara random
b = np.random.rand()

# List untuk menyimpan error tiap epoch (untuk grafik)
error_list = []

# Loop training (maksimal 100 epoch)
for epoch in range(100):
    total_error = 0  # reset total error setiap epoch

    # Loop setiap data
    for i in range(len(X)):
        # Hitung output linear (yin)
        # yin = bias + (x dot w)
        yin = b + np.dot(X[i], w)

        # Hitung error (target - output)
        e = T[i] - yin

        # Update bobot (aturan Adaline)
        w = w + alpha * e * X[i]

        # Update bias
        b = b + alpha * e

        # Akumulasi error kuadrat
        total_error += e**2

    # Hitung Mean Squared Error (MSE)
    mse = total_error / len(X)

    # Simpan error ke list
    error_list.append(mse)

    # Jika error sudah kecil → hentikan training
    if mse < threshold:
        break

# Tampilkan bobot akhir di web
st.write("Bobot akhir:", w)

# Tampilkan bias akhir
st.write("Bias akhir:", b)

# Buat grafik error
fig, ax = plt.subplots()

# Plot error per epoch
ax.plot(error_list)

# Label sumbu X
ax.set_xlabel("Epoch")

# Label sumbu Y
ax.set_ylabel("MSE")

# Tampilkan grafik di Streamlit
st.pyplot(fig)