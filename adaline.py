import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([[1, 0.5],
              [1, 1],
              [0, 0.5]])
T = np.array([1, 1, 0])

# Parameter
alpha = 0.2
threshold = 0.001
max_epoch = 100

# Inisialisasi
w = np.random.rand(2)
b = np.random.rand()
error_list = []

# Training
for epoch in range(max_epoch):
    total_error = 0

    for i in range(len(X)):
        yin = b + np.dot(X[i], w)
        e = T[i] - yin

        # update
        w = w + alpha * e * X[i]
        b = b + alpha * e

        total_error += e**2

    mse = total_error / len(X)
    error_list.append(mse)

    if mse < threshold:
        print(f"Berhenti di epoch {epoch+1}")
        break

# Plot error
plt.plot(error_list)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Grafik Error Adaline")
plt.show()