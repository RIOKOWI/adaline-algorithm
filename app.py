import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Adaline Training")

alpha = st.slider("Learning Rate", 0.01, 1.0, 0.2)
threshold = 0.001

X = np.array([[1, 0.5],
              [1, 1],
              [0, 0.5]])
T = np.array([1, 1, 0])

w = np.random.rand(2)
b = np.random.rand()
error_list = []

for epoch in range(100):
    total_error = 0

    for i in range(len(X)):
        yin = b + np.dot(X[i], w)
        e = T[i] - yin

        w = w + alpha * e * X[i]
        b = b + alpha * e

        total_error += e**2

    mse = total_error / len(X)
    error_list.append(mse)

    if mse < threshold:
        break

st.write("Bobot akhir:", w)
st.write("Bias akhir:", b)

fig, ax = plt.subplots()
ax.plot(error_list)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")

st.pyplot(fig)