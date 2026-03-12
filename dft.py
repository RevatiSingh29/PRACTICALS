import numpy as np
import matplotlib.pyplot as plt

# -------- INPUT --------
x = input("Enter sequence x(n) separated by comma (example: 1,2,3,4): ")
x = np.array([float(i) for i in x.split(",")])

N = len(x)
n = np.arange(N)

print("\nInput sequence x(n):", x)

# -------- FORMULA METHOD --------
Y_formula = np.zeros(N, dtype=complex)

for k in range(N):
    for i in range(N):
        Y_formula[k] += x[i] * np.exp(-2j * np.pi * k * i / N)

mag_formula = np.abs(Y_formula)
phase_formula = np.angle(Y_formula)

print("\n--- Formula Method ---")
for i in range(N):
    print(f"y({i}) = {Y_formula[i]}")

# -------- KERNEL METHOD (FFT) --------
Y_fft = np.fft.fft(x)

mag_fft = np.abs(Y_fft)
phase_fft = np.angle(Y_fft)

print("\n--- Kernel / FFT Method ---")
for i in range(N):
    print(f"y({i}) = {Y_fft[i]}")

# -------- PLOTS --------
plt.figure(figsize=(12,10))

# Input sequence
plt.subplot(3,2,1)
plt.stem(n, x)
plt.title("Input Sequence x(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Formula Magnitude
plt.subplot(3,2,2)
plt.stem(n, mag_formula)
plt.title("Magnitude Spectrum (Formula)")

# Formula Phase
plt.subplot(3,2,3)
plt.stem(n, phase_formula)
plt.title("Phase Spectrum (Formula)")

# FFT Magnitude
plt.subplot(3,2,4)
plt.stem(n, mag_fft)
plt.title("Magnitude Spectrum (FFT)")

# FFT Phase
plt.subplot(3,2,5)
plt.stem(n, phase_fft)
plt.title("Phase Spectrum (FFT)")

# Output sequence
plt.subplot(3,2,6)
plt.stem(n, np.real(Y_fft))
plt.title("Output Sequence y(n)")

plt.tight_layout()
plt.show()
