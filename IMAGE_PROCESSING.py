import numpy as np
import math

# -----------------------------
# MATRIX INPUT
# -----------------------------
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

print("Enter matrix values:")

matrix = []
for i in range(rows):
    while True:
        row = list(map(int, input(f"Row {i+1}: ").split()))
        if len(row) != cols:
            print("Enter exactly", cols, "values")
        else:
            matrix.append(row)
            break

img = np.array(matrix)

print("\nOriginal Matrix:")
print(img)

# -----------------------------
# CALCULATE L VALUE
# -----------------------------
max_val = np.max(img)

L = 1
bits = 0
while L <= max_val:
    L = 2 ** bits
    bits += 1

bits -= 1
L = 2 ** bits

print("\nMaximum value in matrix:", max_val)
print("Closest power of 2 (L):", L)
print("Number of bits:", bits)

# -----------------------------
# LOG TRANSFORMATION
# -----------------------------
c = (L - 1) / np.log(1 + max_val)

log_transformed = c * np.log(1 + img)

# custom rounding rule
def custom_round(x):
    frac = x - int(x)
    if frac < 0.5:
        return int(x)
    else:
        return int(x) + 1

log_transformed = np.vectorize(custom_round)(log_transformed)

print("\nLog Transformation Output:")
print(log_transformed)

# -----------------------------
# CONTRAST STRETCHING
# -----------------------------
def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        val = (s1 / r1) * pix if r1 != 0 else 0
    elif r1 < pix <= r2:
        val = ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        val = ((L - 1 - s2) / (L - 1 - r2)) * (pix - r2) + s2

    frac = val - int(val)
    if frac < 0.5:
        return int(val)
    else:
        return int(val) + 1


r1 = int(input("\nEnter r1: "))
s1 = int(input("Enter s1: "))
r2 = int(input("Enter r2: "))
s2 = int(input("Enter s2: "))

pixelVal_vec = np.vectorize(pixelVal)

contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

print("\nContrast Stretching Output:")
print(contrast_stretched)

# -----------------------------
# BINARY REPRESENTATION
# -----------------------------
print("\nBinary Representation of Matrix:")

binary_matrix = np.vectorize(lambda x: format(x, f'0{bits}b'))(img)
print(binary_matrix)

# -----------------------------
# BIT PLANE SLICING
# -----------------------------
print("\nBit Plane Slicing Outputs:")

for bit in range(bits):
    bit_plane = (img >> bit) & 1
    print(f"\nBit Plane {bit}:")
    print(bit_plane)
