import os
import numpy as np
import matplotlib.pyplot as plt

# Define the path and filename
path = "/Users/abhishek/work/free_surface_2025/wang_kraus_scaled/elongated/data"
filename = "eflux.dat"

# Join the directory and filename to make a full path
filepath = os.path.join(path, filename)

# Load data (skip the first row, assuming space-separated columns)
data = np.loadtxt(filepath, skiprows=1)

# Extract columns
t = data[:, 0]  # 1st column
eflux = data[:, 1]  # 4th column

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t, eflux, marker='o', linestyle='-', label='Column 1 vs Column 4')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy Flux')
plt.title('Plot of Total FLux with TIme')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
