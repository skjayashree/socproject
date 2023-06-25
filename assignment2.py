import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('points2D_Set1.mat')

# Extract data from the file
X = np.array(file['x'])
Y = np.array(file['y'])

X = X.ravel()
Y = Y.ravel()

# Scatter plot of the points
plt.scatter(X, Y)

# Fit a line to the data (linear regression)
coefficients = np.polyfit(X, Y, 1)
line = np.polyval(coefficients, X)

# Overlay the line on the scatter plot
plt.plot(X, line, color='red')

# Display the plot
plt.show()
