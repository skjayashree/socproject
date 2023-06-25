
import scipy.io as sio
import numpy as np
import h5py
from PIL import Image

# Load the dataset from mnist.mat file
dataset = h5py.File('mnist.mat')

# Get the training data
images = dataset['digits_train']
labels = dataset['labels_train']
# Access the actual data array and flatten it
images = np.array(images)  # Convert images dataset to a NumPy array
labels = np.array(labels)  # Convert labels dataset to a NumPy array
images = images.reshape(60000, 784).astype(float)
labels = labels.flatten()  # Flatten the labels array

# Access the actual data array and reshape it
images = np.array(images)  # Convert dataset to a NumPy array
images = images.reshape(60000, 784).astype(float)

# Compute the mean for each digit (0 to 9)
means = []
for digit in range(10):
    digit_images = images[labels.flatten() == digit]
    digit_mean = np.mean(digit_images, axis=0)
    means.append(digit_mean)

# Compute the covariance matrix for each digit (0 to 9)
covariance_matrices = []
for digit in range(10):
    digit_images = images[labels.flatten() == digit]
    digit_covariance = np.cov(digit_images, rowvar=False)
    covariance_matrices.append(digit_covariance)

# Compute the principal mode of variation for each digit (0 to 9)
eigenvalues = []
eigenvectors = []
for digit in range(10):
    digit_covariance = covariance_matrices[digit]
    eigenvalues_digit, eigenvectors_digit = np.linalg.eigh(digit_covariance)
    sorted_indices = np.argsort(eigenvalues_digit)[::-1]  # Sort eigenvalues in descending order
    sorted_eigenvalues = eigenvalues_digit[sorted_indices]
    sorted_eigenvectors = eigenvectors_digit[:, sorted_indices]
    eigenvalues.append(sorted_eigenvalues)
    eigenvectors.append(sorted_eigenvectors)

# Retrieve the principal mode of variation (eigenvector v1) and the corresponding eigenvalue (λ1)
principal_modes = [eigenvectors[digit][:, 0] for digit in range(10)]
eigenvalues_principal_modes = [eigenvalues[digit][0] for digit in range(10)]

# Print the results for each digit (0 to 9)
for digit in range(10):
    print(f"Digit: {digit}")
    print("Mean:", means[digit])
    print("Covariance Matrix:")
    print(covariance_matrices[digit])
    print("Principal Mode of Variation (v1):")
    print(principal_modes[digit])
    print("Corresponding Eigenvalue (λ1):", eigenvalues_principal_modes[digit])
    print()

# Load .mat file using h5py
file = h5py.File('mnist.mat', 'r')

labels = np.array(file['labels_train']).T
images = np.array(file['digits_train']).T

# Accessing the 50th label from the dataset
print("Label of the 50th image:", labels[50])

# Showing the 50th image from the dataset
image = Image.fromarray(images[:, :, 50])
image.show()
