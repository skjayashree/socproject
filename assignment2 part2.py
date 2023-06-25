import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('mnist.mat', 'r')

labels = np.array(file['labels_train'])
images = np.array(file['digits_train'])

# Convert image data type to floating-point
images = images.astype(float)

# Function to compute mean
def compute_mean(digit):
    digit_images = images[labels.flatten() == digit]
    mean = np.mean(digit_images, axis=0)
    return mean

# Function to compute covariance matrix
def compute_covariance(digit):
    digit_images = images[labels.flatten() == digit]
    digit_images_2d = digit_images.reshape(digit_images.shape[0], -1)
    covariance = np.cov(digit_images_2d.T)
    return covariance

# Function to compute principal mode of variation
def compute_principal_mode(digit):
    covariance = compute_covariance(digit)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    principal_eigenvalue = eigenvalues[sorted_indices[0]]
    principal_eigenvector = eigenvectors[:, sorted_indices[0]]
    return principal_eigenvalue, principal_eigenvector

# Compute mean, covariance, and principal mode of variation for each digit
digits = range(10)
for digit in digits:
    mean = compute_mean(digit)
    covariance = compute_covariance(digit)
    principal_eigenvalue, principal_eigenvector = compute_principal_mode(digit)
    
    # Sort eigenvalues and plot as a graph
    sorted_eigenvalues = np.sort(np.real(np.linalg.eigvals(covariance)))[::-1]
    plt.plot(sorted_eigenvalues)
    plt.title(f"Eigenvalues for digit {digit}")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue Magnitude")
    plt.show()

    # Reshape the eigenvector
    reshaped_eigenvector = np.reshape(principal_eigenvector, (28, 28))

    # Show images
    fig, axs = plt.subplots(1, 3)
    images_to_show = [np.real(mean - np.sqrt(principal_eigenvalue) * reshaped_eigenvector),
                      np.real(mean),
                      np.real(mean + np.sqrt(principal_eigenvalue) * reshaped_eigenvector)]
    titles = ["μ - √λ1v1", "μ", "μ + √λ1v1"]
    for i, image in enumerate(images_to_show):
        axs[i].imshow(image.T, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(titles[i])
    plt.suptitle(f"Principal Mode of Variation for digit {digit}")
    plt.show()
