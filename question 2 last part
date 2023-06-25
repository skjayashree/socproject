import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Assuming you have computed the covariance matrix and obtained eigenvalues and eigenvectors
# stored in `eigenvalues_dict` and `eigenvectors_dict`, where keys represent digits (0-9).

# Sort and plot eigenvalues for each digit
for digit in range(10):
    eigenvalues = eigenvalues_dict[digit]
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

    # Plot eigenvalues
    plt.plot(sorted_eigenvalues, label=f"Digit {digit}")

plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.legend()
plt.title('Eigenvalues of Covariance Matrix for Each Digit')
plt.show()

# Display principal mode of variation for digit 1
digit = 1

# Assuming you have computed the mean image `mean_image` and the first eigenvector `v1`
mean_image_path = "path_to_mean_image.png"  # Replace with the actual path to the mean image
mean_image = Image.open(mean_image_path)

sqrt_lambda_v1 = np.sqrt(eigenvalues_dict[digit][0]) * eigenvectors_dict[digit][:, 0]
negative_variation = mean_image - sqrt_lambda_v1.reshape(mean_image.size)
positive_variation = mean_image + sqrt_lambda_v1.reshape(mean_image.size)

# Display the images
fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(negative_variation, cmap='gray')
axs[0].set_title('μ - √λ1v1')
axs[0].axis('off')
axs[1].imshow(mean_image, cmap='gray')
axs[1].set_title('μ')
axs[1].axis('off')
axs[2].imshow(positive_variation, cmap='gray')
axs[2].set_title('μ + √λ1v1')
axs[2].axis('off')
plt.suptitle(f"Principal Mode of Variation for Digit {digit}")
plt.show()
