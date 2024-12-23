import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale mode
image_path = "restore.jpg"  # Use the correct image path
image = Image.open(image_path).convert('L')

# Convert the image to a NumPy array
image_np = np.array(image)

# Define the Gaussian noise parameters
mean = 0
variance = 100
sigma = np.sqrt(variance)

# Generate Gaussian noise
noise = np.random.normal(mean, sigma, image_np.shape)

# Add the noise to the image
noisy_image = image_np + noise

# Clip the values to stay within [0, 255] and convert back to uint8 type
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Display the original and noisy image side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Noisy Image
plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Gaussian Noise (Variance = 100)')
plt.axis('off')

plt.show()