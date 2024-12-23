import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale mode
image_path = "restore.jpg"  # Use the correct image path
image = Image.open(image_path).convert('L')

# Convert the image to a NumPy array
image_np = np.array(image)

# Define a simple 3x3 low-pass filter (averaging filter)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0  # Normalize the kernel

# Get the image dimensions
height, width = image_np.shape

# Initialize the output array with zeros (same size as input image)
blurred_image = np.zeros_like(image_np)

# Perform the convolution operation
for i in range(1, height - 1):
    for j in range(1, width - 1):
        # Extract the 3x3 region of the image
        region = image_np[i-1:i+2, j-1:j+2]
        # Apply the kernel to the region (element-wise multiplication and sum)
        blurred_image[i, j] = np.sum(region * kernel)

# Display the original and blurred image side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Blurred Image
plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image with Low-Pass Filter')
plt.axis('off')

plt.show()