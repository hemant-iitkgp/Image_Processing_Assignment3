import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale
image_path = 'restore.jpg'
image = Image.open(image_path).convert('L')

# Convert the image to a NumPy array
image_np = np.array(image)

# Define a horizontal motion blur kernel
def motion_blur_kernel(size):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)  # Create horizontal line in the center
    return kernel / size  # Normalize the kernel

# Apply the kernel using convolution
def convolve(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad the image to handle edges
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Prepare the output image
    output_image = np.zeros_like(image)
    
    # Convolution operation
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + k_h, j:j + k_w]
            output_image[i, j] = np.sum(region * kernel)
    
    return np.clip(output_image, 0, 255).astype(np.uint8)

# Create a horizontal motion blur kernel (size = 15)
kernel = motion_blur_kernel(15)

# Apply the motion blur
motion_blurred_image = convolve(image_np, kernel)

# Display the original and blurred images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Motion Blurred Image
plt.subplot(1, 2, 2)
plt.imshow(motion_blurred_image, cmap='gray')
plt.title('Motion Blurred Image (Horizontal)')
plt.axis('off')

plt.show()