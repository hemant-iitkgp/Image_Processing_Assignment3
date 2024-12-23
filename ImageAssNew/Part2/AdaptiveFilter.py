import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the original image (before motion blur)
image_path = 'restore.jpg'
original_image = Image.open(image_path).convert('L')
original_image_np = np.array(original_image)

# Apply horizontal motion blur (as done earlier)
def motion_blur_kernel(size):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)  # Create horizontal line in the center
    return kernel / size  # Normalize the kernel

def convolve(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output_image = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + k_h, j:j + k_w]
            output_image[i, j] = np.sum(region * kernel)
    
    return np.clip(output_image, 0, 255).astype(np.uint8)

# Apply the motion blur with a kernel size of 15
kernel = motion_blur_kernel(15)
motion_blurred_image = convolve(original_image_np, kernel)

# Adaptive filtering using Median Filter
def median_filter(image, filter_size=3):
    padded_image = np.pad(image, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), mode='reflect')
    restored_image = np.zeros_like(image)
    
    h, w = image.shape
    
    for i in range(h):
        for j in range(w):
            # Extract the local region (neighborhood)
            region = padded_image[i:i + filter_size, j:j + filter_size]
            # Apply median filtering
            restored_image[i, j] = np.median(region)
    
    return restored_image

# Apply adaptive filtering
restored_image = median_filter(motion_blurred_image, filter_size=5)

# Function to calculate MSE
def mse(original, restored):
    return np.mean((original - restored) ** 2)

# Function to calculate PSNR
def psnr(original, restored):
    max_pixel_value = 255.0
    mse_value = mse(original, restored)
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse_value))

# Calculate MSE and PSNR between the original and restored image
mse_value = mse(original_image_np, restored_image)
psnr_value = psnr(original_image_np, restored_image)

# Print the results
print(f"MSE: {mse_value}")
print(f"PSNR: {psnr_value} dB")

# Display the images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Motion Blurred Image
plt.subplot(1, 3, 2)
plt.imshow(motion_blurred_image, cmap='gray')
plt.title('Motion Blurred Image')
plt.axis('off')

# Restored Image (after adaptive filtering)
plt.subplot(1, 3, 3)
plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image (Adaptive Filter)')
plt.axis('off')

plt.show()