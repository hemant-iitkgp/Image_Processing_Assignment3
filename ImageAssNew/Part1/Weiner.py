import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale
image_path = "restore.jpg"
image = Image.open(image_path).convert('L')

# Convert the image to a NumPy array
image_np = np.array(image)

# Add Gaussian noise (same as before)
mean = 0
variance = 100
sigma = np.sqrt(variance)
noise = np.random.normal(mean, sigma, image_np.shape)
noisy_image = image_np + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to apply Wiener filtering
def wiener_filter(image, kernel_size=5, noise_var=100):
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='reflect')
    filtered_image = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_patch = padded_image[i:i + kernel_size, j:j + kernel_size]
            local_mean = np.mean(local_patch)
            local_var = np.var(local_patch)

            # Wiener filter formula
            if local_var > noise_var:
                filtered_image[i, j] = local_mean + (local_var - noise_var) / local_var * (image[i, j] - local_mean)
            else:
                filtered_image[i, j] = local_mean
    
    return np.clip(filtered_image, 0, 255).astype(np.uint8)

# Apply Wiener filtering
restored_image = wiener_filter(noisy_image)

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

# Calculate MSE and PSNR
mse_value = mse(image_np, restored_image)
psnr_value = psnr(image_np, restored_image)

# Print the results
print(f"MSE: {mse_value}")
print(f"PSNR: {psnr_value} dB")

# Display the noisy and restored images
plt.figure(figsize=(10, 5))

# Noisy Image
plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# Restored Image (after Wiener filtering)
plt.subplot(1, 2, 2)
plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image (Wiener Filtering)')
plt.axis('off')

plt.show()