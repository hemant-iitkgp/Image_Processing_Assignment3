import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale mode
image_path = "restore.jpg"  # Replace with your actual image path
image = Image.open(image_path).convert('L')  # Convert image to grayscale

# Convert the image to a NumPy array
image_np = np.array(image)

# Display the grayscale image
plt.imshow(image_np, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')  # Hide axes
plt.show()