import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

img_float = img.astype(np.float64)

# Calculate scaling constant 'c' safely
max_val = np.max(img_float)
c_log = 255.0 / np.log(1.0 + max_val) if max_val > 0 else 1.0

# Perform transformation
log_transformed = c_log * np.log(1.0 + img_float)

# Crucial Step: Round and clip values to ensure they fit perfectly into 8-bit bounds
log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

# 2. Power Law (Gamma) Transformation
def gamma_transform(image, gamma):
    # Normalize to [0, 1]
    normalized_img = image / 255.0
    # Apply gamma
    gamma_corrected = np.power(normalized_img, gamma)
    # Scale back to [0, 255]
    return np.array(gamma_corrected * 255, dtype=np.uint8)

gamma_0_5 = gamma_transform(img, 0.5)  # Brightens the image
gamma_1_5 = gamma_transform(img, 1.5)  # Darkens the image

# Plotting Results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(log_transformed, cmap='gray'), plt.title('Log Transform')
plt.subplot(2, 2, 3), plt.imshow(gamma_0_5, cmap='gray'), plt.title('Gamma = 0.5 (Bright)')
plt.subplot(2, 2, 4), plt.imshow(gamma_1_5, cmap='gray'), plt.title('Gamma = 1.5 (Dark)')
plt.tight_layout()
plt.show()