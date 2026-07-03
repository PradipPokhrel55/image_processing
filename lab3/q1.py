import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def contrast_stretching(image_path, num_bits):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was successfully loaded
    if img is None:
        raise FileNotFoundError(f"Could not find or open the image at '{image_path}'")
    
    # Get structural min and max intensity values
    r_min, r_max = float(np.min(img)), float(np.max(img))
    
    # Max possible intensity value based on user-defined bits
    L_max = (2 ** num_bits) - 1
    
    # Prevent division by zero if the image is completely flat
    if r_max == r_min:
        return img, img.copy()
    
    # Apply contrast stretching formula
    stretched = ((img - r_min) / (r_max - r_min)) * L_max
    stretched = np.uint8(np.clip(stretched, 0, L_max))
    
    return img, stretched

# --- 2. RUN USING YOUR REAL FILE ---
# This pulls directly from 'image.png' on your computer
original, result = contrast_stretching('image.png', num_bits=8)

# --- 3. PLOT AND DISPLAY IMMEDIATELY ---
plt.figure(figsize=(10, 5))

# Plot Original Image
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray', vmin=0, vmax=255)
plt.title(f'Original (Min: {original.min()}, Max: {original.max()})')
plt.axis('off')

# Plot Enhanced Result
plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray', vmin=0, vmax=255)
plt.title(f'Stretched (Min: {result.min()}, Max: {result.max()})')
plt.axis('off')

plt.tight_layout()
plt.show()