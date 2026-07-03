import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equalization_scratch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not find or open '{image_path}'")
        
    h, w = img.shape
    num_pixels = h * w
    
    # 1. Compute histogram manually
    hist = np.zeros(256, dtype=int)
    for pixel in img.ravel():
        hist[pixel] += 1
        
    # 2. Compute Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()
    
    # 3. Normalize CDF to scale into [0, 255] range
    cdf_normalized = ((cdf - cdf.min()) * 255) / (num_pixels - cdf.min())
    cdf_normalized = np.uint8(cdf_normalized)
    
    # 4. Map original pixels to equalized values
    equalized_img = cdf_normalized[img]
    
    # Calculate modified histogram
    equalized_hist = np.zeros(256, dtype=int)
    for pixel in equalized_img.ravel():
        equalized_hist[pixel] += 1
        
    return img, hist, equalized_img, equalized_hist


# --- 1. SAMPLE GENERATION GUARD (IF FILE MISSING) ---
if not os.path.exists('image.png'):
    print("'image.png' not found. Generating a sample low-contrast image...")
    # Generate a low-contrast image packed tight around the mid-tones (brightness 90 to 140)
    np.random.seed(0)
    base = np.random.normal(115, 8, (300, 300)).astype(np.uint8)
    cv2.putText(base, "LOW CONTRAST", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 140, 3)
    cv2.imwrite('image.png', base)


# --- 2. RUN THE ALGORITHM ---
# Capture all 4 items returned by your function
orig_img, orig_hist, eq_img, eq_hist = histogram_equalization_scratch('image.png')


# --- 3. PLOT IMAGES AND HISTOGRAMS TOGETHER ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-Left: Original Image
axes[0, 0].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original Low-Contrast Image')
axes[0, 0].axis('off')

# Top-Right: Equalized Image
axes[0, 1].imshow(eq_img, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Equalized High-Contrast Image')
axes[0, 1].axis('off')

# Bottom-Left: Original Histogram
axes[1, 0].bar(range(256), orig_hist, width=1.0, color='gray')
axes[1, 0].set_title('Original Histogram Distribution')
axes[1, 0].set_xlim([0, 256])
axes[1, 0].set_xlabel('Intensity Level')
axes[1, 0].set_ylabel('Pixel Count')

# Bottom-Right: Equalized Histogram
axes[1, 1].bar(range(256), eq_hist, width=1.0, color='black')
axes[1, 1].set_title('Equalized (Flattened) Histogram')
axes[1, 1].set_xlim([0, 256])
axes[1, 1].set_xlabel('Intensity Level')
axes[1, 1].set_ylabel('Pixel Count')

plt.tight_layout()
plt.show()