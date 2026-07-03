import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_specification(img_A_path, img_B_path):
    # Read both images in grayscale
    A = cv2.imread(img_A_path, cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(img_B_path, cv2.IMREAD_GRAYSCALE)
    
    if A is None or B is None:
        raise FileNotFoundError("Could not find one or both input images.")
    
    # Compute normalized histograms & CDFs for both images
    hist_A, _ = np.histogram(A.flatten(), 256, [0,256])
    hist_B, _ = np.histogram(B.flatten(), 256, [0,256])
    
    cdf_A = hist_A.cumsum() / hist_A.sum()
    cdf_B = hist_B.cumsum() / hist_B.sum()
    
    # Create matching look-up table (LUT)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the index in cdf_B closest to cdf_A[i]
        idx = np.argmin(np.abs(cdf_A[i] - cdf_B))
        lut[i] = idx
        
    # Apply LUT transformation map to Image A
    matched_A = lut[A]
    
    # Compute histogram of the matched output image for plotting
    hist_matched, _ = np.histogram(matched_A.flatten(), 256, [0,256])
    
    return A, hist_A, B, hist_B, matched_A, hist_matched


# --- 1. SAMPLE GENERATION GUARD (IF FILES ARE MISSING) ---
if not os.path.exists('image_A.png') or not os.path.exists('image_B.png'):
    print("Generating sample images for matching...")
    # Image A: A bright image with a dark circle
    img_a = np.ones((300, 300), dtype=np.uint8) * 200
    cv2.circle(img_a, (150, 150), 70, 50, -1)
    cv2.imwrite('image_A.png', img_a)
    
    # Image B: A very dark/moody image with a medium-gray circle (The Target Style)
    img_b = np.ones((300, 300), dtype=np.uint8) * 40
    cv2.circle(img_b, (150, 150), 70, 120, -1)
    cv2.imwrite('image_B.png', img_b)


# --- 2. RUN THE ALGORITHM ---
A, hist_A, B, hist_B, matched_A, hist_matched = histogram_specification('image.png', 'image1.png')


# --- 3. PLOT IMAGES AND HISTOGRAMS TOGETHER ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Row 1: The Images
axes[0, 0].imshow(A, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Source Image A (Original)')
axes[0, 0].axis('off')

axes[0, 1].imshow(B, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Reference Image B (Target)')
axes[0, 1].axis('off')

axes[0, 2].imshow(matched_A, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Matched Image A (Result)')
axes[0, 2].axis('off')

# Row 2: The Histograms
axes[1, 0].bar(range(256), hist_A, width=1.0, color='blue', alpha=0.7)
axes[1, 0].set_title('Histogram of A')
axes[1, 0].set_xlim([0, 256])

axes[1, 1].bar(range(256), hist_B, width=1.0, color='orange', alpha=0.7)
axes[1, 1].set_title('Histogram of B (Target Shape)')
axes[1, 1].set_xlim([0, 256])

axes[1, 2].bar(range(256), hist_matched, width=1.0, color='green', alpha=0.7)
axes[1, 2].set_title('Histogram of Matched Result')
axes[1, 2].set_xlim([0, 256])

plt.tight_layout()
plt.show()