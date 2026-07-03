import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def image_magnification(image_path, scale_factor=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Could not find or open '{image_path}'")
        
    h, w = img.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Method 1: Pixel Replication (Nearest Neighbor)
    replication_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Method 2: Bilinear Interpolation
    interpolation_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Return original image along with both magnified versions
    return img, replication_out, interpolation_out


# --- 1. SAMPLE GENERATION GUARD (IF FILE MISSING) ---
if not os.path.exists('image.png'):
    print("'image.png' not found. Generating a small 50x50 sharp geometric image...")
    # Creating a very small image so magnification artifacts are obvious
    sample_img = np.zeros((50, 50), dtype=np.uint8)
    # Draw a white square in the middle
    cv2.rectangle(sample_img, (15, 15), (35, 35), 255, -1)
    # Draw a diagonal line passing through
    cv2.line(sample_img, (0, 0), (50, 50), 180, 2)
    cv2.imwrite('image.png', sample_img)


# --- 2. RUN THE MAGNIFICATION ALGORITHM ---
# Magnify the small image by 6 times its original size
scale_factor = 6
original, replication_result, interpolation_result = image_magnification('image.png', scale_factor=scale_factor)


# --- 3. PLOT AND COMPARE SIDE-BY-SIDE ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Left Plot: Original tiny image
axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f'Original Tiny Image\n({original.shape[1]}x{original.shape[0]})')
axes[0].axis('off')

# Middle Plot: Pixel Replication
axes[1].imshow(replication_result, cmap='gray', vmin=0, vmax=255)
axes[1].set_title(f'Pixel Replication (x{scale_factor})\n[Notice Jagged Blocky Edges]')
axes[1].axis('off')

# Right Plot: Bilinear Interpolation
axes[2].imshow(interpolation_result, cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f'Bilinear Interpolation (x{scale_factor})\n[Notice Blurred Soft Edges]')
axes[2].axis('off')

plt.tight_layout()
plt.show()