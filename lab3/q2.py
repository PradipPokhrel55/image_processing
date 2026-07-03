import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def intensity_transformations(A, B, highlight_value=255):
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError("Could not find or open 'image.png'")
    
    # 1. Digital Negative
    negative = 255 - img
    
    # 2. Intensity-level slicing WITH background preservation
    slicing_with_bg = img.copy()
    slicing_with_bg[(img >= A) & (img <= B)] = highlight_value
    
    # 3. Intensity-level slicing WITHOUT background preservation (Background becomes 0)
    slicing_without_bg = np.zeros_like(img)
    slicing_without_bg[(img >= A) & (img <= B)] = highlight_value
    
    return img, negative, slicing_with_bg, slicing_without_bg


# --- 1. OPTIONAL: CREATE A SAMPLE IMAGE IF IT DOES NOT EXIST ---
if not os.path.exists("image.png"):
    print("'image.png' not found. Generating a sample gradient image...")
    sample_img = np.linspace(0, 255, 300*300, dtype=np.uint8).reshape(300, 300)
    cv2.imwrite("image.png", sample_img)


lower_threshold = 100
upper_threshold = 180

original, neg, slice_wb, slice_wob = intensity_transformations(
    A=lower_threshold, 
    B=upper_threshold, 
    highlight_value=255
)


# --- 3. PLOT ALL IMAGES SIDE-BY-SIDE ---
titles = [
    'Original Image', 
    'Digital Negative', 
    'Slicing (With BG)', 
    'Slicing (Without BG)'
]
images = [original, neg, slice_wb, slice_wob]

plt.figure(figsize=(15, 5))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=11)
    plt.axis('off')

plt.tight_layout()
plt.show()