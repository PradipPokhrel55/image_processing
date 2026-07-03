import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def bit_plane_slicing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    planes = []
    
    # Extract bit planes from 0 to 7
    for i in range(8):
        # Perform bitwise AND between pixel and 2^i, then scale to 255 for visibility
        plane = np.bitwise_and(img, 1 << i)
        plane[plane > 0] = 255
        planes.append(plane)
        
    # Plotting the 8 planes
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(planes[i], cmap='gray')
        ax.set_title(f'Bit Plane {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


bit_plane_slicing('image.png')
# Visual Analysis Question Answer:
# The higher-order bit planes (Bit planes 6 and 7) carry the most significant visual information. 
# Bit plane 7 contains the Most Significant Bit (MSB), contributing to the structural and broad structural 
# intensity features, while lower-order planes (like Bit 0) mostly contain high-frequency noise.