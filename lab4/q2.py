import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
# Smooth slightly to reduce noise
img_smoothed = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float64)

# --- 1. Sobel Operator ---center weighted
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# --- 2. Prewitt Operator ---
kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_x = convolve(img_smoothed, kernel_prewitt_x)
prewitt_y = convolve(img_smoothed, kernel_prewitt_y)
prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)

# --- 3. Roberts Operator ---
kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
roberts_x = convolve(img_smoothed, kernel_roberts_x)
roberts_y = convolve(img_smoothed, kernel_roberts_y)
roberts_combined = np.sqrt(roberts_x**2 + roberts_y**2)

# Helper function to plot a row for each operator
def plot_edges(title, x_img, y_img, comb_img, row_idx):
    plt.subplot(3, 3, row_idx*3 + 1), plt.imshow(np.abs(x_img), cmap='gray'), plt.title(f'{title} Horizontal')
    plt.subplot(3, 3, row_idx*3 + 2), plt.imshow(np.abs(y_img), cmap='gray'), plt.title(f'{title} Vertical')
    plt.subplot(3, 3, row_idx*3 + 3), plt.imshow(comb_img, cmap='gray'), plt.title(f'{title} Combined')

plt.figure(figsize=(14, 12))
plot_edges('Sobel', sobel_x, sobel_y, sobel_combined, 0)
plot_edges('Prewitt', prewitt_x, prewitt_y, prewitt_combined, 1)
plot_edges('Roberts', roberts_x, roberts_y, roberts_combined, 2)
plt.tight_layout()
plt.show()