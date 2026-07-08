import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny with 3 different threshold configurations
canny_low = cv2.Canny(img, threshold1=30, threshold2=100)      # Captures more noise/weak edges
canny_medium = cv2.Canny(img, threshold1=80, threshold2=150)    # Balanced choice
canny_high = cv2.Canny(img, threshold1=150, threshold2=250)    # Captures only strong edges

# Plotting Results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original Gray')
plt.subplot(1, 4, 2), plt.imshow(canny_low, cmap='gray'), plt.title('Low Threshold (30, 100)')
plt.subplot(1, 4, 3), plt.imshow(canny_medium, cmap='gray'), plt.title('Mid Threshold (80, 150)')
plt.subplot(1, 4, 4), plt.imshow(canny_high, cmap='gray'), plt.title('High Threshold (150, 250)')
plt.tight_layout()
plt.show()