import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load main image
main_img = cv2.imread('image.png')
main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

# --- Dynamic Template Slicing ---
# Get dimensions of the main image
height, width = main_gray.shape

# Define template size (e.g., 100x100 pixels)
template_h, template_w = 100, 100

# Choose starting coordinates near the center of the image
start_y = height // 2
start_x = width // 2

# Slice the region out of the grayscale main image
template_gray = main_gray[start_y:start_y+template_h, start_x:start_x+template_w]
# Slicing the color version just for visualization purposes later
template_color = main_img[start_y:start_y+template_h, start_x:start_x+template_w]
# ---------------------------------

# Get width and height of the template for the bounding box
w, h = template_gray.shape[::-1]

# Perform template matching
res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Top-left corner of the detected match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw green bounding rectangle on a copy of the original image
output_img = main_img.copy()
cv2.rectangle(output_img, top_left, bottom_right, (0, 255, 0), 5)

# Convert BGR to RGB for correct matplotlib color visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB)), plt.title('Cropped Template')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), plt.title('Detected Object')
plt.tight_layout()
plt.show()