import cv2

# Load a sample image
img = cv2.imread("/Users/pradippokhrel/Desktop/Image_processing/elg21-bird-8788491_1920.jpg")
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Extract and modify a specific pixel value
# Let's target pixel at row=100, col=100
original_pixel = img[100, 100]
print(f"Original pixel at (100,100) [B, G, R]: {original_pixel}")

img[100, 100] = [0, 0, 255] # Change specific pixel to solid red
print(f"Modified pixel at (100,100) [B, G, R]: {img[100, 100]}")

# 2. Drawing a bounding box
# Let's draw a green bounding box around a Region of Interest (ROI)
# Box coordinates: top-left (x1, y1), bottom-right (x2, y2)
x1, y1 = 50, 50
x2, y2 = 200, 200
color = (0, 255, 0) # Green in BGR
thickness = 2
cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

# 3. Copying and pasting a Region of Interest (ROI) to another location
# We grab a patch from the image (make sure it stays within image boundaries)
roi = img[70:150, 70:150] # Extract an 80x80 patch inside our bounding box

# Paste it somewhere else (e.g., starting at row=250, col=50)
# The destination slice size MUST perfectly match the ROI size (80x80)
img[250:330, 50:130] = roi

# Save and show the result
cv2.imwrite("pixel_operations_output.png", img)
cv2.imshow("Pixel Operations", img)
cv2.waitKey(0)
cv2.destroyAllWindows()