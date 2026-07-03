import cv2
import os

def process_image(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Display properties
    height, width, channels = img.shape
    file_size_bytes = os.path.getsize(image_path)
    file_size_kb = file_size_bytes / 1024

    print("--- Image Properties ---")
    print(f"Width:        {width} px")
    print(f"Height:       {height} px")
    print(f"Channels:     {channels} (BGR)")
    print(f"File Size:    {file_size_kb:.2f} KB")
    print("------------------------")

    # 3. Convert and save in another format (e.g., PNG to JPG or vice versa)
    # This example saves whatever input format you used into a PNG
    cv2.imwrite("output_converted.png", img)

    # 4. Resize to 50% and 200%
    img_50 = cv2.resize(img, (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_AREA)
    img_200 = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite("resized_50.png", img_50)
    cv2.imwrite("resized_200.png", img_200)

    # 5. Rotate by 90° and 180° clockwise
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_180 = cv2.rotate(img, cv2.ROTATE_180)

    cv2.imwrite("rotated_90.png", img_90)
    cv2.imwrite("rotated_180.png", img_180)
    
    print("All transformations saved successfully!")

# Replace 'input_image.jpg' with your actual image filename
process_image("/Users/pradippokhrel/Desktop/Image_processing/elg21-bird-8788491_1920.jpg")