import cv2
import datetime

# Initialize the webcam (0 is typically the default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Fetch baseline properties for the video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# If FPS returns 0 or invalid from webcam metadata, default it to standard 30
if fps == 0 or fps > 100:
    fps = 30.0

# Initialize VideoWriter to automatically record a 5-second snippet
# 'XVID' or 'MJPG' are widely compatible formats
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('recorded_5s_clip.avi', fourcc, fps, (frame_width, frame_height))

frame_number = 0
screenshot_count = 0

print("Recording started automatically. Press 'p' for screenshot, 'q' to exit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_number += 1

    # 1. Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Reconvert back to 3-channels so we can write colored text/metadata on it 
    # and save it seamlessly into the standard VideoWriter pipeline.
    display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # 2. Fetch current date and time
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    overlay_text = f"Frame: {frame_number} | {timestamp_str}"

    # Put text overlay onto our display frame
    cv2.putText(display_frame, overlay_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 3. Automatically write to video file up until 5 seconds worth of frames have elapsed
    if frame_number <= int(fps * 5):
        video_writer.write(display_frame)
        if frame_number == int(fps * 5):
            print("--- 5 Seconds of recording complete! Clip saved. ---")

    # Display the final frame live
    cv2.imshow("Webcam Stream", display_frame)

    # Handle keystrokes (wait for 1ms per loop iteration)
    key = cv2.waitKey(1) & 0xFF
    
    # 4. Capture screenshot when 'p' is pressed
    if key == ord('p'):
        screenshot_count += 1
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(filename, display_frame)
        print(f"Screenshot saved: {filename}")

    # 5. Exit cleanly when 'q' is pressed
    elif key == ord('q'):
        break

# Release structural assets completely from CPU memory
cap.release()
video_writer.release()
cv2.destroyAllWindows()