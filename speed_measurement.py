import cv2
import numpy as np
import time

# Constants (change based on your setup)
PIXELS_PER_CM = 10  # Calibrate this based on your video (e.g., ruler in frame)

# Start the camera
cap = cv2.VideoCapture(0)  # Use 0 for default Pi cam

# Allow time for camera to warm up
time.sleep(2)

# Background subtractor to detect motion
fgbg = cv2.createBackgroundSubtractorMOG2()

prev_time = time.time()
prev_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (optional)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract background
    fgmask = fgbg.apply(gray)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None and cv2.contourArea(largest_contour) > 500:
        x, y, w, h = cv2.boundingRect(largest_contour)
        center = (x + w//2, y + h//2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        curr_time = time.time()
        dt = curr_time - prev_time

        if prev_pos is not None and dt > 0:
            dx = center[0] - prev_pos[0]
            dy = center[1] - prev_pos[1]
            distance_px = (dx**2 + dy**2) ** 0.5
            distance_cm = distance_px / PIXELS_PER_CM
            speed = distance_cm / dt  # cm/s

            cv2.putText(frame, f"Speed: {speed:.2f} cm/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print(f"Speed: {speed:.2f} cm/s")

        prev_time = curr_time
        prev_pos = center

    cv2.imshow("Speed Measurement", frame)
    cv2.imshow("Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
