import cv2
import time

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start camera
cap = cv2.VideoCapture(0)
time.sleep(2)

# Blink tracking
blink_count = 0
blink_start = None
blink_speeds = []
eye_detected_prev = True  # Start by assuming eye is open

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Check if eye was previously detected and now gone (blink)
    if len(eyes) == 0 and eye_detected_prev:
        blink_start = time.time()
        eye_detected_prev = False

    # Check if eye reappeared (blink ended)
    elif len(eyes) > 0 and not eye_detected_prev:
        blink_end = time.time()
        blink_duration = blink_end - blink_start
        blink_speed = 1 / blink_duration if blink_duration > 0 else 0
        blink_speeds.append(blink_speed)
        blink_count += 1
        eye_detected_prev = True

    # Display info
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    if blink_speeds:
        avg_blink_speed = sum(blink_speeds[-5:]) / min(len(blink_speeds), 5)  # Rolling average
        cv2.putText(frame, f"Blink Speed: {avg_blink_speed:.2f} Hz", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Eye Blink Speed Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
