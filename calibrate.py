from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera not found!")
    exit()

print("Calibration started.")
print("Keep the fish clearly visible for a few seconds.")
print("Press Q to quit.")

lengths = []

while True:
    ret, frame = camera.read()

    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        box = results[0].boxes.xyxy.cpu().numpy()[0]

        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        pixel_length = max(width, height)

        lengths.append(pixel_length)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Fish length: {pixel_length:.1f} pixels",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("FishTrack Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if lengths:
    pixel_length = np.median(lengths)

    print()
    print(f"Estimated fish length: {pixel_length:.1f} pixels")
    print("Assumed real fish length: 4.5 cm")

    pixels_per_cm = pixel_length / 4.5
    cm_per_pixel = 4.5 / pixel_length

    print(f"Pixels per cm: {pixels_per_cm:.2f}")
    print(f"CM per pixel: {cm_per_pixel:.4f}")
else:
    print("Fish was not detected.")