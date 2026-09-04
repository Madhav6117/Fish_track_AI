from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera not found!")
    exit()

print("Fish AI camera started.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("FishTrack - Custom Fish AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()