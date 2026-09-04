import cv2
import os

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera not found!")
    exit()

os.makedirs("data/images", exist_ok=True)

count = 0

print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    cv2.imshow("Capture Fish Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = f"data/images/fish_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()