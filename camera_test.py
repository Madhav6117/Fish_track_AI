import cv2

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera not found!")
    exit()

print("Camera connected!")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    cv2.imshow("FishTrack Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()