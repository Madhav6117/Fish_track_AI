from ultralytics import YOLO
import cv2
import time
from distance import calculate_distance

# ==============================
# Load Model & Camera
# ==============================

model = YOLO("runs/detect/train/weights/best.pt")
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Camera not found!")
    exit()

# ==============================
# Calibration & Fish Information
# ==============================

FISH_LENGTH_CM = 4.5
FISH_WEIGHT_G = 0.7

PIXELS_PER_CM = 39.77
CM_PER_PIXEL = 1 / PIXELS_PER_CM

# Energy estimation assumption
ENERGY_COST_KCAL_PER_G_M = 0.001

# ==============================
# Tracking Variables
# ==============================

previous_point = None
smoothed_point = None

total_distance_pixels = 0
total_distance_cm = 0
total_distance_m = 0

estimated_calories_observed = 0

# ==============================
# Smoothing Settings
# ==============================

ALPHA = 0.4
MOVEMENT_THRESHOLD = 2

start_time = time.time()

# ==============================
# Startup Information
# ==============================

print("====================================")
print(" FishTrack Distance Tracking Started")
print("====================================")

print(f"Assumed fish length : {FISH_LENGTH_CM} cm")
print(f"Assumed fish weight : {FISH_WEIGHT_G} g")
print(f"Pixels per cm       : {PIXELS_PER_CM}")
print(f"CM per pixel        : {CM_PER_PIXEL:.4f}")

print()
print("Energy estimation:")
print(f"Energy cost         : {ENERGY_COST_KCAL_PER_G_M} kcal/(g*m)")

print()
print("Press Q to quit.")
print()

# ==============================
# Main Tracking Loop
# ==============================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    elapsed_time = time.time() - start_time

    # ==============================
    # YOLO + ByteTrack
    # ==============================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    # ==============================
    # Check Detection
    # ==============================

    if (
        results[0].boxes is not None
        and len(results[0].boxes) > 0
        and results[0].boxes.id is not None
    ):

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        # Since there is only one fish
        box = boxes[0]
        track_id = track_ids[0]

        x1, y1, x2, y2 = box

        # Calculate center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        current_point = (center_x, center_y)

        # ==============================
        # Smooth Fish Position
        # ==============================

        if smoothed_point is None:

            smoothed_point = current_point

        else:

            smooth_x = int(
                ALPHA * current_point[0]
                + (1 - ALPHA) * smoothed_point[0]
            )

            smooth_y = int(
                ALPHA * current_point[1]
                + (1 - ALPHA) * smoothed_point[1]
            )

            smoothed_point = (smooth_x, smooth_y)

        # ==============================
        # Calculate Movement
        # ==============================

        if previous_point is not None:

            movement = calculate_distance(
                previous_point,
                smoothed_point
            )

            # Ignore tiny movements caused by detection noise
            if movement >= MOVEMENT_THRESHOLD:

                total_distance_pixels += movement

                movement_cm = movement * CM_PER_PIXEL

                total_distance_cm += movement_cm

                total_distance_m = total_distance_cm / 100

        previous_point = smoothed_point

        # ==============================
        # Calculate Average Speed
        # ==============================

        if elapsed_time > 0:

            speed_cm_per_sec = (
                total_distance_cm / elapsed_time
            )

        else:

            speed_cm_per_sec = 0

        # ==============================
        # Calculate Observed Calories
        # ==============================

        estimated_calories_observed = (
            total_distance_m
            * FISH_WEIGHT_G
            * ENERGY_COST_KCAL_PER_G_M
        )

        # ==============================
        # Draw Tracking Information
        # ==============================

        cv2.circle(
            frame,
            smoothed_point,
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"ID: {int(track_id)}",
            (center_x, center_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

    else:

        speed_cm_per_sec = 0

    # ==============================
    # Display Statistics
    # ==============================

    cv2.putText(
        frame,
        f"Distance: {total_distance_cm:.2f} cm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Distance: {total_distance_m:.3f} m",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Time: {elapsed_time:.1f} sec",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Speed: {speed_cm_per_sec:.2f} cm/s",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Fish reference: 4.5 cm",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Calories: {estimated_calories_observed:.4f} kcal",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    # ==============================
    # Show Camera
    # ==============================

    cv2.imshow(
        "FishTrack - Distance Tracking",
        frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# Cleanup
# ==============================

camera.release()
cv2.destroyAllWindows()


# ==============================
# Final Calculations
# ==============================

if elapsed_time > 0:

    average_speed = (
        total_distance_cm / elapsed_time
    )

    estimated_calories_observed = (
        total_distance_m
        * FISH_WEIGHT_G
        * ENERGY_COST_KCAL_PER_G_M
    )

else:

    average_speed = 0
    estimated_calories_observed = 0


# ==============================
# Final Result
# ==============================

print()
print("====================================")
print("           FINAL RESULT")
print("====================================")

print(f"Observation time   : {elapsed_time:.1f} seconds")
print(f"Distance            : {total_distance_cm:.2f} cm")
print(f"Distance            : {total_distance_m:.3f} m")
print(f"Average speed       : {average_speed:.2f} cm/s")
print(f"Fish weight         : {FISH_WEIGHT_G:.1f} g")
print(f"Calories observed   : {estimated_calories_observed:.4f} kcal")

print("====================================")