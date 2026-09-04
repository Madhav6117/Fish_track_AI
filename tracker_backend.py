from ultralytics import YOLO
import cv2
import time
import threading

from distance import calculate_distance


class FishTracker:

    def __init__(self):

        # -----------------------------
        # MODEL
        # -----------------------------

        self.model = YOLO(
            "runs/detect/train/weights/best.pt"
        )

        # -----------------------------
        # CAMERA
        # -----------------------------

        self.camera = cv2.VideoCapture(1)

        if not self.camera.isOpened():
            raise RuntimeError("USB camera could not be opened.")

        # -----------------------------
        # CALIBRATION
        # -----------------------------

        self.fish_length_cm = 4.5

        self.pixels_per_cm = 39.77

        self.cm_per_pixel = (
            1 / self.pixels_per_cm
        )

        # -----------------------------
        # FISH INFORMATION
        # -----------------------------

        self.fish_weight_g = 0.7

        # Engineering estimate
        self.energy_cost_kcal_per_g_m = 0.001

        # -----------------------------
        # TRACKING VARIABLES
        # -----------------------------

        self.previous_point = None

        self.smoothed_point = None

        self.total_distance_pixels = 0

        self.total_distance_cm = 0

        self.total_distance_m = 0

        # -----------------------------
        # SMOOTHING
        # -----------------------------

        self.alpha = 0.4

        self.movement_threshold = 2

        # -----------------------------
        # TIME
        # -----------------------------

        self.start_time = None

        # -----------------------------
        # CURRENT DATA
        # -----------------------------

        self.track_id = 0

        self.confidence = 0

        self.speed_cm_s = 0

        self.calories = 0

        self.calories_per_minute = 0

        self.elapsed_seconds = 0

        # -----------------------------
        # VIDEO FRAME
        # -----------------------------

        self.latest_frame = None

        # -----------------------------
        # THREAD
        # -----------------------------

        self.running = False

        self.thread = None

        self.lock = threading.Lock()


    # ==========================================
    # START TRACKER
    # ==========================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.start_time = time.time()

        self.thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True
        )

        self.thread.start()


    # ==========================================
    # TRACKING LOOP
    # ==========================================

    def _tracking_loop(self):

        while self.running:

            ret, frame = self.camera.read()

            if not ret:
                continue

            # -----------------------------
            # TIME
            # -----------------------------

            self.elapsed_seconds = (
                time.time() - self.start_time
            )

            # -----------------------------
            # YOLO + BYTETRACK
            # -----------------------------

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.5,
                verbose=False
            )

            result = results[0]

            # -----------------------------
            # DETECTION EXISTS
            # -----------------------------

            if (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.boxes.id is not None
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                # Use first detected fish
                box = boxes[0]

                track_id = track_ids[0]

                confidence = confidences[0]

                # -----------------------------
                # BOUNDING BOX
                # -----------------------------

                x1, y1, x2, y2 = box

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                current_point = (
                    center_x,
                    center_y
                )

                # -----------------------------
                # SMOOTHING
                # -----------------------------

                if self.smoothed_point is None:

                    self.smoothed_point = (
                        current_point
                    )

                else:

                    smooth_x = int(
                        self.alpha * current_point[0]
                        +
                        (1 - self.alpha)
                        * self.smoothed_point[0]
                    )

                    smooth_y = int(
                        self.alpha * current_point[1]
                        +
                        (1 - self.alpha)
                        * self.smoothed_point[1]
                    )

                    self.smoothed_point = (
                        smooth_x,
                        smooth_y
                    )

                # -----------------------------
                # DISTANCE
                # -----------------------------

                if self.previous_point is not None:

                    movement = calculate_distance(
                        self.previous_point,
                        self.smoothed_point
                    )

                    if movement >= self.movement_threshold:

                        self.total_distance_pixels += movement

                        movement_cm = (
                            movement
                            * self.cm_per_pixel
                        )

                        self.total_distance_cm += (
                            movement_cm
                        )

                        self.total_distance_m = (
                            self.total_distance_cm
                            / 100
                        )

                self.previous_point = (
                    self.smoothed_point
                )

                # -----------------------------
                # SPEED
                # -----------------------------

                if self.elapsed_seconds > 0:

                    self.speed_cm_s = (
                        self.total_distance_cm
                        /
                        self.elapsed_seconds
                    )

                # -----------------------------
                # CALORIES
                # -----------------------------

                self.calories = (
                    self.total_distance_m
                    *
                    self.fish_weight_g
                    *
                    self.energy_cost_kcal_per_g_m
                )

                elapsed_minutes = (
                    self.elapsed_seconds / 60
                )

                if elapsed_minutes > 0:

                    self.calories_per_minute = (
                        self.calories
                        /
                        elapsed_minutes
                    )

                # -----------------------------
                # STORE DATA
                # -----------------------------

                self.track_id = int(track_id)

                self.confidence = (
                    float(confidence) * 100
                )

                # -----------------------------
                # DRAW TRACKING
                # -----------------------------

                cv2.rectangle(
                    frame,
                    (
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    self.smoothed_point,
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"ID: {self.track_id}",
                    (
                        center_x,
                        center_y - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {self.total_distance_m:.3f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Speed: {self.speed_cm_s:.2f} cm/s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # -----------------------------
            # SAVE FRAME
            # -----------------------------

            with self.lock:

                self.latest_frame = frame.copy()


    # ==========================================
    # GET STATISTICS
    # ==========================================

    def get_stats(self):

        with self.lock:

            return {

                "distance_m":
                    round(
                        self.total_distance_m,
                        3
                    ),

                "speed_cm_s":
                    round(
                        self.speed_cm_s,
                        2
                    ),

                "calories":
                    round(
                        self.calories,
                        4
                    ),

                "calories_per_minute":
                    round(
                        self.calories_per_minute,
                        4
                    ),

                "elapsed_seconds":
                    round(
                        self.elapsed_seconds,
                        1
                    ),

                "track_id":
                    self.track_id,

                "confidence":
                    round(
                        self.confidence,
                        1
                    ),

                "fish_length_cm":
                    self.fish_length_cm
            }


    # ==========================================
    # GET CAMERA FRAME
    # ==========================================

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()


    # ==========================================
    # STOP TRACKER
    # ==========================================

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2
            )

        self.camera.release()