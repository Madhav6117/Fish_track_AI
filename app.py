from flask import (
    Flask,
    render_template,
    Response,
    jsonify
)

from tracker_backend import FishTracker

import cv2


app = Flask(__name__)


# ==========================================
# START AI TRACKER
# ==========================================

tracker = FishTracker()

tracker.start()


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ==========================================
# LIVE CAMERA
# ==========================================

@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            frame = tracker.get_frame()

            if frame is None:
                continue

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                +
                frame_bytes
                +
                b"\r\n"
            )


    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ==========================================
# STATISTICS API
# ==========================================

@app.route("/api/stats")
def stats():

    return jsonify(
        tracker.get_stats()
    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )