<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# [Fish Track AI] 🎯


## Basic Details
### Team Name: Lays


### Team Members
- Team Lead: Madhav Menon - Muthoot Institute of Technology and Science
- Member 2: Cyril K Eldho - Muthoot Institute of Technology and Science

Project Description

FishTrack is an AI-powered computer vision system that monitors a fish using a USB webcam and estimates how far it swims. It uses custom-trained YOLO object detection, ByteTrack tracking, and real-time image processing to calculate swimming distance, speed, tracking confidence, and estimated energy expenditure.

The Problem (that doesn't exist)

Have you ever wondered:

“How many meters does my fish swim in a day?” 🐟💨

Fish don't have fitness trackers, smartwatches, or Strava accounts. So we're solving the problem of tracking a fish's daily swimming activity.

The Solution:
FishTrack watches the fish continuously using a USB webcam and AI.

The system:

Detects the fish using a custom-trained YOLO model.
Tracks its movement using ByteTrack.
Finds the fish's center position in every frame.
Smooths the movement to reduce camera/detection noise.
Converts pixel movement into real-world distance using calibration.
Calculates swimming speed.
Estimates calories/energy expenditure based on distance and assumed fish weight.
Displays everything on a real-time Flask dashboard.

Basically:

Camera → AI → Tracking → Distance → Speed → Calories → Fancy Dashboard

Technical Details
Technologies/Components Used
For Software:
Python 3.13
Flask – Web dashboard and API
OpenCV – Webcam capture and image processing
Ultralytics YOLO11n – Custom fish detection
ByteTrack – Object tracking
HTML5
CSS3
JavaScript
Chart.js – Real-time data visualization
Label Studio – Dataset annotation
PowerShell / VS Code – Development tools
AI / Machine Learning
Custom-trained YOLO11n
Custom fish dataset
YOLO-format annotations
Object detection + tracking pipeline
Model trained for 30 epochs
Input resolution: 640 × 640
Model Performance

Our trained model achieved approximately:

Precision: 99.8%
Recall: 100%
mAP@50: 99.5%
mAP@50-95: 78%
For Hardware:
Main Components
💻 Laptop/PC
📷 USB Webcam
🐟 Fish
🥣 Transparent fish bowl
💡 Suitable lighting
Specifications

Camera

USB webcam
Used as the primary monitoring camera
Camera index: 1

Fish Bowl

Transparent bowl
Fish visible from the camera
Camera positioned above/near the bowl

Computer

Runs the YOLO model
Processes webcam frames
Runs Flask server
Displays the monitoring dashboard
Tools Required
Python
VS Code
USB webcam
Internet connection for initial package installation
Fish 
Bowl 


Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/FishTrack.git

Go into the project folder:

cd FishTrack

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

.\venv\Scripts\Activate.ps1

Install the required libraries:

pip install -r requirements.txt

If required, install the main dependencies manually:

pip install flask opencv-python ultralytics
Dataset Preparation

The project uses a custom fish dataset.

Dataset structure:

dataset/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/

The dataset is annotated using Label Studio with the fish class.

Model Training

Train the custom YOLO model using:

python -c "from ultralytics import YOLO; model=YOLO('yolo11n.pt'); model.train(data='data.yaml', epochs=30, imgsz=640, batch=8, device='cpu', workers=0)"

After training, the best model will be located at:

runs/detect/train/weights/best.pt
Run

Make sure the USB webcam is connected.

Activate the virtual environment:

.\venv\Scripts\Activate.ps1

Start the Flask application:

python app.py

You should see:

* Running on http://127.0.0.1:5000

Open your browser and go to:

http://127.0.0.1:5000

🎉 The FishTrack AI Dashboard should now be running.

System Architecture
              🐟 FISH
                 │
                 ▼
          📷 USB WEBCAM
                 │
                 ▼
             OpenCV
                 │
                 ▼
          YOLO Fish Detection
                 │
                 ▼
             ByteTrack
                 │
                 ▼
       Fish Center Coordinates
                 │
                 ▼
       Movement Smoothing
                 │
                 ▼
        Pixel → CM Calibration
                 │
                 ▼
        Swimming Distance
                 │
          ┌──────┴──────┐
          ▼             ▼
        Speed        Calories
          │             │
          └──────┬──────┘
                 ▼
        🌐 Flask Dashboard
⚠️ Project Note

FishTrack estimates image-plane swimming distance from the camera footage. Because a normal webcam does not directly measure the fish's 3D movement, the calculated distance is an approximation based on camera calibration.
# Screenshots (Add at least 3)
![Screenshot1]<img width="1600" height="820" alt="Dashboard 1" src="https://github.com/user-attachments/assets/68b4a0b2-d236-4ada-b61b-4ae63d276e7c" />
Image of our dashboard -1

![Screenshot2]<img width="1600" height="878" alt="Dashboard 2" src="https://github.com/user-attachments/assets/7b981951-0f16-41f0-96c1-53ae47959eb9" />
Image of our dashboard -2

![Screenshot3]<img width="1600" height="909" alt="IDE" src="https://github.com/user-attachments/assets/2c1646d6-eb65-4c44-8e68-1c8df24f7f25" />
IDE interface

# Diagrams
![Workflow]<img width="1536" height="1024" alt="Workflow" src="https://github.com/user-attachments/assets/cfc83829-7e06-4267-bf2b-6faf55512f86" />
FishTrack AI – Workflow Model Explanation
1. Input – Fish and Camera
The system begins with a fish placed inside a transparent bowl. A USB webcam is positioned to continuously observe the fish. The camera provides the live video required for tracking the fish's movement.

2. Frame Capture
The live video is captured frame by frame using OpenCV. Each frame is processed before being passed to the AI model.

3. Fish Detection
Each frame is given to the custom-trained YOLO11n model. The model identifies the fish and produces a bounding box around it along with a confidence score indicating how certain the detection is.

4. Object Tracking
After detection, ByteTrack assigns a unique tracking ID to the fish. This allows the system to identify the same fish across consecutive frames and continuously follow its movement.

5. Center Point Extraction
The center coordinates (x, y) of the detected bounding box are calculated. These coordinates represent the approximate position of the fish in each frame.

6. Movement Smoothing
Small fluctuations in the detected position can occur because of camera movement, water movement, or detection noise. A smoothing technique is therefore applied to the center coordinates to produce a more stable movement path.

7. Pixel-to-Centimeter Conversion
The movement detected by the camera is initially measured in pixels. A calibration value is used to convert pixel movement into real-world distance in centimeters.

For this prototype, the fish's approximate length is used as the reference for calibration.

8. Distance Calculation
The system calculates the distance between consecutive positions of the fish. These individual movements are added together to obtain the total swimming distance.

The distance is finally represented in meters.

9. Speed Calculation
The swimming speed is calculated using:

Speed = Total Distance / Elapsed Time

The result is continuously updated as the fish moves.

10. Energy / Calorie Estimation
The system also provides an estimated calorie expenditure based on the swimming distance, assumed fish weight, and an energy-cost factor.

This value is an engineering estimate for the prototype, rather than a scientifically validated measurement of fish metabolism.

11. Data Storage and API
The calculated values such as distance, speed, calories, tracking ID, confidence, and elapsed time are maintained by the backend. A Flask API provides these values to the dashboard in real time.

12. Dashboard Visualization
A Flask web application displays the collected information through a user-friendly dashboard.

The dashboard provides:

Live camera feed
Swimming distance
Swimming speed
Estimated calories
Tracking confidence
Tracking ID
Elapsed tracking time
Graphs and analytics
13. Output
The processed information is presented as real-time visual data. The user can observe both the fish and its calculated swimming statistics through the dashboard.

14. Final Result
The final system provides an AI-based method for monitoring fish movement and estimating its swimming activity.

The complete workflow can be summarized as:

Fish → USB Camera → Frame Capture → YOLO11n Detection → ByteTrack Tracking → Center Point → Smoothing → Pixel-to-CM Conversion → Distance Calculation → Speed & Calorie Estimation → Flask API → Dashboard → Real-Time Results

Technologies Used
Python – Main programming language
OpenCV – Camera and image processing
YOLO11n – Fish detection
ByteTrack – Fish tracking
Flask – Backend and web application
Chart.js – Dashboard graphs
HTML/CSS/JavaScript – Dashboard interface

### Project Demo
# Video
https://drive.google.com/file/d/1NQPP1s8Qe5IEC5OIX5It2L8BJIr27NJO/view?usp=sharing





---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



