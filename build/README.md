
# Project Title

A brief description of what this project does and who it's for

# 🚦 Real-Time Traffic Monitoring System

A Flask-based AI-powered web application that uses YOLOv8, OCR, and object tracking to detect over-speeding vehicles from video footage, extract their number plates, and notify the owners via email and SMS alerts.


## File Structure
project-root/

├── app.py                      # Main Flask application

├── requirements.txt            # Python dependencies

├── vehicle_data.csv            # Vehicle details (plate_number, name, contact, email)


├── templates/
│   └── index.html              # Frontend template


├── models/
│   └── yolo11n.pt              # YOLOv8 model file


├── notifier/   
├── email_sender.py         # Sends email alerts

└── sms_send.py             # Sends SMS alerts

└── uploads/                    # Uploaded video files

## Installation

1. Clone the repository:
   git clone https://github.com/Ankit638/RealTimeTrafficMonitoringSystem.git
   cd RealTimeTrafficMonitoringSystem

2. Install dependencies:
   pip install -r requirements.txt

If using CPU only:
   pip uninstall paddlepaddle-gpu
   pip install paddlepaddle==2.5.2

### How to Run 

Run the Flask app:
   python app.py

Then open your browser and visit:
   http://127.0.0.1:5000

### Features

Upload a video or use mobile camera

Detect vehicle movement and calculate speed

Use OCR to extract number plate

Verify plate against CSV data

Send Email & SMS if speed limit is violated

View all detected violations in JSON at /vehicle_info.csv

### ✅ **Optional: Use with Mobile Camera**

1. Install an IP webcam app on your phone (e.g. 'IP Webcam').

2. Replace the IP in app.py under use_mobile_camera() route.

3. Visit http://127.0.0.1:5000/use_mobile_camera to start live feed.

### ✅ **CSV Format (vehicle_data.csv)**

plate_number,name,contact,email
MH12AB1234,John Doe,9876543210,john@example.com
DL05CD9876,Jane Smith,8765432109,jane@example.com

### ✅ **Technologies Used**

Flask

YOLOv8 (Ultralytics)

OpenCV

PaddleOCR

SORT Tracking Algorithm

Pandas

SMTP & SMS API

### ✅ **Author**
Created by KintaMAX
For academic/project/demo purposes only.