from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
import cv2
from ultralytics import YOLO
import time
import numpy as np
import imutils
import re
from sort import Sort
from paddleocr import PaddleOCR
from notifier.email_sender import send_email
import pandas as pd
from notifier.sms_send import send_sms_alert


cap = None


# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Constants & Globals
SPEED_LIMIT = 80
video_source = None
current_video = None
violated_vehicles = {}
stored_plates = {}
notified_vehicles = set()

# Load models and data
model = YOLO("models/yolo11n.pt").to("cuda")
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
tracker = Sort()
plate_regex = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}$')

# Load vehicle info data
vehicle_data = pd.read_csv("vehicle_data.csv")  # Should contain plate_number,name,contact,email

# =============================== FUNCTIONS ===============================

def set_video_source(filename):
    global video_source, current_video, violated_vehicles, stored_plates, notified_vehicles
    video_source = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    current_video = video_source
    violated_vehicles.clear()
    stored_plates.clear()
    notified_vehicles.clear()

def extract_number_plate(frame, x1, y1, x2, y2):
    try:
        plate_roi = frame[y1:y2, x1:x2]
        plate_roi = imutils.resize(plate_roi, width=250)
        result = ocr.ocr(plate_roi)
        if result and result[0]:
            text = result[0][0][-1][0].replace(" ", "").upper()
            if plate_regex.match(text):
                return text
    except Exception as e:
        print(f"❌ OCR Error: {e}")
    return "Not Detected"

def get_owner_details(plate_number):
    match = vehicle_data[vehicle_data["plate_number"].str.upper() == plate_number]
    if not match.empty:
        row = match.iloc[0]
        return {
            "name": row["name"],
            "contact": row["contact"],
            "email": row["email"]
        }
    return None

def generate_frames():
    global video_source
    if not video_source:
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Error: Could not open video source!")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    


    frame_count = 0
    previous_positions = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        try:
            results = model(frame, device='cuda', conf=0.5)
            detections = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:  # Vehicle classes
                        detections.append([x1, y1, x2, y2, float(conf)])

            if detections:
                tracked_objects = tracker.update(np.array(detections, dtype=np.float32))
                DETECTION_LINE_Y = 300

                for obj in tracked_objects:
                    x1, y1, x2, y2, obj_id = map(int, obj)
                    center_y = (y1 + y2) // 2
                    if center_y < DETECTION_LINE_Y:
                        continue

                    current_time = time.time()
                    center_x = (x1 + x2) // 2
                    if obj_id in previous_positions:
                        prev_x, prev_y, prev_time = previous_positions[obj_id]
                        distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        time_diff = current_time - prev_time
                        speed = (distance / time_diff) * 5.6
                    else:
                        speed = 0

                    previous_positions[obj_id] = (center_x, center_y, current_time)

                    # Number Plate Extraction
                    plate_number = stored_plates.get(obj_id)
                    if not plate_number:
                        plate_number = extract_number_plate(frame, x1, y1, x2, y2)
                        stored_plates[obj_id] = plate_number

                    # Get vehicle owner info
                    owner = get_owner_details(plate_number)

                    # Violation Logic
                    violation = "None"
                    color = (0, 255, 0)
                    if speed > SPEED_LIMIT:
                        violation = "Over Speed"
                        color = (0, 0, 255)

                    violated_vehicles[obj_id] = {
                        "id": obj_id,
                        "plate": plate_number,
                        "speed": f"{speed:.2f} km/h",
                        "violation": violation
                    }

                    # Send Email Notification (only if match found in CSV)
                    if violation != "None" and obj_id not in notified_vehicles and owner:
                        subject = f"Traffic Violation Alert - {plate_number}"
                        body = f"""
🚨 Violation Detected 🚨

Vehicle Plate: {plate_number}
Owner: {owner['name']}
Contact: {owner['contact']}
Speed: {speed:.2f} km/h
Violation: {violation}

Please take necessary action.
                        """
                        try:
                            send_email(owner["email"], subject, body)
                            print(f"📧 Email sent to {owner['email']} for vehicle {plate_number}")
                            notified_vehicles.add(obj_id)
                        except Exception as e:
                            print(f"❌ Email sending failed: {e}")

                        try:
                            sms_msg = f"Your vehicle {plate_number} was detected over-speeding at {speed:.2f} km/h. Please adhere to road safety rules."
                            send_sms_alert(plate_number, sms_msg)
                        except Exception as sms_err:
                            print(f"❌ SMS failed: {sms_err}")

                    # Annotate Frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} Plate:{plate_number} Speed:{speed:.1f} km/h",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            continue

    cap.release()

# =============================== ROUTES ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    set_video_source(file.filename)
    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/vehicle_info")
def vehicle_info():
    return jsonify(list(violated_vehicles.values()))

@app.route("/use_mobile_camera")
def use_mobile_camera():
    global video_source
    # Change the IP here to match your phone's IP webcam stream
    ip_webcam_url = "http://192.168.29.177:8080/video"
    video_source = ip_webcam_url
    return redirect(url_for("index"))

# =============================== RUN ===============================

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
