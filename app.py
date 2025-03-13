from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask import send_from_directory
import cv2
import os
import time
import threading
import playsound
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Create evidence folder if not exists
if not os.path.exists('evidence'):
    os.makedirs('evidence')

# Simple User Data (For Authentication)
users = {'admin': 'password'}

# Global variable for camera selection
camera_index = 0

def play_alert():
    playsound.playsound('data/alert.mp3')

def capture_evidence(frame):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f'evidence/person_{timestamp}.jpg'
    cv2.imwrite(image_path, frame)
    print(f"[INFO] Evidence Captured: {image_path}")

def detect_person_talking(frame):
    results = model(frame)[0]
    detections = results.boxes.data.cpu().numpy()

    persons = []
    phones = []
    talking_detected = False  # Flag to confirm talking detection

    # Step 1: Store detected persons and phones separately
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        label = results.names[class_id]

        if label == 'person':
            persons.append((x1, y1, x2, y2))

        if label == 'cell phone':
            phones.append((x1, y1, x2, y2))

    # Step 2: Check if any phone is near a person's head
    for (px1, py1, px2, py2) in persons:
        person_height = py2 - py1  # Calculate person's height
        head_y_threshold = py1 + (person_height * 0.3)  # Define head region (Top 30%)

        for (hx1, hy1, hx2, hy2) in phones:
            # Check if the phone is within the head region
            if py1 <= hy1 <= head_y_threshold:
                talking_detected = True
                # Draw **RED** box around person
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                break  # No need to check further

    # Step 3: If talking detected, play alert and capture evidence
    if talking_detected:
        capture_evidence(frame)
        threading.Thread(target=play_alert).start()

        # Turn Whole Frame Red
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        alpha = 0.3  # Transparency for visibility
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def generate():
    global camera_index
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_person_talking(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'user' in session:
        evidence_list = os.listdir('evidence')
        evidence_list.sort(reverse=True)
        return render_template('admin.html', evidence_list=evidence_list)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('admin'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/change_camera', methods=['POST'])
def change_camera():
    global camera_index
    camera_option = request.form.get('camera_option')

    if camera_option == 'laptop':
        camera_index = 0  # Default Laptop Camera
    elif camera_option == 'usb':
        camera_index = 1  # External USB Camera

    return "Camera Changed Successfully!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
