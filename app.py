from flask import Flask, render_template, Response, request, redirect, url_for, session, send_from_directory
import os
import cv2
import time
import threading
import playsound
import torch
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load YOLOv8 Nano Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# Create evidence folder if not exists
EVIDENCE_FOLDER = "evidence"
if not os.path.exists(EVIDENCE_FOLDER):
    os.makedirs(EVIDENCE_FOLDER)

# User Authentication Data
users = {'admin': 'password'}

# Global variables
camera_index = 0
alert_playing = False


def play_alert():
    """Plays an alert sound if not already playing."""
    global alert_playing
    if not alert_playing:
        alert_playing = True
        playsound.playsound('data/alert.mp3')
        alert_playing = False


def capture_evidence(frame):
    """Captures and saves evidence images when a violation is detected."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(EVIDENCE_FOLDER, f'violation_{timestamp}.jpg')
    cv2.imwrite(image_path, frame)
    print(f"[INFO] Evidence Captured: {image_path}")


def detect_person_talking(frame):
    """Detects persons using phones and marks them with a red frame."""
    global alert_playing

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    persons = []
    phones = []
    talking_detected = False

    # Step 1: Store detected persons and phones separately
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        label = model.names[class_id]

        if label == 'person':
            persons.append((x1, y1, x2, y2))

        elif label == 'cell phone':
            phones.append((x1, y1, x2, y2))

    # Step 2: Check if any phone is near a person's head
    for (px1, py1, px2, py2) in persons:
        person_height = py2 - py1
        head_y_threshold = py1 + (person_height * 0.3)

        for (hx1, hy1, hx2, hy2) in phones:
            if py1 <= hy1 <= head_y_threshold:
                talking_detected = True
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                break

    # Step 3: If talking detected, play alert and capture evidence
    if talking_detected:
        capture_evidence(frame)
        if not alert_playing:
            threading.Thread(target=play_alert, daemon=True).start()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def generate():
    """Generates frames for video streaming."""
    global camera_index
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera feed not available!")
            break

        frame = detect_person_talking(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    """Admin page for viewing captured evidence images."""
    if 'user' in session:
        evidence_list = sorted(os.listdir(EVIDENCE_FOLDER), reverse=True)
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
    """Logs out the admin user."""
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/evidence/<filename>')
def get_evidence(filename):
    """Serve evidence images."""
    return send_from_directory(EVIDENCE_FOLDER, filename)


@app.route('/change_camera', methods=['POST'])
def change_camera():
    """Changes the camera source based on user selection."""
    global camera_index
    camera_option = request.form.get('camera_option')

    if camera_option == 'laptop':
        camera_index = 0
    elif camera_option == 'usb':
        camera_index = 1

    return "Camera Changed Successfully!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
