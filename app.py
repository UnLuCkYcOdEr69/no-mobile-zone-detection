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

# Load YOLOv8 model directly
model = YOLO('yolov8m.pt')

# Create evidence folder if not exists
if not os.path.exists('evidence'):
    os.makedirs('evidence')

# Simple User Data (In Real Application Use Database)
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


@app.route('/evidence/<filename>')
def get_evidence(filename):
    return send_from_directory('evidence', filename)


def detect_person_phone(frame):
    results = model(frame)[0]
    detections = results.boxes.data.cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        label = results.names[class_id]

        # Detect person and phone
        if label == 'person':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if label == 'cell phone':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Capture Image and Trigger Alert Instantly
            capture_evidence(frame)
            threading.Thread(target=play_alert).start()

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
        frame = detect_person_phone(frame)
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
        # Display all evidence
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

