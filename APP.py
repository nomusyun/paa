from flask import Flask, render_template, Response, jsonify, request
from djitellopy import Tello
import cv2
import numpy as np
import time

# --- YOLOv8 (Person Detection) の初期化 ---
MODEL_PATH = "yolov8n.onnx"
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
except Exception as e:
    print(f"YOLOモデルの読み込みエラー: {e}")
    net = None

app = Flask(__name__)

# Telloの初期化と接続
tello = Tello()
try:
    tello.connect()
    tello.streamon()
except Exception as e:
    print(f"Tello接続エラー: {e}")

# --- 制御用グローバル変数 ---
tracking_active = False  # PC画面からの承認による追跡フラグ

def detect_and_track_person(frame):
    global tracking_active
    if net is None:
        return frame

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0].T

    boxes, scores = [], []
    x_factor = width / 640.0
    y_factor = height / 640.0

    for row in outputs:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        confidence = classes_scores[class_id]

        if class_id == 0 and confidence > 0.5:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            box_w = int(w * x_factor)
            box_h = int(h * y_factor)

            boxes.append([left, top, box_w, box_h])
            scores.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)

    if len(indices) > 0:
        # OpenCVのバージョン違いによる型エラーを完全に防ぐ安全な処理
        if hasattr(indices, "flatten"):
            indices = indices.flatten()
        
        idx = indices[0]
        if isinstance(idx, (list, np.ndarray, tuple)):
            idx = idx[0]
            
        x, y, w, h = boxes[int(idx)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if tracking_active:
            control_drone_to_person(x, y, w, h, width, height)
    elif tracking_active:
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except:
            pass

    return frame

def control_drone_to_person(x, y, w, h, frame_width, frame_height):
    global tracking_active
    if not tracking_active:
        return

    center_x = frame_width // 2
    person_center_x = x + w // 2
    error_x = person_center_x - center_x

    yaw_speed = 0
    fb_speed = 0

    if abs(error_x) > 50:
        yaw_speed = int(error_x * 0.25)
        yaw_speed = max(-35, min(35, yaw_speed))

    person_area = w * h
    target_area = 50000
    if person_area < target_area * 0.8:
        fb_speed = 15
    elif person_area > target_area * 1.2:
        fb_speed = -15

    try:
        tello.send_rc_control(0, fb_speed, 0, yaw_speed)
    except:
        pass

def generate_frames():
    while True:
        try:
            frame_read = tello.get_frame_read()
            if frame_read is None:
                time.sleep(0.05)
                continue
            
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.05)
                continue

            frame = detect_and_track_person(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            time.sleep(0.05)
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def command():
    global tracking_active
    data = request.json
    cmd = data.get('cmd')
    
    try:
        if cmd == 'takeoff':
            tello.takeoff()
        elif cmd == 'land':
            tracking_active = False
            tello.land()
        elif cmd == 'start_tracking':
            tracking_active = True
        elif cmd == 'stop_tracking':
            tracking_active = False
            tello.send_rc_control(0, 0, 0, 0)
        elif cmd == 'emergency':
            tracking_active = False
            tello.emergency()
        elif cmd == 'flip':
            tello.flip_forward() # 一回転（前転）
        elif cmd == 'up': tello.move_up(30)
        elif cmd == 'down': tello.move_down(30)
        elif cmd == 'forward': tello.move_forward(30)
        elif cmd == 'back': tello.move_back(30)
        elif cmd == 'left': tello.move_left(30)
        elif cmd == 'right': tello.move_right(30)
        elif cmd == 'cw': tello.rotate_clockwise(30)
        elif cmd == 'ccw': tello.rotate_counter_clockwise(30)
    except Exception as e:
        print(f"コマンド実行エラー: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
    return jsonify({"status": "ok", "command": cmd, "tracking": tracking_active})

# --- 写真撮影用のエンドポイント ---
@app.route('/snapshot', methods=['POST'])
def snapshot():
    try:
        frame_read = tello.get_frame_read()
        if frame_read is not None:
            frame = frame_read.frame
            if frame is not None:
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"写真を保存しました: {filename}")
                return jsonify({"status": "ok", "filename": filename})
    except Exception as e:
        print(f"写真撮影エラー: {e}")
    return jsonify({"status": "error", "message": "Failed to capture frame"})

@app.route('/telemetry')
def telemetry():
    try:
        battery = tello.get_battery()
        height = tello.get_height()
        temp = tello.get_temperature()
    except:
        battery, height, temp = 0, 0, 0

    return jsonify({
        "battery": battery,
        "height": height,
        "temp": temp,
        "tracking": tracking_active
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
