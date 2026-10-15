from flask import Flask, render_template, Response, jsonify, request
from djitellopy import Tello
import cv2
import time
import threading

# --- FaceDetectorYN (YuNet) の初期化 ---
MODEL_PATH = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(640, 480),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

app = Flask(__name__)

# Telloの初期化と接続
tello = Tello()
tello.connect()
tello.streamon()

# --- 制御用グローバル変数 ---
tracking_active = False  # PC画面からの承認による追跡フラグ

def control_drone_to_face(x, y, w, h, frame_width, frame_height):
    """顔の位置に応じてドローンを旋回・前進後退させる関数"""
    global tracking_active
    if not tracking_active:
        return

    # 画面の中心座標
    center_x = frame_width // 2
    center_y = frame_height // 2

    # 顔の中心座標
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    # 画面中心からのズレ（エラー値）
    error_x = face_center_x - center_x
    
    # 簡易的な追従制御（デッドゾーンを設けて安定させる）
    yaw_speed = 0
    fb_speed = 0

    # 左右のズレが大きい場合は旋回 (Yaw) で追う
    if abs(error_x) > 50:
        yaw_speed = int(error_x * 0.2)  # ゲイン調整
        yaw_speed = max(-30, min(30, yaw_speed)) # 速度制限

    # 顔の大きさ（面積）で距離を調整（近すぎたら後退、遠すぎたら前進）
    face_area = w * h
    target_area = 30000  # 理想の顔面積目安
    if face_area < target_area * 0.8:
        fb_speed = 15   # 前進
    elif face_area > target_area * 1.2:
        fb_speed = -15  # 後退

    # PCからの安全な操作（追跡が有効な間のみRC信号を送る）
    if tracking_active:
        tello.send_rc_control(0, fb_speed, 0, yaw_speed)

def generate_frames():
    global tracking_active
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            continue

        height, width, _ = frame.shape
        detector.setInputSize((width, height))
        
        _, faces = detector.detect(frame)
        
        if faces is not None and len(faces) > 0:
            # 最も手前（最初）の顔をターゲットにする
            face = faces[0]
            box = list(map(int, face[:4]))
            x, y, w, h = box[0], box[1], box[2], box[3]

            # 画面に枠を描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 追跡承認されている場合のみドローンを動かす
            if tracking_active:
                control_drone_to_face(x, y, w, h, width, height)
        else:
            # 顔が見つからないかつ追跡中の場合は停止ホバリング
            if tracking_active:
                tello.send_rc_control(0, 0, 0, 0)

        # 映像をJPEGに変換してストリーミング
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ルーティング ---

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def command():
    global tracking_active
    data = request.json
    cmd = data.get('cmd')
    
    if cmd == 'takeoff':
        tello.takeoff()
    elif cmd == 'land':
        tracking_active = False  # 着陸時は自動追跡を強制解除
        tello.land()
    elif cmd == 'start_tracking':
        tracking_active = True   # PC画面からの承認で追跡開始
    elif cmd == 'stop_tracking':
        tracking_active = False  # 追跡停止（ドローンはその場でホバリング）
        tello.send_rc_control(0, 0, 0, 0)
    elif cmd == 'emergency':
        tracking_active = False
        tello.emergency()
    # 通常の手動移動コマンド（実行時は安全のため一度停止信号を挟む等も可能）
    elif cmd == 'up': tello.move_up(30)
    elif cmd == 'down': tello.move_down(30)
    elif cmd == 'forward': tello.move_forward(30)
    elif cmd == 'back': tello.move_back(30)
    elif cmd == 'left': tello.move_left(30)
    elif cmd == 'right': tello.move_right(30)
    elif cmd == 'cw': tello.rotate_clockwise(30)
    elif cmd == 'ccw': tello.rotate_counter_clockwise(30)
        
    return jsonify({"status": "ok", "command": cmd, "tracking": tracking_active})

@app.route('/telemetry')
def telemetry():
    return jsonify({
        "battery": tello.get_battery(),
        "height": tello.get_height(),
        "temp": tello.get_temperature(),
        "tracking": tracking_active
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
