import cv2
import time
import pyttsx3
import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

# 初始化語音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 初始化 YOLOv8 pose 模型
model = YOLO('yolov8n-pose.pt')

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

# 初始參數
last_alarm_time = 0
cooldown = 5  # 秒
sound_threshold = 0.4  # 聲音突波門檻

print("🔴 使用聲音 + 姿勢結合方式偵測霸凌，按 Q 結束")

def check_bullying_pose(keypoints):
    for person in keypoints:
        if len(person) >= 6:
            left_shoulder = person[5]
            right_shoulder = person[6]
            if left_shoulder[1] > 0 and right_shoulder[1] > 0:
                diff = abs(left_shoulder[1] - right_shoulder[1])
                if diff > 60:
                    return True
    return False

def check_sound_peak():
    duration = 0.5  # 秒
    fs = 16000  # 取樣率
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    volume_norm = np.linalg.norm(audio)  # 音量強度
    return volume_norm > sound_threshold

def speak_warning():
    engine.say("已偵測到霸凌行為，請立刻停止")
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ 讀取畫面失敗，略過此幀")
        continue

    try:
        # 姿勢辨識
        results = model.predict(source=frame, conf=0.5, task='pose', verbose=False)
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        annotated_frame = results[0].plot()

        # 聲音偵測（並非每一幀都偵測聲音，以節省資源）
        now = time.time()
        pose_alert = check_bullying_pose(keypoints_list)
        sound_alert = check_sound_peak() if pose_alert else False

        if pose_alert and sound_alert and now - last_alarm_time > cooldown:
            print("🚨 結合聲音與姿勢偵測到霸凌行為！")

            # 警告語音
            speak_warning()

            # 儲存當前畫面
            filename = f"bullying_detected_{int(now)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 影像儲存於：{filename}")

            last_alarm_time = now

        cv2.imshow("Bullying Detection", annotated_frame)

    except Exception as e:
        print(f"❌ 推論錯誤：{e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
