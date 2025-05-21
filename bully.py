import cv2
import time
import pyttsx3
from ultralytics import YOLO

# 初始化語音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 語速可調整

# 載入 YOLOv8 pose 模型
model = YOLO('yolov8n-pose.pt')

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("🔴 使用 YOLOv8 Pose 偵測霸凌，按 Q 結束")

# 初始化警報冷卻機制
last_alarm_time = 0
cooldown = 5  # 每 5 秒最多觸發一次

def check_bullying(keypoints):
    for person in keypoints:
        if len(person) >= 6:
            left_shoulder = person[5]
            right_shoulder = person[6]
            if left_shoulder[1] > 0 and right_shoulder[1] > 0:
                diff = abs(left_shoulder[1] - right_shoulder[1])
                if diff > 60:
                    return True
    return False

def speak_warning():
    engine.say("已偵測到霸凌行為，請停止")
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ 讀取畫面失敗，略過此幀")
        continue

    try:
        results = model.predict(source=frame, conf=0.5, task='pose', verbose=False)
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        annotated_frame = results[0].plot()

        if check_bullying(keypoints_list):
            now = time.time()
            if now - last_alarm_time > cooldown:
                print("🚨 偵測到可能的霸凌行為！")
                speak_warning()
                last_alarm_time = now

        cv2.imshow("Bullying Detection", annotated_frame)

    except Exception as e:
        print(f"❌ 推論時發生錯誤：{e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2
