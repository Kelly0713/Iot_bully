import cv2
import time
import pyttsx3
import threading
import speech_recognition as sr
from ultralytics import YOLO

# 初始化語音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 載入 YOLOv8 pose 模型
model = YOLO('yolov8n-pose.pt')

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("🔴 使用 YOLOv8 Pose 偵測霸凌（含語音關鍵字），按 Q 結束")

# 初始化警報冷卻機制
last_alarm_time = 0
cooldown = 5

# ✅ 關鍵字清單（可自行擴充）
bully_keywords = ["幹你娘機掰", "幹你娘", "幹", "廢物", "去死", "你很爛", "死肥豬"]

# 初始化語音辨識器
recognizer = sr.Recognizer()
mic = sr.Microphone()
audio_last_alarm_time = 0

def speak_warning():
    engine.say("已偵測到霸凌行為，請停止")
    engine.runAndWait()

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

# ✅ 背景語音監聽與關鍵字辨識
def audio_monitor():
    global audio_last_alarm_time
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎙️ 語音監聽中...")

        while True:
            try:
                audio = recognizer.listen(source, timeout=3)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"🗣️ 偵測語音內容：{text}")

                for keyword in bully_keywords:
                    if keyword in text:
                        now = time.time()
                        if now - audio_last_alarm_time > cooldown:
                            print("🚨 語音偵測到霸凌關鍵字！")
                            speak_warning()
                            audio_last_alarm_time = now
                        break

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"❌ 語音辨識錯誤：{e}")
                continue

# 🔁 啟動語音偵測為背景執行緒
audio_thread = threading.Thread(target=audio_monitor, daemon=True)
audio_thread.start()

# 🔍 主循環：視覺偵測霸凌姿勢
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
                print("🚨 偵測到可能的肢體霸凌行為！")
                speak_warning()
                last_alarm_time = now

            cv2.putText(
                annotated_frame,
                "🚨 Bullying Detected!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        cv2.imshow("Bullying Detection", annotated_frame)

    except Exception as e:
        print(f"❌ 推論時發生錯誤：{e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 結束資源
cap.release()
cv2.destroyAllWindows()
