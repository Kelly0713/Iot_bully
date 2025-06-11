import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email(subject, body):
    # 帳號設定
    sender_email = "acs111111@gm.ntcu.edu.tw"
    receiver_email = "acs111111@gm.ntcu.edu.tw"
    app_password = "utoz lird axdt jzvz"

    # 組 Email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # 寄出
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

    print("✅ 通知信已寄出")

def alert_fight(location="不明", time_str="未知時間"):
    subject = "⚠️ 偵測到打架行為"
    body = f"地點：{location}\n時間：{time_str}\n請警衛立即處理。"
    send_alert_email(subject, body)
