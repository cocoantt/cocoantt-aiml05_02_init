from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from datetime import datetime

current_time = datetime.now()
def send_mail():
    content = MIMEMultipart()  #建立MIMEMultipart物件
    content["subject"] = "火災告警"  #郵件標題
    content["from"] = "penweru920@gmail.com"  #寄件者
    content["to"] = "penweru920@gmail.com" #收件者
    content.attach(MIMEText(f'{current_time}\n偵測事件:發生火災'))  #郵件內容

    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        try:
            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            smtp.login("penweru920@gmail.com", "tsuz keln rnxg cfel")  # 登入寄件者gmail
            smtp.send_message(content)  # 寄送郵件
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)

