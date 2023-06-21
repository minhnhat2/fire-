
import cvzone
import cv2
import math
from ultralytics import YOLO

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_to_server():
    url = 'http://192.168.38.143:3000/dashboard'
    deviceId = '642ba948492b600742b592cc'
    payload = {'deviceId': deviceId}

    try:
        response = requests.post(url, data=payload)
        # Handle the server response if necessary
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(e)

def send_email():
    sender_email = 'lnhat1938@gmail.com'
    sender_password = 'minhnhat2002'
    receiver_email = 'minhnhat2k2135@gmail.com'
    subject = 'Fire Detected'
    message = 'Fire has been detected! Please take necessary actions.'

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")
    except smtplib.SMTPException as e:
        print("Error: Unable to send email")
        print(e)


# Running real time from webcam
cap = cv2.VideoCapture('fire4.mp4')
model = YOLO('fire1.pt')

# Reading the classes
classnames = ['fire']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    fire_detected = False

    # Getting bbox, confidence and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
                fire_detected = True

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if fire_detected:
        send_email()
        send_to_server()


