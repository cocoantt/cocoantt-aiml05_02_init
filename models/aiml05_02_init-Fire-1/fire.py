from ultralytics import YOLO
# from ultralytics.nn.modules.conv import DCNConv
import cvzone
import cv2
import math

# Running real time from webcam
cap = cv2.VideoCapture(0) 
model = YOLO('fire.pt')


# Reading the classes
classnames = ['fire', 'smoke', 'other']

fall_count = 0
warning_threshold = 10

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    result = model(frame,stream=True)

    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 70:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)
        
    try:
        for r in result:
            if 0 in r.boxes.cls:   # fire
                fall_count += 1
            elif 1 in r.boxes.cls: # smoke
                fall_count += 1
    except RuntimeError: 
        fall_count = 0             # clear

    # send message once
    if fall_count == warning_threshold:  
        print('sending warning message')


    cv2.imshow('frame',frame)
    cv2.waitKey(1)
