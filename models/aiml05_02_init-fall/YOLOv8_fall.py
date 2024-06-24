import numpy as np
import cv2
from ultralytics import YOLO
import shutil
import importlib.util

#from mail.content_fall import send_mail
# Construct the full path to module2.py
module2_path = 'C:/AIOT/Project_02/aiml05_02_init/models/mail/content_fall.py'  # Adjust the path accordingly
module2_name = 'module2'

# Importing a specific function dynamically
spec = importlib.util.spec_from_file_location(module2_name, module2_path)
module2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module2)

# Access the specific function
send_mail = getattr(module2, 'send_mail')

# Now you can use specific_function in module1.py 

# 目前路徑
video_path = 'C:/AIOT/Project_02/aiml05_02_init/models/aiml05_02_init-fall/Fall.mp4'
runs_folder_path = 'C:/AIOT/Project_02/aiml05_02_init/runs'                               # 輸出的資料夾路徑
results_path = runs_folder_path + '/detect/predict/image0.jpg'                     # 這一行路徑不用改
model_path = 'C:/AIOT/Project_02/aiml05_02_init/models/aiml05_02_init-fall/best.pt' # Fall 模型的路徑

cap = cv2.VideoCapture(video_path)

# 降低畫素以加速模型運算 (if needed)
# try 640*480 or 320*240
'''
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
#'''

model = YOLO(model_path)
# results = model(source=video_path, show=True, conf=0.4, save=True)


# send messages
fall_count = 0
warning_threshold = 10

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret or cv2.waitKey(30) == 27: break

    results = model(source=frame, show=True, conf=0.4, save=True)

    img = cv2.imread(results_path)
    
    # Warning Alarm preparation
    try:
        for r in results:
            if 1 in r.boxes.cls:   # falling
                fall_count += 1
            elif 0 in r.boxes.cls: # fallen
                fall_count += 1
    except RuntimeError: 
        fall_count = 0             # clear

    # send message once
    if fall_count == warning_threshold:  
        print('sending warning message')
        send_mail()
        



# 使用 shutil 的 rmtree 刪除輸出資料夾
try:
    shutil.rmtree(runs_folder_path)
    # print(f"成功刪除資料夾 {runs_folder_path}")
except OSError as e: pass
    # print(f"Error: {runs_folder_path} : {e.strerror}")


cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)