from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt

from UIFunctions import *
from ui.home import Ui_MainWindow

#from utils.capnums import Camera
from utils.rtsp_win import Window
from collections import deque
import numpy as np
import time
import json
import sys
import cv2
import os

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')  # image suffixes
VID_FORMATS = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm')  # video suffixes

def is_video_file(path):
    suffix = path.split('.')[-1].lower()

    if suffix in IMG_FORMATS:
        return False
    elif suffix in VID_FORMATS:
        return True
    else:
        print(f"Suffix '{suffix}' is invalid!")
        raise ValueError("Invalid file suffix")

class YoloPredictor(QObject):
    yolo2main_pre_img = Signal(np.ndarray)   # raw image signal
    yolo2main_res_img = Signal(np.ndarray)   # test result signal
    yolo2main_status_msg = Signal(str)       # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)              # fps
    yolo2main_labels = Signal(dict)          # Detected target results (number of each category)
    yolo2main_progress = Signal(int)         # Completeness
    yolo2main_class_num = Signal(int)        # Number of categories detected
    yolo2main_target_num = Signal(int)       # Targets detected

    def __init__(self):
        super().__init__()

        # GUI args
        self.used_model_name = None      # The detection model name to use
        self.source = ''                 # input source
        self.stop_dtc = False            # Termination detection
        self.continue_dtc = True         # pause   
        self.labels_dict = {}            # return a dictionary of results
        self.progress_value = 0          # progress bar
        self.YoloConfig = dict()
        self.YoloConfig['model'] = None
        self.YoloConfig['iou'] = 0.70  # iou
        self.YoloConfig['conf'] = 0.25 # conf
        self.YoloConfig['rate'] = 30  # delay, ms
        self.YoloConfig['save_res'] = False  # Save test results
        self.YoloConfig['save_txt'] = False  # save label(txt) file

        # Usable if setup is done
        self.model = None
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.data_path = None
        self.source_type = None
        self.batch = None

        self.fps_counter = 0
        self.fps_frames = deque(maxlen=10)  # 存储最近 10 帧的时间
        self.fps = 0

    def loadmodel(self):
        if not self.model or self.used_model_name != self.YoloConfig['model']:
            self.yolo2main_status_msg.emit('Loading Model...')
            print('model used = ', self.YoloConfig['model'])
            self.model = YOLO(self.YoloConfig['model'])
            self.used_model_name = self.YoloConfig['model']

    def run(self):
        print("running detection ......")
        # set model
        self.loadmodel()
        self.yolo2main_status_msg.emit('Detecting...')
        print('conf threshold = ', self.YoloConfig['conf'])
        print('iou threshold = ', self.YoloConfig['iou'])
        print('save = ', self.YoloConfig['save_res'])
        print('save_txt = ', self.YoloConfig['save_txt'])

        if not is_video_file(self.source):
            res = self.model.predict(self.source, save=self.YoloConfig['save_res'],
                                     save_txt=self.YoloConfig['save_txt'], imgsz=640, conf=self.YoloConfig['conf'],
                                     iou=self.YoloConfig['iou'], device=0)

            preprocess_speed = res[0].speed['preprocess']
            inference_speed = res[0].speed['inference']
            postprocess_speed = res[0].speed['postprocess']
            total_infer_speed = preprocess_speed + inference_speed + postprocess_speed
            fps = 1000 / total_infer_speed
            print("FPS:", fps)

            detected_boxes = res[0].boxes
            # print(res[0])

            print("detected boxes ......")
            # Create an empty list to store the class IDs
            class_ids = []
            # Iterate over each box and extract the class ID
            for box in detected_boxes:
                class_id = box.cls  # get the class id
                class_id_cpu = class_id.cpu()  # move the value to CPU
                class_id_int = int(class_id_cpu.item())  # convert to integer
                class_ids.append(class_id_int)  # append to the list

            # Print the class ID
            print("class ids = ", class_ids)
            total_classes = len(set(class_ids))
            total_ids = len(class_ids)
            # Send test results
            orig_img = res[0].orig_img
            annotated_img = res[0].plot()
            self.yolo2main_pre_img.emit(orig_img)  # Before testing
            self.yolo2main_res_img.emit(annotated_img)  # after detection
            # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
            self.yolo2main_class_num.emit(total_classes)
            self.yolo2main_target_num.emit(total_ids)
            self.yolo2main_fps.emit(str(int(fps)))
            self.yolo2main_status_msg.emit('Detection completed')

        else:
            cap = cv2.VideoCapture(self.source)

            # Loop through the video frames
            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate
            all_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # total frames
            while cap.isOpened():
                # Termination detection
                if self.stop_dtc:
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    self.stop_dtc = False
                    break
                # Read a frame from the video
                success, frame = cap.read()
                count += 1  # frame count +1
                self.progress_value = int(count / all_count * 1000)  # progress bar(0~1000)
                self.yolo2main_progress.emit(self.progress_value)  # progress bar
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                if success:
                    # Run YOLOv8 inference on the frame
                    res = self.model.predict(frame, save=self.YoloConfig['save_res'], save_txt = self.YoloConfig['save_txt'], imgsz=640, conf=self.YoloConfig['conf'], iou=self.YoloConfig['iou'], device=0)

                    preprocess_speed = res[0].speed['preprocess']
                    inference_speed = res[0].speed['inference']
                    postprocess_speed = res[0].speed['postprocess']
                    total_infer_speed = preprocess_speed + inference_speed + postprocess_speed
                    fps = 1000 / total_infer_speed
                    print("FPS:", fps)

                    detected_boxes = res[0].boxes

                    print("detected boxes ......")
                    # Create an empty list to store the class IDs
                    class_ids = []
                    # Iterate over each box and extract the class ID
                    for box in detected_boxes:
                        class_id = box.cls  # get the class id
                        class_id_cpu = class_id.cpu()  # move the value to CPU
                        class_id_int = int(class_id_cpu.item())  # convert to integer
                        class_ids.append(class_id_int)  # append to the list

                    # Print the class ID
                    print("class ids = ", class_ids)
                    total_classes = len(set(class_ids))
                    total_ids = len(class_ids)
                    # Send test results
                    orig_img = res[0].orig_img
                    annotated_img = res[0].plot()
                    self.yolo2main_pre_img.emit(orig_img)  # Before testing
                    self.yolo2main_res_img.emit(annotated_img)  # after detection
                    self.yolo2main_class_num.emit(total_classes)
                    self.yolo2main_target_num.emit(total_ids)

                    # if self.YoloConfig['rate'] != 0:
                    #     time.sleep(self.YoloConfig['rate'] / 1000)  # delay , ms
                else:
                    # Break the loop if the end of the video is reached
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

    def run_camera(self, frame):
        print("running camera detection ......")
        # set model
        self.loadmodel()
        self.yolo2main_status_msg.emit('Detecting...')
        print('conf threshold = ', self.YoloConfig['conf'])
        print('iou threshold = ', self.YoloConfig['iou'])
        print('save = ', self.YoloConfig['save_res'])
        print('save_txt = ', self.YoloConfig['save_txt'])

        res = self.model.predict(frame, save=self.YoloConfig['save_res'],
                                 save_txt=self.YoloConfig['save_txt'], imgsz=640, conf=self.YoloConfig['conf'],
                                 iou=self.YoloConfig['iou'], device=0)

        preprocess_speed = res[0].speed['preprocess']
        inference_speed = res[0].speed['inference']
        postprocess_speed = res[0].speed['postprocess']
        total_infer_speed = preprocess_speed + inference_speed + postprocess_speed
        fps = 1000 / total_infer_speed
        #print("FPS:", fps)

        # 每读取一帧增加计数器
        self.fps_counter += 1

        # 如果达到 10 帧，则计算 FPS
        if self.fps_counter == 10:
            elapsed_time = self.fps_frames[-1] - self.fps_frames[0]  # 计算最近 10 帧的时间差
            average_fps = 10 / elapsed_time
            self.fps = int(average_fps)
            print("Average FPS:", self.fps)

            self.fps_counter = 0  # 重置计数器
            self.fps_frames.clear()  # 清空时间队列

        # 记录当前时间
        self.fps_frames.append(time.time())

        detected_boxes = res[0].boxes
        # print(res[0])

        print("detected boxes ......")
        # Create an empty list to store the class IDs
        class_ids = []
        # Iterate over each box and extract the class ID
        for box in detected_boxes:
            class_id = box.cls  # get the class id
            class_id_cpu = class_id.cpu()  # move the value to CPU
            class_id_int = int(class_id_cpu.item())  # convert to integer
            class_ids.append(class_id_int)  # append to the list

        # Print the class ID
        print("class ids = ", class_ids)
        total_classes = len(set(class_ids))
        total_ids = len(class_ids)
        # Send test results
        orig_img = res[0].orig_img
        annotated_img = res[0].plot()
        self.yolo2main_pre_img.emit(orig_img)  # Before testing
        self.yolo2main_res_img.emit(annotated_img)  # after detection
        self.yolo2main_class_num.emit(total_classes)
        self.yolo2main_target_num.emit(total_ids)
        self.yolo2main_fps.emit(str(self.fps))  #  FPS

    def run_rtsp_frame(self, frame):
        print("running rtsp/http stream detection ......")
        print("stream source = ", self.source)
        # set model
        self.loadmodel()
        self.yolo2main_status_msg.emit('Detecting...')
        print('conf threshold = ', self.YoloConfig['conf'])
        print('iou threshold = ', self.YoloConfig['iou'])
        print('save = ', self.YoloConfig['save_res'])
        print('save_txt = ', self.YoloConfig['save_txt'])

        # Run YOLOv8 inference on the frame
        res = self.model.predict(frame, save=self.YoloConfig['save_res'], save_txt = self.YoloConfig['save_txt'], imgsz=640, conf=self.YoloConfig['conf'], iou=self.YoloConfig['iou'], device=0)

        preprocess_speed = res[0].speed['preprocess']
        inference_speed = res[0].speed['inference']
        postprocess_speed = res[0].speed['postprocess']
        total_infer_speed = preprocess_speed + inference_speed + postprocess_speed

        # 每读取一帧增加计数器
        self.fps_counter += 1

        # 如果达到 10 帧，则计算 FPS
        if self.fps_counter == 10:
            elapsed_time = self.fps_frames[-1] - self.fps_frames[0]  # 计算最近 10 帧的时间差
            average_fps = 10 / elapsed_time
            self.fps = int(average_fps)
            print("Average FPS:", self.fps)

            self.fps_counter = 0  # 重置计数器
            self.fps_frames.clear()  # 清空时间队列

        # 记录当前时间
        self.fps_frames.append(time.time())

        detected_boxes = res[0].boxes

        print("detected boxes ......")
        # Create an empty list to store the class IDs
        class_ids = []
        # Iterate over each box and extract the class ID
        for box in detected_boxes:
            class_id = box.cls  # get the class id
            class_id_cpu = class_id.cpu()  # move the value to CPU
            class_id_int = int(class_id_cpu.item())  # convert to integer
            class_ids.append(class_id_int)  # append to the list

        # Print the class ID
        print("class ids = ", class_ids)
        total_classes = len(set(class_ids))
        total_ids = len(class_ids)
        # Send test results
        orig_img = res[0].orig_img
        annotated_img = res[0].plot()
        self.yolo2main_pre_img.emit(orig_img)  # Before testing
        self.yolo2main_res_img.emit(annotated_img)  # after detection
        self.yolo2main_class_num.emit(total_classes)
        self.yolo2main_target_num.emit(total_ids)
        self.yolo2main_fps.emit(str(self.fps))  # FPS

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    main2yolo_loadmodel_sgl = Signal()  # The main window sends a load model signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(95, 95, 239))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        
        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)     # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()                                # Create a Yolo instance
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.YoloConfig['model'] = "./models/%s" % self.select_model
        self.yolo_thread = QThread()                                  # Create yolo thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video)) 
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))             
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))              
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))       
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))     
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.main2yolo_loadmodel_sgl.connect(self.yolo_predict.loadmodel)
        self.yolo_predict.moveToThread(self.yolo_thread)              

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)
        
        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.open_camera)  #open_cam
        self.src_rtsp_button.clicked.connect(self.open_rtsp) #open_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # top right settings button
        
        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('Please select a image/video source before starting detection...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')           
                self.yolo_predict.continue_dtc = True   # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
            self.pre_video.clear()           # clear image display  
            self.res_video.clear()          
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'    
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']     
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Loaded File：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            #self.stop()

    def open_camera(self):
        """Initialize camera.
        """
        print("open camera...")
        self.capture = cv2.VideoCapture(0)
        self.video_size = QSize(640, 480)  # 设置视频帧的宽度和高度
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer_cam = QTimer()
        self.timer_cam.timeout.connect(self.predict_cam_stream)
        if self.yolo_predict.YoloConfig['rate'] != 0:
            self.timer_cam.start(self.yolo_predict.YoloConfig['rate'])

    def predict_cam_stream(self):
        """Predict frame from camera
        """
        _, frame = self.capture.read()
        self.yolo_predict.run_camera(frame)
        if self.yolo_predict.stop_dtc:
            self.timer_cam.stop()
            self.yolo_predict.yolo2main_status_msg.emit('Detection terminated!')
            self.yolo_predict.stop_dtc = False

    # select network source
    def open_rtsp(self):
        print("open rtsp/http ...")
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "http://10.0.0.29:8554/test"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']

        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        self.rtsp_window.close()
        try:
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading stream：{}'.format(ip))

            self.capture = cv2.VideoCapture(ip)
            if not self.capture.isOpened():
                print("failed to open stream source.")
            else:
                print("stream source opened.")
            self.timer_rtsp = QTimer()
            self.timer_rtsp.timeout.connect(self.predict_rtsp_stream)
            if self.yolo_predict.YoloConfig['rate'] !=0:
                self.timer_rtsp.start(self.yolo_predict.YoloConfig['rate'])
        except Exception as e:
            self.show_status('%s' % e)

    def predict_rtsp_stream(self):
        """Read frame from rtsp/http stream.
        """
        _, frame = self.capture.read()
        self.yolo_predict.run_rtsp_frame(frame)
        if self.yolo_predict.stop_dtc:
            self.timer_rtsp.stop()
            self.yolo_predict.yolo2main_status_msg.emit('Detection terminated!')
            self.yolo_predict.stop_dtc = False
    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.YoloConfig['save_res'] = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.YoloConfig['save_res'] = True
    
    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.YoloConfig['save_txt'] = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.YoloConfig['save_txt'] = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.70
            conf = 0.25
            rate = 30
            save_res = 0   
            save_txt = 0    
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.70
                conf = 0.25
                rate = 30
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.yolo_predict.YoloConfig['iou'] = iou
        self.yolo_predict.YoloConfig['conf'] = conf
        self.yolo_predict.YoloConfig['rate'] = rate
        self.iou_spinbox.setValue(iou)
        self.conf_spinbox.setValue(conf)
        self.speed_slider.setValue(rate)
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.YoloConfig['save_res'] = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.YoloConfig['save_txt'] = (False if save_txt==0 else True )
        self.run_button.setChecked(False)  
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100)        # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.YoloConfig['iou'] = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.YoloConfig['conf'] = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.YoloConfig['speed'] = x  # ms

    # change model
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.YoloConfig['model'] = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)
        # load model signal emission
        self.main2yolo_loadmodel_sgl.emit()


    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
