import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette, Color
from supervision import Detections, BoxAnnotator
colors=[Color(r=255, g=64, b=64), Color(r=255, g=161, b=160)]

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device used: {self.device}")

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names 
        self.box_annotator = BoxAnnotator(color=ColorPalette(colors=colors), thickness=3)
        
    def load_model(self):
        model = YOLO("custom_train_yolov10s.pt")
        model.fuse()
        return model

    def plot_boxes(self, results, frame, conf_threshold = 0.6):
        xyxys =[]
        confidence = []
        class_ids = []

        boxes = results[0].boxes
        class_array = boxes.cls.cpu().numpy().astype(int)
        conf_array = boxes.conf.cpu().numpy()
        xyxy_array = boxes.xyxy.cpu().numpy()

        for class_id, conf, xyxy in zip(class_array, conf_array, xyxy_array):
            if conf < conf_threshold:
                continue
            xyxys.append(xyxy)
            confidence.append(conf)
            class_ids.append(class_id)

        for bbox, conf, class_id in zip(xyxys, confidence, class_ids):
            x1, y1, x2, y2 = bbox
            label = self.CLASS_NAMES_DICT.get(class_id, str(class_id))

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

            # Draw label with confidence
            cv2.putText(frame, f"{label}, conf: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        # set the resolution for the frame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            results = self.model(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            if end_time - start_time != 0:
                fps = 1/np.round(end_time - start_time, 2)
            # print(fps)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
# detector = ObjectDetection(capture_index=0)
detector = ObjectDetection("video/1.avi")

detector()
