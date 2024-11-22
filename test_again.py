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
        model = YOLO("yolov10s.pt")
        model.fuse()
        return model
    def predict(self, frame):
        return self.model(frame)
    
    def plot_boxes(self, results, frame):
        xyxys = []
        confidence = []
        class_ids = []

        for result in results[0]:

            class_id = result.boxes.cls.cpu().numpy().astype(int)
            if class_id == 0:
                # get the coordinate
                xyxys.append(result.boxes.xyxy.cpu().numpy())

                # get the confidence score
                confidence.append(result.boxes.conf.cpu().numpy())

                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                
        detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]}"]
            
            #crappu code by the way
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # print(x1, y1, x2, y2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

            # frame = self.box_annotator.annotate(scene = frame, detections=detections)
            cv2.putText(frame, f"{self.labels[0]}, conf: {confidence:0.2f}", (int(x1), int(y1-20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
            results = self.predict(frame)
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
detector = ObjectDetection("C:/Users/acer/Downloads/traffic_video.mp4")

detector()
