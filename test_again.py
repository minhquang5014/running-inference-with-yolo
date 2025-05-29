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
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([12, 255, 255])
        self.lower_green = np.array([30, 100, 100])
        self.upper_green = np.array([92, 255, 255])
        self.lower_blue = np.array([95, 120, 120])
        self.upper_blue = np.array([130, 255, 255])

        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"
        print(f"Device used: {self.device}")

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names 
        self.box_annotator = BoxAnnotator(color=ColorPalette(colors=colors), thickness=3)
        
    def load_model(self):
        model = YOLO("custom_train_yolov10s_2.pt")
        model.fuse()
        return model

    def plot_boxes(self, results, frame, conf_threshold = 0.6):
        xyxys = []
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
        boxes_out = []
        for bbox, conf, class_id in zip(xyxys, confidence, class_ids):
            x1, y1, x2, y2 = bbox.astype(int)
            label = self.CLASS_NAMES_DICT.get(class_id, str(class_id))

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

            # Draw label with confidence
            cv2.putText(frame, f"{label}, conf: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            boxes_out.append(((x1, y1, x2, y2), class_id))
        return frame, boxes_out

    def video(self):
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

            # return the frame and the 4 coordinates
            frame, boxes = self.plot_boxes(results, frame)
            print(boxes)

            # take the region inside the 4 coordinates as ROI
            # extract the ROI by cutting the frame
            if len(boxes) != 0:
                for (x1, y1, x2, y2), class_id in boxes:
                    # if class_id == 0:
                    #     continue
                    w1 = x2 - x1
                    h1 = y2 - y1
                    ROI = frame[y1:y2, x1:x2]

                    # convert the ROI to HSV color format
                    hsv_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

                    masked_red = cv2.inRange(hsv_roi, self.lower_red, self.upper_red)
                    masked_blue = cv2.inRange(hsv_roi, self.lower_blue, self.upper_blue)
                    masked_green = cv2.inRange(hsv_roi, self.lower_green, self.upper_green)

                    contours_red, _ = cv2.findContours(masked_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_blue, _ = cv2.findContours(masked_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_green, _ = cv2.findContours(masked_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours_red:
                        if cv2.contourArea(contour) > 1/4 * w1 * h1:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                            cv2.putText(frame, "red", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    for contour in contours_blue:
                        if cv2.contourArea(contour) > 1/4 * w1 * h1:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                            cv2.putText(frame, "blue", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    for contour in contours_green:
                        if cv2.contourArea(contour) > 1/4 * w1 * h1:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                            cv2.putText(frame, "green", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check dominant color
                # red_pixels = np.sum(masked_red > 0)
                # blue_pixels = np.sum(masked_blue > 0)
                # green_pixels = np.sum(masked_green > 0)

                # max_color = max(red_pixels, green_pixels, blue_pixels)
                # if max_color == red_pixels:
                #     detected_color = "red"
                # elif max_color == green_pixels:
                #     detected_color = "green"
                # elif max_color == blue_pixels:
                #     detected_color = "blue"
                # else:
                #     detected_color = None

                # cv2.putText(frame, f'{detected_color}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

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

    def image(self, image):
        img = cv2.imread(image)
        img = cv2.resize(img, (640, 480))
        results = self.model(img)

        # return the frame and the 4 coordinates
        frame, boxes = self.plot_boxes(results, img)

        # take the region inside the 4 coordinates as ROI
        # extract the ROI by cutting the frame
        if len(boxes) != 0:
            for (x1, y1, x2, y2), class_id in boxes:
                # if class_id == 0:
                #     continue
                w1 = x2 - x1
                h1 = y2 - y1
                ROI = frame[y1:y2, x1:x2]

                # convert the ROI to HSV color format
                hsv_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

                masked_red = cv2.inRange(hsv_roi, self.lower_red, self.upper_red)
                masked_blue = cv2.inRange(hsv_roi, self.lower_blue, self.upper_blue)
                masked_green = cv2.inRange(hsv_roi, self.lower_green, self.upper_green)

                contours_red, _ = cv2.findContours(masked_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_blue, _ = cv2.findContours(masked_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, _ = cv2.findContours(masked_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detection_lst = []
                for contour in contours_red:
                    if cv2.contourArea(contour) > 1/4 * w1 * h1:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                        cv2.putText(frame, "red", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                for contour in contours_blue:
                    if cv2.contourArea(contour) > 1/4 * w1 * h1:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                        cv2.putText(frame, "blue", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                for contour in contours_green:
                    if cv2.contourArea(contour) > 1/4 * w1 * h1:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (76, 153, 0), 3)  # Draw rectangle
                        cv2.putText(frame, "green", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imshow("Yolov8 Detection", img)
        cv2.imshow("HSV color", hsv_roi)
        cv2.waitKey(0)

# detector = ObjectDetection(capture_index=0)
detector = ObjectDetection("video/3.avi")
detector.video()
# detector.image("images/11.jpg")
