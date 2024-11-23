import torch
import numpy as np
import cv2
import time

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"device used: {self.device}")

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)
        return model
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    def class_to_label(self, x):
        return self.classes[int(x)]
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
    
    def __call__(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 225), 2)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Create a new object and execute.
detection = ObjectDetection()
detection()
