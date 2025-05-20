import onnxruntime as ort
import yaml
import cv2
import numpy as np
from time import time
# with open("data.yaml", "r") as f:
#     data = yaml.safe_load(f)

# CLASSES = data['names']   # ['NG', 'black', 'white']
CLASSES = ['NG', 'black', 'white']
INPUT_SIZE = (640, 640)
CONF_THRESH = 0.5
IOU_THRESH = 0.45

ONNX_MODEL_PATH = "custom_train_yolov10s.onnx"

session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


input_name = session.get_inputs()[0].name
# output_shape = session.get_outputs()[0].shape
# print(f"ONNX model output shape: {output_shape}")
def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch
    return img.astype(np.float32)

def run_video():
    cap = cv2.VideoCapture("video/1.avi")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[0], frame.shape[1]
        start_time = time()
        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        # (7, 8400) outputs 8400 predictions and each has 7 values - yolov8n model
        # for the yolov10s, it gives out the output (6, 300), no need to transpose the matrix
        preds = outputs[0][0] # 6 which includes (class_id, conf_score, coordinates)
        boxes = []
        for pred in preds:
            class_id = int(pred[-1])
            score = float(pred[-2])
            x1, y1, x2, y2= map(int, pred[:4])

            if score < CONF_THRESH:
                continue
            print(class_id, score)
            
            # calculate the exact coordinate
            x1 = x1 / INPUT_SIZE[0] * width
            y1 = y1 / INPUT_SIZE[1] * height
            x2 = x2 / INPUT_SIZE[0] * width
            y2 = y2 / INPUT_SIZE[1] * height
        
            # print(f"`conf score {score}, coordinate {x1, y1, x2, y2}")
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))
        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box
            print(cls_id)
            label = f"{CLASSES[cls_id]} {score:.2f}"
            print(f"total label {label}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        end_time = time()
        if end_time - start_time != 0:
            fps = 1/ (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 ONNX Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_image():
    img = cv2.imread("3.jpg")
    input_tensor = preprocess(img)
    outputs = session.run(None, {input_name: input_tensor})
    preds = outputs[0][0] # 6 which includes (class_id, conf_score, coordinates)
    boxes = []
    for pred in preds:
        class_id = int(pred[-1])
        score = float(pred[-2])
        print(score)
        x1, y1, x2, y2 = map(int, pred[:4])
        if score < CONF_THRESH:
            continue
        # print(f"`conf score {score}, coordinate {x1, y1, x2, y2}")
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))
    for box in boxes:
        x1, y1, x2, y2, score, cls_id = box
        print(cls_id, score)
        label = f"{CLASSES[cls_id] if cls_id < len(CLASSES) else 'unknown'} {score:.2f}"
        print(f"{label}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 ONNX Inference", img)
    cv2.waitKey(0)
    
run_video()