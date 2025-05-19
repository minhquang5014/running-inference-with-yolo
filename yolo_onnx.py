import onnxruntime as ort
import numpy as np
import cv2
import time
import yaml
# === Config ===
ONNX_MODEL_PATH = "best.onnx"
CLASSES = ['NG', 'black', 'White']

INPUT_SIZE = (640, 640)
CONF_THRESH = 0.5
IOU_THRESH = 0.45

# === Setup ONNX Runtime session ===
session = ort.InferenceSession(ONNX_MODEL_PATH)

input_name = session.get_inputs()[0].name

# === Preprocessing ===
def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch
    return img

# === Post-processing ===
def xywh2xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def postprocess(prediction, conf_thres=CONF_THRESH):
    boxes = []
    for det in prediction:
        for *xywh, conf, cls in det:
            if conf > conf_thres:
                boxes.append((xywh2xyxy(xywh), float(conf), int(cls)))
    return boxes

# === Real-time Inference ===
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video/1.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})

    preds = outputs[0][0]
    boxes = []

    for pred in preds:
        conf = pred[4]
        if conf < CONF_THRESH:
            continue
        class_scores = pred[5:]
        cls_id = np.argmax(class_scores)
        score = class_scores[cls_id]
        if score * conf < CONF_THRESH:
            continue

        cx, cy, w, h = pred[0:4]
        x1, y1, x2, y2 = xywh2xyxy([cx, cy, w, h])
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(cls_id)))

    for box in boxes:
        x1, y1, x2, y2, score, cls_id = box
        print(cls_id)
        try:
            label = f"{CLASSES[cls_id]} {score:.2f}"
        except:
            label = None
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
