import onnxruntime as ort
import yaml
import cv2
import numpy as np
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

CLASSES = data['names']   # ['NG', 'black', 'white']

INPUT_SIZE = (640, 640)
CONF_THRESH = 0.5
IOU_THRESH = 0.45

ONNX_MODEL_PATH = "best.onnx"

session = ort.InferenceSession(ONNX_MODEL_PATH)

input_name = session.get_inputs()[0].name
output_shape = session.get_outputs()[0].shape
print(f"ONNX model output shape: {output_shape}")
def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch
    return img

def xywh2xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def postprocess(prediction, conf_thres=CONF_THRESH):
    boxes = []
    for xywh, conf, cls in prediction:
        if conf > conf_thres:
            boxes.append((xywh2xyxy(xywh), float(conf), int(cls)))
    return boxes

cap = cv2.VideoCapture("video/1.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    preds = outputs[0][0].T # (7, 8400) outputs 8400 predictions and each has 7 values

    for pred in preds:
        conf = pred[4]
        if conf < CONF_THRESH:
            continue
        class_scores = pred[5:]
        cls_id = np.argmax(class_scores)

    cv2.imshow("YOLOv8 ONNX Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()