import cv2
import numpy as np
import yaml

# === CONFIGURATION ===
model_path = "yolov5n.onnx"      # Path to your ONNX model
yaml_path = "data.yaml"          # Path to your YAML file with class names
input_size = 640                 # Input size (YOLOv5/YOLOv8 typically uses 640)
conf_threshold = 0.25
nms_threshold = 0.45

# === LOAD CLASS NAMES FROM YAML ===
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
    class_names = data['names']

# === LOAD MODEL ===
net = cv2.dnn.readNetFromONNX(model_path)
# === SET BACKEND AND TARGET ===
use_cuda = False  # Set to False to use CPU

if use_cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Or DNN_TARGET_CUDA for full precision
    print("[INFO] Using CUDA for inference.")
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] Using CPU for inference.")
# === START VIDEO CAPTURE (webcam) ===
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_height, orig_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0] 
    print(outputs.shape)

    boxes = []
    confidences = []
    class_ids = []

    x_factor = orig_width / input_size
    y_factor = orig_height / input_size

    for output in outputs:
        obj_conf = output[4]
        if obj_conf > conf_threshold:
            class_scores = output[5:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id] * obj_conf

            if conf > conf_threshold:
                cx, cy, w, h = output[0:4]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO ONNX Live Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEAN UP ===
cap.release()
cv2.destroyAllWindows()

# 