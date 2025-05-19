import cv2
import numpy as np
import onnxruntime as ort

# --- CONFIGURATION ---
onnx_path = "best.onnx"  # Path to your exported YOLO ONNX model
class_names = ['NG', 'black', 'white']  # Replace with your classes

# Image input size used during training/export
input_width, input_height = 640, 640

# Confidence threshold and NMS threshold
conf_threshold = 0.4
nms_threshold = 0.5

# --- LOAD ONNX MODEL ---
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# --- PREPROCESSING FUNCTION ---
def preprocess(frame):
    image = cv2.resize(frame, (input_width, input_height))
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Shape: [1, 3, H, W]
    return image

# --- POSTPROCESSING FUNCTION ---
def postprocess(output, orig_shape):
    boxes, scores, class_ids = [], [], []
    rows = output.shape[1]

    for i in range(rows):
        row = output[0][i]
        confidence = row[4]
        if confidence >= conf_threshold:
            class_score = row[5:]
            class_id = np.argmax(class_score)
            score = class_score[class_id]
            if score > conf_threshold:
                x_center, y_center, w, h = row[0:4]
                x = int((x_center - w / 2) * orig_shape[1])
                y = int((y_center - h / 2) * orig_shape[0])
                w = int(w * orig_shape[1])
                h = int(h * orig_shape[0])
                boxes.append([x, y, w, h])
                scores.append(float(score))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    results = []
    for i in indices.flatten():
        results.append((boxes[i], class_ids[i], scores[i]))
    return results

# --- MAIN LOOP ---
cap = cv2.VideoCapture("video/video_07332025_11h33m31s.avi")  # Use 0 for webcam, or replace with video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess(frame)
    outputs = session.run(None, {input_name: input_image})
    detections = postprocess(outputs[0], frame.shape[:2])

    for box, class_id, score in detections:
        x, y, w, h = box
        label = f"{class_names[class_id]}: {score:.2f}" if 0 <= class_id < len(class_names) else print(f"Invalid class id: {class_id}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO ONNX Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
