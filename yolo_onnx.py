import onnxruntime as ort
import cv2
import numpy as np
from time import time

# === Config ===
ONNX_MODEL_PATH = "custom_train_yolov10s_2.onnx"
CLASSES = ['NG', 'black','object', 'white']
INPUT_SIZE = (640, 640)
CONF_THRESH = 0.5

# === Letterbox Resize ===
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, dw, dh

# === Preprocess ===
def preprocess(image):
    img, ratio, dw, dh = letterbox(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch
    return img.astype(np.float32), ratio, dw, dh

# === ONNX Runtime Setup ===
session_options = ort.SessionOptions()
session_options.log_severity_level = 1
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    sess_options=session_options,
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)

input_name = session.get_inputs()[0].name

# === Inference ===
def run_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time()
        input_tensor, ratio, dw, dh = preprocess(frame)

        outputs = session.run(None, {input_name: input_tensor})
        preds = outputs[0][0]  # [x1, y1, x2, y2, conf, cls]

        boxes = []
        for pred in preds:
            score = float(pred[4])
            if score < CONF_THRESH:
                continue
            class_id = int(pred[5])
            x1, y1, x2, y2 = pred[0:4]

            # Reverse letterbox scaling
            x1 = max(0, int((x1 - dw) / ratio))
            y1 = max(0, int((y1 - dh) / ratio))
            x2 = max(0, int((x2 - dw) / ratio))
            y2 = max(0, int((y2 - dh) / ratio))

            boxes.append((x1, y1, x2, y2, score, class_id))

        # Draw detections
        for x1, y1, x2, y2, score, cls_id in boxes:
            label = f"{CLASSES[cls_id]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1 / (time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv10s ONNX Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Run the script ===
if __name__ == "__main__":
    run_video()
