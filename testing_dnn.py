import cv2
import numpy as np
import yaml

with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

CLASSES = data["names"]

# configuration
MODEL_PATH = "custom_train_yolov10s.onnx"
VIDEO_INPUT = (640, 640)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# load the model 
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print(dir(net))
# set up device used - cuda or cpu

# run inference on a real video