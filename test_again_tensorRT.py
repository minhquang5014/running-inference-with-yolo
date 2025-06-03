import torch
import cv2
import random
import time
import pathlib
from ultralytics import YOLO

import modules.utils as utils
# from modules.autobackend import AutoBackend

import yaml

CONF_THRESHOLD = 0.6

def load_class_names_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def get_detection_result(model, frame):
    # Update object localizer
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    result = results[0].cpu()

    # Get information from result
    box = result.boxes.xyxy.numpy()
    conf = result.boxes.conf.numpy()
    cls = result.boxes.cls.numpy().astype(int)

    return cls, conf, box

def detection(model_path, source, name):
    # Check File Extension
    file_extension = pathlib.Path(model_path).suffix

    # Load the Model
    model = YOLO(model_path)

    # Class Name and Colors
    label_map = model.names
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]

    # FPS Detection
    frame_count = 0
    total_fps = 0
    avg_fps = 0

    # FPS Video
    video_cap = cv2.VideoCapture(source)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))

    video_frames = []

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        # # Start Time
        start = time.time()
        # Detection
        cls, conf, box = get_detection_result(model, frame)

        # Pack together for easy use
        detection_output = list(zip(cls, conf, box))
        image_output = utils.draw_box(frame, detection_output, label_map, COLORS)

        end = time.time()
        # # End Time

        # Draw FPS
        frame_count += 1
        fps = 1 / (end - start)
        total_fps = total_fps + fps
        avg_fps = total_fps / frame_count

        image_output = utils.draw_fps(avg_fps, image_output)

        # Append frame to array
        video_frames.append(image_output)

        #
        print("(%2d / %2d) Frames Processed" % (frame_count, total_frames))

    # Get a file name
    file_name = utils.get_name(source)
    # Get Save Path
    folder_name = name
    save_path = utils.get_save_path(file_name, folder_name)
    # Create VideoWriter object.
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), int(avg_fps), (frame_width, frame_height))

    for frame in video_frames:
        out.write(frame)

    out.release()

    print("Video is saved in: "+save_path)

def detection_webcam(capture_index, model_path, yaml_path):
    model = YOLO(model_path)
    print(model)
    label_map = load_class_names_from_yaml(yaml_path)
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]

    cap = cv2.VideoCapture(capture_index)  # 0 = default webcam

    frame_count = 0
    total_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        start = time.time()
        cls, conf, box = get_detection_result(model, frame)
        class_ids = []
        confidence = []
        xyxys = []
        for class_id, conf, xyxy in zip(cls, conf, box):
            if conf < CONF_THRESHOLD:
                continue
            xyxys.append(xyxy)
            confidence.append(conf)
            class_ids.append(class_id)
        detection_output = list(zip(class_ids, confidence, xyxys))
        image_output = utils.draw_box(frame, detection_output, label_map, COLORS)

        end = time.time()
        if (end - start) != 0:
            fps = 1 / (end - start)
        total_fps += fps
        frame_count += 1
        avg_fps = total_fps / frame_count

        image_output = utils.draw_fps(avg_fps, image_output)

        cv2.imshow("YOLOv8 Webcam Detection", image_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detection_webcam(0, "model/custom_train_yolov10s_3.engine", "data.yaml")