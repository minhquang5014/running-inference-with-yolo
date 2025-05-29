import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load TensorRT engine
def load_engine(trt_file_path):
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers for input/output
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings, stream

# Preprocess image (adapt to your model's input size, likely 640x640 for YOLOv10s)
def preprocess_image(image_path, input_shape=(640, 640)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, input_shape)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # CHW
    image_expanded = np.expand_dims(image_transposed, axis=0)     # NCHW
    return image, np.ascontiguousarray(image_expanded)

# Postprocess outputs (this is basic; update according to your model's output)
def postprocess(output, original_img, conf_threshold=0.4):
    detections = output.reshape(-1, 6)  # Assuming format: x1, y1, x2, y2, conf, class_id
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        if conf > conf_threshold:
            cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(original_img, f"ID:{int(class_id)} {conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return original_img

# Run inference
def infer(engine, image_path):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    orig_image, input_image = preprocess_image(image_path)
    np.copyto(inputs[0]['host'], input_image.ravel())

    # Transfer to device
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer back to host
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    # Wait for completion
    stream.synchronize()

    output = outputs[0]['host']
    result_image = postprocess(output, orig_image)
    return result_image

# Main
if __name__ == '__main__':
    engine = load_engine("yolov10s_model.trt")
    image_path = "images/11.jpg"  # Replace with your image path
    result_img = infer(engine, image_path)
    
    cv2.imshow("Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
