from ultralytics import YOLO

model = YOLO('custom_train_yolov10s_2.pt')
model.export(format='onnx', dynamic=True)

# import torch

# model = torch.load('best.pt')['model'].float().eval()
# dummy_input = torch.zeros((1, 3, 640, 640))  # Make sure this matches training input
# torch.onnx.export(model, dummy_input, "best.onnx",
#                   input_names=["images"], output_names=["output"],
#                   opset_version=12, export_params=True, do_constant_folding=True)
