from ultralytics import YOLO
import os

IMG_WIDTH = 320
IMG_HEIGHT = 256
OPSET = 12
SIMPLIFY = True

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov10n.pt",]

for model_path in MODELS:
    model = YOLO(model_path)
    output_name = model_path.replace(".pt", f"_{IMG_WIDTH}x{IMG_HEIGHT}.onnx")

    model.export(
        format="onnx",
        imgsz=[IMG_HEIGHT, IMG_WIDTH], #[H, W] order
        opset=OPSET,
        simplify=SIMPLIFY,
        dynamic=False, #fixed input shape
        half=False,
    )

    default_out = model_path.replace(".pt", ".onnx")
    if os.path.exists(default_out) and default_out != output_name:
        os.rename(default_out, output_name)
        print(f"saved: {output_name}")
    else:
        print(f"saved: {default_out}")

