"""
Export Ultralytics YOLO checkpoints to fixed-size ONNX for Unity Sentis.

Examples:
  python export_yolo_onnx.py
  python export_yolo_onnx.py --sizes 320x256 256x192
  python export_yolo_onnx.py --width 256 --height 192 --models yolov10n.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov10n.pt"]
DEFAULT_SIZES = [(320, 256), (256, 192)]
OPSET = 12
SIMPLIFY = True


def parse_sizes(raw_sizes: list[str] | None) -> list[tuple[int, int]]:
    if not raw_sizes:
        return DEFAULT_SIZES

    sizes: list[tuple[int, int]] = []
    for item in raw_sizes:
        if "x" not in item:
            raise ValueError(f"Invalid size '{item}'. Use WIDTHxHEIGHT, e.g. 256x192.")
        width_str, height_str = item.lower().split("x", 1)
        width, height = int(width_str), int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid size '{item}'. Width and height must be positive.")
        sizes.append((width, height))
    return sizes


def export_model(model_path: str, width: int, height: int, output_dir: Path) -> Path:
    model = YOLO(model_path)
    stem = Path(model_path).stem
    output_name = output_dir / f"{stem}_{width}x{height}.onnx"

    model.export(
        format="onnx",
        imgsz=[height, width],  # Ultralytics expects [H, W]
        opset=OPSET,
        simplify=SIMPLIFY,
        dynamic=False,
        half=False,
    )

    default_out = Path(model_path).with_suffix(".onnx")
    if default_out.exists() and default_out.resolve() != output_name.resolve():
        if output_name.exists():
            output_name.unlink()
        default_out.rename(output_name)

    if not output_name.exists():
        raise FileNotFoundError(f"Export failed for {model_path} at {width}x{height}.")

    print(f"saved: {output_name}")
    return output_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO models to fixed-size ONNX.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Checkpoint paths, e.g. yolov10n.pt",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Export sizes as WIDTHxHEIGHT. Default: 320x256 256x192",
    )
    parser.add_argument("--width", type=int, help="Single export width")
    parser.add_argument("--height", type=int, help="Single export height")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for exported ONNX files",
    )
    args = parser.parse_args()

    if (args.width is None) ^ (args.height is None):
        parser.error("Provide both --width and --height, or neither.")

    if args.width is not None and args.height is not None:
        sizes = [(args.width, args.height)]
    else:
        sizes = parse_sizes(args.sizes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"skip: checkpoint not found -> {model_path}")
            continue
        for width, height in sizes:
            export_model(model_path, width, height, output_dir)


if __name__ == "__main__":
    main()
