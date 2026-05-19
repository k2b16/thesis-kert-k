"""
this avoids the TensorFlow Loop operator issue that breaks Unity Sentis.

git clone https://github.com/qfgaohao/pytorch-ssd.git
COCO weights download: https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth

models/mb2-ssd-lite_op12.onnx        (raw)
models/mb2-ssd-lite_norm_op12.onnx   (with baked normalization → use this in Unity)
"""

from pathlib import Path
import torch
import onnx
from onnx import helper, numpy_helper
import numpy as np

try:
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
except ImportError:
    raise ImportError("could not import pytorch-ssd.")

WEIGHTS  = "models/mb2-ssd-lite-mp-0_686.pth"
LABELS   = "models/voc-model-labels.txt" #21 classes + background
RAW_OUT  = "models/mb2-ssd-lite_op12.onnx"
NORM_OUT = "models/mb2-ssd-lite_norm_op12.onnx"
OPSET    = 12 #safest (in range 15-7)

class_names = [l.strip() for l in Path(LABELS).read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"loaded {len(class_names)} classes")

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(WEIGHTS)
net.eval()

dummy = torch.randn(1, 3, 300, 300)
Path(RAW_OUT).parent.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    torch.onnx.export(
        net,
        dummy,
        RAW_OUT,
        input_names=["input"],
        output_names=["scores", "boxes"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"raw ONNX: {RAW_OUT}")

def patch_normalize(in_path, out_path):
    m = onnx.load(in_path)
    g = m.graph
    inp = g.input[0].name

    def const(name, val): return numpy_helper.from_array(np.array(val, dtype=np.float32), name=name)

    g.initializer.extend([const("k255", 255.0),
                          const("k127", 127.0),
                          const("k128", 128.0)
    ])

    n1, n2, n3 = inp + "_mul255", inp + "_sub127", inp + "_div128"
    nodes = [helper.make_node("Mul", [inp, "k255"], [n1], name="norm_mul255"),
             helper.make_node("Sub", [n1, "k127"], [n2], name="norm_sub127"),
             helper.make_node("Div", [n2, "k128"], [n3], name="norm_div128"),
    ]

    for node in g.node: node.input[:] = [n3 if x == inp else x for x in node.input]
    for n in reversed(nodes): g.node.insert(0, n)

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"normalized ONNX: {out_path}")

patch_normalize(RAW_OUT, NORM_OUT)