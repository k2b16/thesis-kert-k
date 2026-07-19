### Object detection implementation into Hololens 2 for human-robot cooperation

Using OpenXR 1.14.3 version and MRTK3 Toolkit to make the base of the project. Model will be implemented with Sentis 2 from unity.<br>
**From now on, the work will continue in Setis, as a resreach paper tested between 3 evironments and Sentis (formerly known as Barracuda) came out as the most optimal solution**<br>

| Component | Version |
|---|---|
| Unity | 6000.0.49f1 |
| Unity Sentis | 2.1.3 |
| MRTK3 | latest |
| OpenXR | 1.14.3 |
| Target device | Microsoft HoloLens 2 |
| Build target | Universal Windows Platform (UWP) |
| Inference backend | CPU (GPU backend unavailable/unstable) |

Selected based on Lazar (2021), which compared TensorFlow.js, Unity Barracuda, WinML, and TensorFlow.NET for HoloLens 2 inference. Sentis (formerly Barracuda) was the most optimal solution.

## Models

| Model | Architecture | Training data | Input | Parameters | Labels |
|---|---|---|---|---|---|
| YOLOv8n | Anchor-free | COCO | 320×256, **256×192** | 3.2M | Coco80.txt |
| YOLOv8s | Anchor-free | COCO | 320×256, **256×192** | 11.2M | Coco80.txt |
| YOLOv10n | NMS-free | COCO | 320×256, **256×192** | 2.3M | Coco80.txt |
| Tiny YOLOv2 | Anchor-based | VOC | 416×416 | 15.9M | Voc20.txt |
| SSD MobileNetV1 | Multi-scale | VOC | 300×300 | 5.1M | voc-model-labels.txt |
| SSD MobileNetV2 | Inverted residual | VOC | 300×300 | 4.3M | voc-model-labels.txt |

## Results
 
### HoloLens 2 On-Device Performance
 
| Model | Avg ms | P50 ms | P95 ms | Render FPS | Inf/s | Avg Dets | Memory (MB) |
|---|---|---|---|---|---|---|---|
| **YOLOv10n** | **145.0** | 137.5 | 187.1 | 14.7 | **6.9** | 0.53 | 131.9 |
| **YOLOv8n** | 168.4 | 160.1 | 215.6 | **24.3** | 5.9 | 1.77 | 90.1 |
| SSD-MobileNetV2 | 207.9 | 209.5 | 261.9 | 22.3 | 4.8 | 0.66 | 128.5 |
| SSD-MobileNetV1 | 258.5 | 254.8 | 294.8 | 13.5 | 3.9 | 0.65 | 155.9 |
| YOLOv8s | 450.4 | 445.5 | 511.6 | 7.5 | 2.2 | 1.78 | 110.7 |
| Tiny YOLOv2 | 521.0 | 511.3 | 604.1 | 3.0 | 1.9 | 0.14 | 245.2 |

### Detection Accuracy (COCO eval protocol, 20-image test set)
 
| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| **YOLOv8s** | **0.133** | **0.122** |
| YOLOv10n | 0.108 | 0.097 |
| YOLOv8n | 0.107 | 0.096 |
| SSD-MobileNetV2 | 0.009 | 0.004 |
| SSD-MobileNetV1 | 0.005 | 0.002 |
| Tiny YOLOv2 | 0.000 | 0.000 |

## Key Technical Findings
 
### NHWC vs NCHW Tensor Layout
Unity Sentis `DownloadToArray()` always returns data in **NHWC order** regardless of the shape reported in the ONNX file. This was discovered empirically:

## Test Dataset
 
- **Platform:** Roboflow
- **Images:** 20 (test split from 200-image dataset)
- **Annotations:** 171
- **Categories:** 13 (chair, tv, dining table, keyboard, mouse, laptop, bottle, person, couch, potted plant, backpack, handbag, book)
- **Format:** COCO JSON
- **Annotation tool:** Roboflow manual annotation
**Why VOC models scored near zero:**  
VOC covers only 7 of the 13 test categories. keyboard, mouse, laptop, backpack, handbag, and book are not in VOC — those detections are impossible regardless of model quality.

## Project Structure
 
```
Assets/
├── Scripts/
│   ├── SentisYoloRunner.cs      ← YOLO inference (YOLOv8/v10/TinyYOLOv2)
│   ├── SentisSsdRunner.cs       ← SSD inference (MobileNetV1/V2)
│   ├── BenchmarkLogger.cs       ← Performance logging to CSV
│   ├── DetectionOverlay.cs      ← AR bounding box rendering
│   ├── CanvasFollowCamera.cs    ← Canvas attachment to camera
│   └── ModelDebugger.cs         ← Output tensor inspection
├── Models/                      ← ONNX model files
└── Labels/
    ├── Coco80.txt               ← YOLO COCO models
    ├── Voc20.txt                ← Tiny YOLOv2
    └── voc-model-labels.txt     ← SSD models (21 classes incl. BACKGROUND)
 
Evaluation/
├── evaluate_map.py              ← mAP evaluation script
├── models/                      ← ONNX files for evaluation
├── labels/                      ← Label files
└── test/                        ← Test images + _annotations.coco.json
 
Models/
├── export_yolo_onnx.py          ← YOLOv8/v10 ONNX export (320×256 and 256×192)
├── convert_ssdv2_onnx.py        ← SSD MobileNetV2 conversion
└── requirements.txt
```
 
---

## Model optimization

### Resize (recommended first step)

Smaller inputs reduce HoloLens inference time with modest accuracy loss. Export both default and resized variants:

```bash
cd Models/
pip install ultralytics onnx onnxsim
python export_yolo_onnx.py --sizes 320x256 256x192
```

Copy the new ONNX files into:

- `Evaluation/models/` for mAP evaluation
- `Obj_Detect_Sentis/Assets/Models/` for HoloLens deployment

Runners auto-read input width/height from the loaded model, so you only need to assign the new `ModelAsset`.

Evaluate resized models on PC:

```bash
cd Evaluation/
python evaluate_map.py --test_dir test --models_dir models --labels_dir labels --include-resized
```

### Uint8 / Float16 quantization (Unity Sentis)

Sentis **does not import** externally quantized ONNX (`QuantizeLinear` / `DequantizeLinear` ops). Quantize inside Unity instead:

1. Place ONNX models in `Obj_Detect_Sentis/Assets/Models/`
2. In Unity: select a model asset → **Thesis → Quantize Selected Model Assets → Uint8** (or Float16)
3. Assign the generated `*_uint8.sentis` or `*_fp16.sentis` file to the runner's **Model Asset** field
4. Re-run HoloLens benchmarks and compare latency + detection quality

Uint8 mainly reduces disk/memory footprint; inference speed on CPU may change little. Resizing usually gives a larger latency win on HoloLens.

---
 
## mAP Evaluation
 
```bash
cd Evaluation/
python evaluate_map.py --test_dir test --models_dir models --labels_dir labels
```
 
Results saved to `thesis_results.json`.
 
Evaluation hardware: AMD Ryzen 7 3800X, 32GB RAM, Windows 10, CPU backend via ONNX Runtime.
 
---
 
## Scene Setup
 
```
SampleScene
├── MRTK XR Rig
│   └── Camera Offset
│       └── Main Camera
│           └── CameraPanelAnchor
│               └── Canvas (World Space, 1920×1080, scale 0.001, Z=2)
│                   └── OverlayRoot (DetectionOverlay.cs)
├── Runner_YOLOv8n   (SentisYoloRunner + BenchmarkLogger)
├── Runner_YOLOv8s
├── Runner_YOLOv10n
├── Runner_TinyYOLOv2
├── Runner_SSDv1
└── Runner_SSDv2
```

## References
 
- Lazar (2021) — justification for Unity Sentis over WinML/TensorFlow.NET
- Jocher et al. (2023) — Ultralytics YOLOv8
- Wang et al. (2024) — YOLOv10, arXiv:2405.14458
- Redmon & Farhadi (2017) — YOLO9000/TinyYOLOv2
- Liu et al. (2016) — SSD: Single Shot MultiBox Detector
- Howard et al. (2017) — MobileNets
- Sandler et al. (2018) — MobileNetV2
- Lin et al. (2014) — Microsoft COCO evaluation protocol
- Dangberg (2024) — YOLOv8/v10 on HoloLens 2 with Unity Sentis (reference implementation)
