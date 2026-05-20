import argparse
import json
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MODELS = [
    {
        "name": "YOLOv8n",
        "file": "yolov8n_320x256.onnx",
        "type": "yolov8",
        "width": 320,
        "height": 256,
        "labels": "Coco80.txt",
        "threshold": 0.45,
    },
    {
        "name": "YOLOv8s",
        "file": "yolov8s_320x256.onnx",
        "type": "yolov8",
        "width": 320,
        "height": 256,
        "labels": "Coco80.txt",
        "threshold": 0.45,
    },
    {
        "name": "YOLOv10n",
        "file": "yolov10n_320x256.onnx",
        "type": "yolov10",
        "width": 320,
        "height": 256,
        "labels": "Coco80.txt",
        "threshold": 0.45,
    },
    {
        "name": "TinyYOLOv2",
        "file": "tinyyolov2-8.onnx",
        "type": "tinyyolov2",
        "width": 416,
        "height": 416,
        "labels": "Voc20.txt",
        "threshold": 0.20,
    },
    {
        "name": "SSD-MobileNetV1",
        "file": "mb1-ssd_norm_op15.onnx",
        "type": "ssd",
        "width": 300,
        "height": 300,
        "labels": "voc-model-labels.txt",
        "threshold": 0.45,
    },
    {
        "name": "SSD-MobileNetV2",
        "file": "mb2-ssd-lite_norm_op15.onnx",
        "type": "ssd",
        "width": 300,
        "height": 300,
        "labels": "voc-model-labels.txt",
        "threshold": 0.45,
    },
]

# Tiny YOLOv2 anchors
TINY_V2_ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

# SSD priors
SSD_SPECS = [
    dict(fmap=19, shrink=16,  box_min=60,  box_max=105, ratios=[2,3]),
    dict(fmap=10, shrink=32,  box_min=105, box_max=150, ratios=[2,3]),
    dict(fmap=5,  shrink=64,  box_min=150, box_max=195, ratios=[2,3]),
    dict(fmap=3,  shrink=100, box_min=195, box_max=240, ratios=[2,3]),
    dict(fmap=2,  shrink=150, box_min=240, box_max=285, ratios=[2,3]),
    dict(fmap=1,  shrink=300, box_min=285, box_max=330, ratios=[2,3]),
]

def build_ssd_priors(input_size=300):
    #Generate 3000 SSD priors, Output clipped [0, 1]
    priors = []
    for spec in SSD_SPECS:
        scale = input_size / spec["shrink"]
        for j in range(spec["fmap"]):
            for i in range(spec["fmap"]):
                cx = (i + 0.5) / scale
                cy = (j + 0.5) / scale
                # small square
                s = spec["box_min"]
                priors.append([cx, cy, s/input_size, s/input_size])
                #geo-mean square
                s = np.sqrt(spec["box_min"] * spec["box_max"])
                priors.append([cx, cy, s/input_size, s/input_size])

                #aspect-ratio 
                s = spec["box_min"]
                for r in spec["ratios"]:
                    rr = np.sqrt(r)
                    priors.append([cx, cy, (s/input_size)*rr, (s/input_size)/rr])
                    priors.append([cx, cy, (s/input_size)/rr, (s/input_size)*rr])
    return np.clip(np.array(priors, dtype=np.float32), 0, 1)

def sigmoid(x):
    #prevent overflow
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def nms(boxes, scores, iou_thr=0.45):
    #NMS, pattern from Ross Girshick's Fast R-CNN py_cpu_nms(2015)
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        # IoU of top scoring box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou = inter/(areas[i]+areas[order[1:]]-inter+1e-5)
        # keep those whose IoU below threshold
        order = order[1:][iou<=iou_thr]
    return keep

def preprocess(img_bgr, width, height):
    # ONNX preprocessing, non-letterbox resize, exported with '_norm' suffix
    img = cv2.resize(img_bgr, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[np.newaxis]
    return img

def decode_yolov8(output, width, height, threshold, num_classes=80):
    #YOLOv8 decode
    pred = output[0]  # [84, 8400]
    anchors = pred.shape[1]
    dets = []
    for a in range(anchors):
        #best class for acnhor
        class_scores = pred[4:, a]
        best_c = int(np.argmax(class_scores))
        best_s = float(class_scores[best_c])
        if best_s < threshold: continue
        #box is (cx, cy, w, h)
        cx,cy,bw,bh = pred[0,a],pred[1,a],pred[2,a],pred[3,a]
        x1 = max(0, (cx-bw/2)/width)
        y1 = max(0, (cy-bh/2)/height)
        x2 = min(1, (cx+bw/2)/width)
        y2 = min(1, (cy+bh/2)/height)
        dets.append([x1,y1,x2,y2,best_s,best_c])
    return dets

def decode_yolov10(output, width, height, threshold):
    #emits fixed 300x6 top-K, layout auto-det
    pred = output[0]
    # Auto-detect layout
    if pred.shape[0] == 300 and pred.shape[1] == 6:
        rows = pred
    else:
        rows = pred.T
    dets = []
    for row in rows:
        if len(row) < 6: continue
        x1,y1,x2,y2,score,cls = row[0],row[1],row[2],row[3],row[4],row[5]
        if score < threshold: continue
        dets.append([x1/width, y1/height, x2/width, y2/height, float(score), int(cls)])
    return dets

def decode_tinyyolov2(output, width, height, threshold, num_classes=20):
    #out 13x13 grid x 5 anchors x 25 attrs, ! usses .flatten()
    raw = output.flatten()
    grid = 13; num_anchors = 5; box_attrs = 25
    dets = []
    for ay in range(grid):
        for ax in range(grid):
            for a in range(num_anchors):
                base = (ay*grid+ax)*(num_anchors*box_attrs) + a*box_attrs
                if base+box_attrs > len(raw): continue
                tx,ty,tw,th = raw[base],raw[base+1],raw[base+2],raw[base+3]
                conf = sigmoid(raw[base+4])
                if conf < threshold: continue
                #num stabble softmax
                class_scores = raw[base+5:base+5+num_classes]
                max_s = class_scores.max()
                exp_s = np.exp(class_scores - max_s)
                softmax = exp_s / exp_s.sum()
                best_c = int(np.argmax(softmax))
                best_s = float(softmax[best_c] * conf)
                if best_s < threshold: continue
                #box decode
                cx = (ax + sigmoid(tx)) / grid
                cy = (ay + sigmoid(ty)) / grid
                bw = (TINY_V2_ANCHORS[a*2]   * np.exp(tw)) / grid
                bh = (TINY_V2_ANCHORS[a*2+1] * np.exp(th)) / grid
                x1 = max(0, cx-bw/2); y1 = max(0, cy-bh/2)
                x2 = min(1, cx+bw/2); y2 = min(1, cy+bh/2)
                dets.append([x1,y1,x2,y2,best_s,best_c])
    return dets

def decode_ssd(scores_out, boxes_out, threshold, input_size=300):
    #SSD decode, ported from qfgaohao/pytorch-ssd
    priors = build_ssd_priors(input_size)
    scores = scores_out[0]; boxes = boxes_out[0]
    num_priors, num_classes = scores.shape
    dets = []
    for i in range(num_priors):
        #best foreground class
        #class 0 is bgr
        best_c = int(np.argmax(scores[i,1:]))+1
        best_s = float(scores[i,best_c])
        if best_s < threshold: continue
        #variance scaled decode
        dx,dy,dw,dh = boxes[i]
        pr = priors[i]
        cx = dx*0.1*pr[2]+pr[0]; cy = dy*0.1*pr[3]+pr[1]
        w  = np.exp(dw*0.2)*pr[2]; h  = np.exp(dh*0.2)*pr[3]
        x1 = max(0,cx-w/2); y1 = max(0,cy-h/2)
        x2 = min(1,cx+w/2); y2 = min(1,cy+h/2)
        dets.append([x1,y1,x2,y2,best_s,best_c])
    return dets

#iterates over every img
def run_model(model_cfg, test_dir, coco_gt, label_map, models_dir):
    model_path = os.path.join(models_dir, model_cfg["file"])
    if not os.path.exists(model_path):
        print(f"Model file not found")
        return None
    #CPU provider
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    results = []
    img_ids = coco_gt.getImgIds()
    total_time = 0

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(test_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]

        inp = preprocess(img, model_cfg["width"], model_cfg["height"])
        #pure inference time
        t0 = time.time()
        outputs = session.run(output_names, {input_name: inp})
        total_time += time.time() - t0

        mtype = model_cfg["type"]
        threshold = model_cfg["threshold"]

        #dispatch right decoder
        if mtype == "yolov8":
            dets = decode_yolov8(outputs[0], model_cfg["width"], model_cfg["height"], threshold)
        elif mtype == "yolov10":
            dets = decode_yolov10(outputs[0], model_cfg["width"], model_cfg["height"], threshold)
        elif mtype == "tinyyolov2":
            dets = decode_tinyyolov2(outputs[0], model_cfg["width"], model_cfg["height"], threshold)
        elif mtype == "ssd":
            scores_out = next(o for o,n in zip(outputs, output_names) if "score" in n.lower())
            boxes_out  = next(o for o,n in zip(outputs, output_names) if "box"   in n.lower())
            dets = decode_ssd(scores_out, boxes_out, threshold)
        else:
            dets = []

        #per class NMS, except YOLOv10
        if dets and mtype not in ("yolov10",):
            dets_arr = np.array(dets)
            final_dets = []
            for cls in np.unique(dets_arr[:,5].astype(int)):
                mask = dets_arr[:,5].astype(int) == cls
                cls_dets = dets_arr[mask]
                kept = nms(cls_dets[:,:4], cls_dets[:,4])
                final_dets.extend(cls_dets[kept].tolist())
            dets = final_dets
        # convert COCO-format result
        for det in dets:
            x1,y1,x2,y2,score,cls_idx = det
            cls_idx = int(cls_idx)
            if cls_idx not in label_map: continue
            coco_cat_id = label_map[cls_idx]
            #map normalized [0, 1] coords back to org img
            bx = x1 * orig_w
            by = y1 * orig_h
            bw = (x2-x1) * orig_w
            bh = (y2-y1) * orig_h

            results.append({
                "image_id":    img_id,
                "category_id": coco_cat_id,
                "bbox":        [bx, by, bw, bh],
                "score":       float(score),
            })

    avg_ms = (total_time / len(img_ids)) * 1000 if img_ids else 0
    return results, avg_ms
# label mapping
def build_label_map(labels_file, model_type, coco_gt):
    if not os.path.exists(labels_file):
        print("Labels not found")
        return {}

    with open(labels_file) as f:
        labels = [l.strip() for l in f if l.strip()]
    #build name, COCO id lookup
    name_to_coco = {}
    for cat in coco_gt.dataset["categories"]:
        name_to_coco[cat["name"].lower()] = cat["id"]

    label_map = {}
    for idx, name in enumerate(labels):
        name_lower = name.lower()
        if name_lower in ("background", "bg", "__background__"):
            continue
        if name_lower in name_to_coco:
            label_map[idx] = name_to_coco[name_lower]
        #try common aliases
        aliases = {
            "aeroplane": "airplane", "motorbike": "motorcycle",
            "sofa": "couch", "diningtable": "dining table",
            "pottedplant": "potted plant", "tvmonitor": "tv",
        }
        if name_lower in aliases and aliases[name_lower] in name_to_coco:
            label_map[idx] = name_to_coco[aliases[name_lower]]

    return label_map
# pycocotools eval
def evaluate(results, coco_gt):
    if not results:
        return {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0}

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "mAP50_95": float(evaluator.stats[0]),
        "mAP50":    float(evaluator.stats[1]),
        #stats 2 is mAP@0.75
        "precision":float(evaluator.stats[2]),
        "recall":   float(evaluator.stats[8]),
    }

#CLI entry, iterated model registry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",   default="test",   help="Path to test folder with images + _annotations.coco.json")
    parser.add_argument("--models_dir", default="models", help="Path to folder with ONNX model files")
    parser.add_argument("--labels_dir", default=".",      help="Path to folder with label .txt files")
    args = parser.parse_args()

    ann_file = os.path.join(args.test_dir, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        print("Annotation not found")
        return
    coco_gt = COCO(ann_file)

    print("\n" + "="*70)
    print(f"{'Model':<20} {'mAP@0.5':>8} {'mAP@0.5:0.95':>13} {'Avg ms':>8}")
    print("="*70)

    all_results = []

    for model_cfg in MODELS:
        print(f"\nEval: {model_cfg['name']}")

        labels_path = os.path.join(args.labels_dir, model_cfg["labels"])
        label_map = build_label_map(labels_path, model_cfg["type"], coco_gt)

        if not label_map:
            print("No label mapping found")
            continue

        result = run_model(model_cfg, args.test_dir, coco_gt, label_map, args.models_dir)
        if result is None: continue

        detections, avg_ms = result

        metrics = evaluate(detections, coco_gt)

        print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")

        all_results.append({
            "model":      model_cfg["name"],
            "mAP50":      metrics["mAP50"],
            "mAP50_95":   metrics["mAP50_95"],
            "avg_ms":     avg_ms,
        })

    # final summary table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(f"{'Model':<22} {'mAP@0.5':>8} {'mAP@0.5:0.95':>13} {'Inf ms':>8}")
    print("-"*70)
    for r in sorted(all_results, key=lambda x: x["mAP50"], reverse=True):
        print(f"{r['model']:<22} {r['mAP50']:>8.4f} {r['mAP50_95']:>13.4f} {r['avg_ms']:>8.1f}")
    print("="*70)

    # Save results to JSON
    with open("thesis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to thesis_results.json")

if __name__ == "__main__":
    main()
