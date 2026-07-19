using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;

// each model follows documented output format
//      Tiny YOLOv2 - VOC ancors from darknet (cfg/yolov2-tiny-voc.cfg)
//          decode formulas from YOLO9000 paper
//      YOLOv8 - tensors follow the ultralytics ONNX export formathttps://docs.ultralytics.com.
//      YOLOv10 - tensor follows YOLO10 paper (Wang et al. 2024)
public class SentisYoloRunner : MonoBehaviour {
    public enum YoloVersion { YOLOv8, YOLOv10, TinyYOLOv2 }

    [Header("model")]
    public ModelAsset modelAsset;
    public YoloVersion modelType = YoloVersion.YOLOv8;

    public string outputName = "output0"; //ultralytics default, for YOLOv2 its "grid"

    [Header("labels")]
    public TextAsset labelsTxt; //coco80 for YOLO10/8 and VOC for YOLOv2
    string[] _labels;

    [Header("scene refs")]
    public RawImage video;
    public DetectionOverlay overlay;

    [Header("inference")]
    public BackendType backend = BackendType.CPU; //cpu only
    [Range(0.05f, 0.5f)] public float intervalSec = 0.2f; //5 fps
    [Range(0f, 1f)] public float scoreThreshold = 0.45f;

    [Header("NMS  (YOLOv8 only)")] //YOLOv10 built in NMS
    [Range(0.1f, 0.9f)] public float iouThreshold = 0.45f;
    public int maxDetections = 20;

    [Header("input size")]
    public int inputWidth = 320; //best for Hololens 2
    public int inputHeight = 256;

    [Header("HoloLens 2")]
    public bool hideVideoOnDevice = true;
    public float cameraReadyTimeout = 15f;

    Worker _worker;
    WebCamTexture _cam;
    Tensor<float> _input;
    TextureTransform _tx;
    float _nextTime; //throttle

    readonly List<DetectionOverlay.Detection> _final = new();
    readonly Dictionary<int, List<DetectionOverlay.Detection>> _perClass = new();

    //init
    IEnumerator Start() {
        if (labelsTxt != null) _labels = labelsTxt.text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);

#if WINDOWS_UWP
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);

        if (!Application.HasUserAuthorization(UserAuthorization.WebCam)) {
            Debug.LogError("cam permission denied.");
            enabled = false; yield break;
        }
#endif

        yield return null;
        //pick first non front facing device
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0) {
            Debug.LogError("no cam found.");
            enabled = false; yield break;
        }

        string chosen = devices[0].name;
        foreach (var d in devices) if (!d.isFrontFacing) { chosen = d.name; break; }

        _cam = new WebCamTexture(chosen, 896, 504, 30);
        _cam.Play();
        // hide debug in prev
        if (video != null) {
#if WINDOWS_UWP
            if (hideVideoOnDevice)video.gameObject.SetActive(false);
            else{
                video.color = Color.white;
                video.material = null;
            }
#else
            
            video.color = Color.white;
            video.material = null;
#endif
        }
        // wait for webcam to give frames
        //reference start so doesnt hang
        float t0 = Time.time;
        while (true) {
            if (_cam == null) break;
            if (Time.time - t0 > cameraReadyTimeout) { Debug.LogWarning($"cam timeout {cameraReadyTimeout}."); break; }
            if (_cam.isPlaying && _cam.width > 100 && _cam.didUpdateThisFrame) {Debug.Log($"cam frame received: {_cam.width}x{_cam.height}"); break; }
            if (Time.frameCount % 30 == 0) Debug.Log($"waiting for cam...");
            yield return null;
        }
        // rot/vart flip into preview
        if (video != null && video.gameObject.activeSelf) {
            video.texture = _cam;
            video.rectTransform.localEulerAngles = new Vector3(0, 0, -_cam.videoRotationAngle);
            video.uvRect = _cam.videoVerticallyMirrored ? new Rect(0, 1, 1, -1) : new Rect(0, 0, 1, 1);
        }

        if (modelAsset == null) {
            enabled = false;
            yield break;
        }
        // log models output
        var model = ModelLoader.Load(modelAsset);
        Debug.Log($"model output ({model.outputs.Count})");
        foreach (var o in model.outputs) Debug.Log($"name='{o.name}'");

        ApplyInputShapeFromModel(model);

        //pre allocated tensor
        _worker = new Worker(model, backend);
        _input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
        _tx = new TextureTransform()
            .SetDimensions(inputWidth, inputHeight, 3)
            .SetTensorLayout(TensorLayout.NCHW)
            .SetCoordOrigin(CoordOrigin.TopLeft);
        //defer first inf by 1
        _nextTime = Time.time + 1.0f;
        Debug.Log($"ready w model={modelType} input={inputWidth}x{inputHeight}");
    }

    //per-frame driver
    void Update()
    {
        if (_cam == null || _cam.width <= 100) return;
        //if webcam stop, restart
        if (!_cam.isPlaying) {
            if (Time.frameCount % 300 == 0) { Debug.LogWarning("cam stopped"); _cam.Play(); }
            return;
        }
        if (Time.time < _nextTime) return; //throttle
        if (!_cam.didUpdateThisFrame) return; //skip if no frame
        _nextTime = Time.time + intervalSec;
        RunOnce();
    }
    // single shot inference
    void RunOnce() {
        if (_worker == null || overlay == null) return;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        TextureConverter.ToTensor(_cam, _input, _tx);
        _worker.Schedule(_input);

        var outputT = _worker.PeekOutput(outputName) as Tensor<float>;
        Tensor<float> outputCPU = outputT; //output is alreadi cpu resident
        if (backend != BackendType.CPU) outputCPU = outputT.ReadbackAndClone();

        float[] raw = outputCPU.DownloadToArray();
        sw.Stop();
        float inferenceMs = (float)sw.Elapsed.TotalMilliseconds;

        if (backend != BackendType.CPU) outputCPU.Dispose();

        if (Time.frameCount % 120 == 0) Debug.Log($"inf={inferenceMs:F0}ms, shape={outputT.shape}");

        _final.Clear();

        //dispatch to decoder
        if (modelType == YoloVersion.YOLOv8) DecodeYoloV8(raw, outputT.shape);
        else if (modelType == YoloVersion.YOLOv10) DecodeYoloV10(raw, outputT.shape);
        else DecodeTinyYoloV2(raw, outputT.shape);

        overlay.Render(_final, scoreThreshold);
        BenchmarkLogger.Instance?.LogFrame(inferenceMs, _final.Count);
    }

    //yiny YOLOv2 decode
    // grid: 13x13
    // anchors are grid-cell units
    static readonly float[] TinyV2Anchors = { 1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f };
    const int TinyV2NumAnchors = 5;
    const int TinyV2NumClasses = 20;
    const int TinyV2GridSize = 13;
    const int TinyV2BoxAttrs = 25; //4 boc + 1 obj + 20 classes

    void DecodeTinyYoloV2(float[] raw, TensorShape shape) {
        _perClass.Clear();
        //diag block
        //NCHW and NHWC indexing
        if (Time.frameCount % 120 == 0)
        {
            float globalMax = float.MinValue;
            float globalMin = float.MaxValue;
            int globalMaxIdx = 0;
            for (int i = 0; i < raw.Length; i++) {
                if (raw[i] > globalMax) { globalMax = raw[i]; globalMaxIdx = i; }
                if (raw[i] < globalMin) globalMin = raw[i];
            }
            float globalMaxSigmoid = Sigmoid(globalMax);

            //probe assuming NCHW
            int stride = TinyV2GridSize * TinyV2GridSize;
            float maxConfNCHW = 0f;
            for (int a2 = 0; a2 < TinyV2NumAnchors; a2++)
                for (int g = 0; g < stride; g++) {
                    float c2 = Sigmoid(raw[(a2 * TinyV2BoxAttrs + 4) * stride + g]);
                    if (c2 > maxConfNCHW) maxConfNCHW = c2;
                }
            // probe assuming NHWC
            float maxConfNHWC = 0f;
            for (int ay2 = 0; ay2 < TinyV2GridSize; ay2++)
                for (int ax2 = 0; ax2 < TinyV2GridSize; ax2++)
                    for (int a2 = 0; a2 < TinyV2NumAnchors; a2++) {
                        int bi = (ay2 * TinyV2GridSize + ax2) * (TinyV2NumAnchors * TinyV2BoxAttrs) + a2 * TinyV2BoxAttrs + 4;
                        float c2 = Sigmoid(raw[bi]);
                        if (c2 > maxConfNHWC) maxConfNHWC = c2;
                    }
        }
        //main decode
        //idx assumes NHWC
        for (int ay = 0; ay < TinyV2GridSize; ay++) {
            for (int ax = 0; ax < TinyV2GridSize; ax++) {
                for (int a = 0; a < TinyV2NumAnchors; a++) {
                    int baseIdx = (ay * TinyV2GridSize + ax) * (TinyV2NumAnchors * TinyV2BoxAttrs) + a * TinyV2BoxAttrs;

                    float tx = raw[baseIdx + 0];
                    float ty = raw[baseIdx + 1];
                    float tw = raw[baseIdx + 2];
                    float th = raw[baseIdx + 3];
                    float conf = Sigmoid(raw[baseIdx + 4]);

                    //if raw objectness if below threshold, skip softmax
                    if (conf < scoreThreshold * 0.5f) continue;
                    //stable softmax
                    float[] classScores = new float[TinyV2NumClasses];
                    float maxScore = float.MinValue;
                    for (int c = 0; c < TinyV2NumClasses; c++) {
                        classScores[c] = raw[baseIdx + 5 + c];
                        if (classScores[c] > maxScore) maxScore = classScores[c];
                    }
                    float expSum = 0f;
                    for (int c = 0; c < TinyV2NumClasses; c++) {
                        classScores[c] = Mathf.Exp(classScores[c] - maxScore);
                        expSum += classScores[c];
                    }
                    //final per class score
                    int bestC = -1;
                    float bestS = 0f;
                    for (int c = 0; c < TinyV2NumClasses; c++) {
                        float s = (classScores[c] / expSum) * conf;
                        if (s > bestS) { bestS = s; bestC = c; }
                    }

                    if (bestC < 0 || bestS < scoreThreshold) continue;

                    // cell offset + sigmoid center
                    float cx = (ax + Sigmoid(tx)) / TinyV2GridSize;
                    float cy = (ay + Sigmoid(ty)) / TinyV2GridSize;
                    float bw = (TinyV2Anchors[a * 2] * Mathf.Exp(tw)) / TinyV2GridSize;
                    float bh = (TinyV2Anchors[a * 2 + 1] * Mathf.Exp(th)) / TinyV2GridSize;

                    float xmin = Mathf.Clamp01(cx - bw * 0.5f);
                    float ymin = Mathf.Clamp01(cy - bh * 0.5f);
                    float xmax = Mathf.Clamp01(cx + bw * 0.5f);
                    float ymax = Mathf.Clamp01(cy + bh * 0.5f);

                    if (xmax <= xmin || ymax <= ymin) continue;

                    var d = MakeDetection(xmin, ymin, xmax, ymax, bestS, bestC);
                    if (!_perClass.TryGetValue(bestC, out var list)) {
                        list = new List<DetectionOverlay.Detection>();
                        _perClass[bestC] = list;
                    }
                    list.Add(d);
                }
            }
        }
        // per clas NMS, merge and cap score
        foreach (var kv in _perClass) {
            var kept = HardNms(kv.Value, iouThreshold);
            _final.AddRange(kept);
        }

        _final.Sort((a, b) => b.score.CompareTo(a.score));
        if (_final.Count > maxDetections)
            _final.RemoveRange(maxDetections, _final.Count - maxDetections);
    }

    static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));
    //YOLOv8 decode
    //format follows Ultralytic documented ONNX
    void DecodeYoloV8(float[] raw, TensorShape shape) {
        int rows = shape[1]; // 4 + classes
        int anchors = shape[2]; // total grid cells accros levels
        int numClasses = rows - 4;

        _perClass.Clear();

        for (int a = 0; a < anchors; a++) {
            //best class for anchor
            int bestC = -1;
            float bestS = 0f;
            for (int c = 0; c < numClasses; c++) {
                //transposed layout
                float s = raw[(4 + c) * anchors + a];
                if (s > bestS) { bestS = s; bestC = c; }
            }

            //box values are pixel-space
            if (bestC < 0 || bestS < scoreThreshold) continue;
            float cx = raw[0 * anchors + a];
            float cy = raw[1 * anchors + a];
            float bw = raw[2 * anchors + a];
            float bh = raw[3 * anchors + a];
            float xmin = Mathf.Clamp01((cx - bw * 0.5f) / inputWidth);
            float ymin = Mathf.Clamp01((cy - bh * 0.5f) / inputHeight);
            float xmax = Mathf.Clamp01((cx + bw * 0.5f) / inputWidth);
            float ymax = Mathf.Clamp01((cy + bh * 0.5f) / inputHeight);

            if (xmax <= xmin || ymax <= ymin) continue;

            var d = MakeDetection(xmin, ymin, xmax, ymax, bestS, bestC);

            if (!_perClass.TryGetValue(bestC, out var list)) {
                list = new List<DetectionOverlay.Detection>();
                _perClass[bestC] = list;
            }
            list.Add(d);
        }
        //per class NMS, merge and cap
        foreach (var kv in _perClass) {
            var kept = HardNms(kv.Value, iouThreshold);
            _final.AddRange(kept);
        }

        _final.Sort((a, b) => b.score.CompareTo(a.score));
        if (_final.Count > maxDetections)
            _final.RemoveRange(maxDetections, _final.Count - maxDetections);
    }

    //YOLOv10 decode
    //emits fixed top-K
    void DecodeYoloV10(float[] raw, TensorShape shape)
    {
        const int numDets = 300;
        const int numFields = 6;
        //proble layout
        bool useRowMajor = false;
        for (int i = 0; i < Mathf.Min(10, numDets); i++) {
            float score = raw[i * numFields + 4];
            if (score > 0.01f && score <= 1.0f) { useRowMajor = true; break; }
        }

        if (!useRowMajor) {
            //column-major
            for (int i = 0; i < numDets; i++)
            {
                float x1 = raw[0 * numDets + i];
                float y1 = raw[1 * numDets + i];
                float x2 = raw[2 * numDets + i];
                float y2 = raw[3 * numDets + i];
                float score = raw[4 * numDets + i];
                int classId = Mathf.RoundToInt(raw[5 * numDets + i]);

                if (score < scoreThreshold) continue;
                float xmin = Mathf.Clamp01(x1 / inputWidth);
                float ymin = Mathf.Clamp01(y1 / inputHeight);
                float xmax = Mathf.Clamp01(x2 / inputWidth);
                float ymax = Mathf.Clamp01(y2 / inputHeight);

                if (xmax <= xmin || ymax <= ymin) continue;
                _final.Add(MakeDetection(xmin, ymin, xmax, ymax, score, classId));
            }
        } else {
            //row major
            for (int i = 0; i < numDets; i++) {
                int offset = i * numFields;
                float x1 = raw[offset + 0];
                float y1 = raw[offset + 1];
                float x2 = raw[offset + 2];
                float y2 = raw[offset + 3];
                float score = raw[offset + 4];
                int classId = Mathf.RoundToInt(raw[offset + 5]);

                if (score < scoreThreshold) continue;

                float xmin = Mathf.Clamp01(x1 / inputWidth);
                float ymin = Mathf.Clamp01(y1 / inputHeight);
                float xmax = Mathf.Clamp01(x2 / inputWidth);
                float ymax = Mathf.Clamp01(y2 / inputHeight);

                if (xmax <= xmin || ymax <= ymin) continue;
                _final.Add(MakeDetection(xmin, ymin, xmax, ymax, score, classId));
            }
        }
        // no NMS pass
        _final.Sort((a, b) => b.score.CompareTo(a.score));
        if (_final.Count > maxDetections) _final.RemoveRange(maxDetections, _final.Count - maxDetections);
    }

    void ApplyInputShapeFromModel(Model model) {
        if (model.inputs.Count == 0) return;

        var shape = model.inputs[0].shape;
        if (shape.rank != 4) return;

        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        if (channels == 3 && height > 0 && width > 0) {
            inputHeight = height;
            inputWidth = width;
            Debug.Log($"input size from model: {inputWidth}x{inputHeight}");
        }
    }

    // HELPERS
    DetectionOverlay.Detection MakeDetection(float xmin, float ymin, float xmax, float ymax, float score, int classId) {
        return new DetectionOverlay.Detection {
            left = xmin,
            top = ymin,
            right = xmax,
            bottom = ymax,
            score = score,
            classId = classId,
            label = (_labels != null && classId < _labels.Length) ? _labels[classId] : classId.ToString()
        };
    }
    // NMS, sort by score
    List<DetectionOverlay.Detection> HardNms(List<DetectionOverlay.Detection> dets, float iouThr) {
        dets.Sort((a, b) => b.score.CompareTo(a.score));
        var kept = new List<DetectionOverlay.Detection>();
        for (int i = 0; i < dets.Count; i++) {
            bool keep = true;
            for (int k = 0; k < kept.Count; k++) if (IoU(dets[i], kept[k]) > iouThr) { keep = false; break; }
            if (keep) kept.Add(dets[i]);
        }
        return kept;
    }
    //intersection over union
    //divide by zero to degenarate boxes
    float IoU(DetectionOverlay.Detection a, DetectionOverlay.Detection b) {
        float ix1 = Mathf.Max(a.left, b.left);
        float iy1 = Mathf.Max(a.top, b.top);
        float ix2 = Mathf.Min(a.right, b.right);
        float iy2 = Mathf.Min(a.bottom, b.bottom);
        float inter = Mathf.Max(0f, ix2 - ix1) * Mathf.Max(0f, iy2 - iy1);
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);
        return inter / (areaA + areaB - inter + 1e-5f);
    }
    //release resources on UWP
    void OnDestroy() {
        _worker?.Dispose();
        _input?.Dispose();
        if (_cam != null && _cam.isPlaying) _cam.Stop();
    }
}