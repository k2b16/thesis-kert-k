using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;

public class SentisSsdRunner : MonoBehaviour {
    [Header("model")]
    public ModelAsset modelAsset;
    public string scoresOutputName = "scores";
    public string boxesOutputName = "boxes";

    [Header("labels")]
    public TextAsset labelsTxt;
    string[] _labels;

    [Header("scene refs")]
    public RawImage video;
    public DetectionOverlay overlay;

    [Header("inference")]
    public BackendType backend = BackendType.CPU;
    [Range(0.05f, 0.5f)] public float intervalSec = 0.2f;
    [Range(0f, 1f)] public float scoreThreshold = 0.45f;

    [Header("NMS")]
    [Range(0.1f, 0.9f)] public float iouThreshold = 0.45f;
    public int candidateSize = 200;
    public int maxDetections = 20;

    [Header("SSD Input")]
    public int inputSize = 300;
    public CoordOrigin coordOrigin = CoordOrigin.TopLeft;

    [Header("HoloLens 2")]
    public bool hideVideoOnDevice = true;
    public float cameraReadyTimeout = 15f;

    Worker _worker;
    WebCamTexture _cam;
    Tensor<float> _input;
    TextureTransform _tx;

    bool _inferenceRunning = false;
    readonly List<DetectionOverlay.Detection> _final = new();
    readonly List<DetectionOverlay.Detection> _pendingFinal = new();
    readonly Dictionary<int, List<DetectionOverlay.Detection>> _perClass = new();

    const float CenterVariance = 0.1f;
    const float SizeVariance = 0.2f;

    readonly List<Vector4> _priors = new();
    bool _priorsReady = false;

    struct Spec {
        public int fmap, shrink;
        public float boxMin, boxMax;
        public float[] ratios;
        public Spec(int f, int s, float mn, float mx, float[] r) { fmap = f; shrink = s; boxMin = mn; boxMax = mx; ratios = r; }
    }

    static readonly Spec[] Specs = {
        new Spec(19, 16,  60, 105, new float[]{2,3}),
        new Spec(10, 32, 105, 150, new float[]{2,3}),
        new Spec( 5, 64, 150, 195, new float[]{2,3}),
        new Spec( 3,100, 195, 240, new float[]{2,3}),
        new Spec( 2,150, 240, 285, new float[]{2,3}),
        new Spec( 1,300, 285, 330, new float[]{2,3}),
    };

    IEnumerator Start() {
        if (labelsTxt != null)
            _labels = labelsTxt.text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);

#if WINDOWS_UWP
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam)) {
            Debug.LogError("cam permission denied.");
            enabled = false; yield break;
        }
#endif
        yield return null;

        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0) {
            Debug.LogError("no webcam");
            enabled = false; yield break;
        }
        string chosen = devices[0].name;
        foreach (var d in devices) if (!d.isFrontFacing) { chosen = d.name; break; }

        Debug.Log($"came: '{chosen}'");
        _cam = new WebCamTexture(chosen, 896, 504, 30);
        _cam.Play();

        if (video != null) {
#if WINDOWS_UWP
            if (hideVideoOnDevice) video.gameObject.SetActive(false);
            else { video.color = Color.white; video.material = null; }
#else
            video.color = Color.white; video.material = null;
#endif
        }

        float t0 = Time.time;
        while (true) {
            if (_cam == null) break;
            if (_cam.isPlaying && _cam.width > 100 && _cam.didUpdateThisFrame) { Debug.Log($"cam ready: {_cam.width}x{_cam.height}"); break; }
            if (Time.frameCount % 30 == 0) Debug.Log($"wait {_cam.isPlaying} size={_cam.width}x{_cam.height}");
            yield return null;
        }

        if (video != null && video.gameObject.activeSelf) {
            video.texture = _cam;
            video.rectTransform.localEulerAngles = new Vector3(0, 0, -_cam.videoRotationAngle);
            video.uvRect = _cam.videoVerticallyMirrored ? new Rect(0, 1, 1, -1) : new Rect(0, 0, 1, 1);
        }

        if (modelAsset == null) {
            Debug.LogError("model not assigned.");
            enabled = false; yield break;
        }

        var model = ModelLoader.Load(modelAsset);
        foreach (var o in model.outputs) Debug.Log($"name='{o.name}'");

        _worker = new Worker(model, backend);
        _input = new Tensor<float>(new TensorShape(1, 3, inputSize, inputSize));
        _tx = new TextureTransform()
            .SetDimensions(inputSize, inputSize, 3)
            .SetTensorLayout(TensorLayout.NCHW)
            .SetCoordOrigin(coordOrigin);

        BuildPriors();
        Debug.Log("[SSD] Ready.");
    }

    void Update() {
        if (_cam == null || _cam.width <= 100) return;
        if (!_cam.isPlaying) return;
        if (_pendingFinal.Count > 0 && overlay != null) {
            _final.Clear();
            _final.AddRange(_pendingFinal);
            _pendingFinal.Clear();
            overlay.Render(_final, scoreThreshold);
        }

        if (!_inferenceRunning && _cam.didUpdateThisFrame) StartCoroutine(InferenceCoroutine());
    }

    IEnumerator InferenceCoroutine() {
        _inferenceRunning = true;

        if (_worker == null || overlay == null) { _inferenceRunning = false; yield break; }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        TextureConverter.ToTensor(_cam, _input, _tx);
        _worker.Schedule(_input);
        yield return null;
        var scoresT = _worker.PeekOutput(scoresOutputName) as Tensor<float>;
        var boxesT = _worker.PeekOutput(boxesOutputName) as Tensor<float>;

        if (scoresT == null || boxesT == null) {
            Debug.LogError($"scores='{scoresOutputName}' boxes='{boxesOutputName}'");
            _inferenceRunning = false; yield break;
        }

        Tensor<float> scoresCPU = scoresT;
        Tensor<float> boxesCPU = boxesT;

        if (backend != BackendType.CPU) {
            scoresCPU = scoresT.ReadbackAndClone();
            boxesCPU = boxesT.ReadbackAndClone();
        }

        float[] scores = scoresCPU.DownloadToArray();
        float[] boxes = boxesCPU.DownloadToArray();
        sw.Stop();
        float inferenceMs = (float)sw.Elapsed.TotalMilliseconds;

        if (backend != BackendType.CPU) {
            scoresCPU.Dispose();
            boxesCPU.Dispose();
        }
        DecodeAndNms(scoresT.shape, scores, boxesT.shape, boxes);

        _pendingFinal.Clear();
        _pendingFinal.AddRange(_final);
        BenchmarkLogger.Instance?.LogFrame(inferenceMs, _final.Count);
        if (intervalSec > 0) yield return new WaitForSeconds(intervalSec);
        _inferenceRunning = false;
    }

    void BuildPriors() {
        _priors.Clear();
        foreach (var spec in Specs) {
            float scale = (float)inputSize / spec.shrink;
            for (int j = 0; j < spec.fmap; j++)
                for (int i = 0; i < spec.fmap; i++) {
                    float cx = (i + 0.5f) / scale;
                    float cy = (j + 0.5f) / scale;

                    float s = spec.boxMin;
                    _priors.Add(new Vector4(cx, cy, s / inputSize, s / inputSize));
                    s = Mathf.Sqrt(spec.boxMin * spec.boxMax);
                    _priors.Add(new Vector4(cx, cy, s / inputSize, s / inputSize));

                    s = spec.boxMin;
                    foreach (var ar in spec.ratios) {
                        float r = Mathf.Sqrt(ar);
                        _priors.Add(new Vector4(cx, cy, (s / inputSize) * r, (s / inputSize) / r));
                        _priors.Add(new Vector4(cx, cy, (s / inputSize) / r, (s / inputSize) * r));
                    }
                }
        }
        for (int k = 0; k < _priors.Count; k++) {
            var p = _priors[k];
            p.x = Mathf.Clamp01(p.x); p.y = Mathf.Clamp01(p.y);
            p.z = Mathf.Clamp01(p.z); p.w = Mathf.Clamp01(p.w);
            _priors[k] = p;
        }
        _priorsReady = true;
        Debug.Log($"{_priors.Count} priors.");
    }

    void DecodeAndNms(TensorShape scoresShape, float[] scores, TensorShape boxesShape, float[] boxes) {
        int numPriors = scoresShape[1];
        int numClasses = scoresShape[2];

        _final.Clear();
        _perClass.Clear();

        if (!_priorsReady) BuildPriors();
        for (int i = 0; i < numPriors; i++) {
            int bestC = -1;
            float bestS = 0f;
            int sBase = i * numClasses;
            for (int c = 1; c < numClasses; c++) {
                float s = scores[sBase + c];
                if (s > bestS) { bestS = s; bestC = c; }
            }
            if (bestC < 0 || bestS < scoreThreshold) continue;

            int bBase = i * 4;
            var pr = _priors[i];
            float cx = boxes[bBase + 0] * CenterVariance * pr.z + pr.x;
            float cy = boxes[bBase + 1] * CenterVariance * pr.w + pr.y;
            float w = Mathf.Exp(boxes[bBase + 2] * SizeVariance) * pr.z;
            float h = Mathf.Exp(boxes[bBase + 3] * SizeVariance) * pr.w;

            float xmin = Mathf.Clamp01(cx - w * 0.5f);
            float ymin = Mathf.Clamp01(cy - h * 0.5f);
            float xmax = Mathf.Clamp01(cx + w * 0.5f);
            float ymax = Mathf.Clamp01(cy + h * 0.5f);
            if (xmax <= xmin || ymax <= ymin) continue;

            var d = new DetectionOverlay.Detection {
                left = xmin,
                top = ymin,
                right = xmax,
                bottom = ymax,
                score = bestS,
                classId = bestC,
                label = (_labels != null && bestC < _labels.Length) ? _labels[bestC] : bestC.ToString()
            };
            if (!_perClass.TryGetValue(bestC, out var list)) {
                list = new List<DetectionOverlay.Detection>();
                _perClass[bestC] = list;
            }
            list.Add(d);
        }

        foreach (var kv in _perClass) _final.AddRange(HardNms(kv.Value, iouThreshold, candidateSize));
        _final.Sort((a, b) => b.score.CompareTo(a.score));
        if (_final.Count > maxDetections) _final.RemoveRange(maxDetections, _final.Count - maxDetections);
    }

    List<DetectionOverlay.Detection> HardNms(List<DetectionOverlay.Detection> dets, float iouThr, int candSize) {
        dets.Sort((a, b) => b.score.CompareTo(a.score));
        if (dets.Count > candSize) dets = dets.GetRange(0, candSize);
        var kept = new List<DetectionOverlay.Detection>();
        for (int i = 0; i < dets.Count; i++) {
            bool keep = true;
            for (int k = 0; k < kept.Count; k++) if (IoU(dets[i], kept[k]) > iouThr) {keep = false; break; }
            if (keep) kept.Add(dets[i]);
        }
        return kept;
    }

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

    void OnDestroy() {
        _worker?.Dispose();
        _input?.Dispose();
        if (_cam != null && _cam.isPlaying) _cam.Stop();
    }
}