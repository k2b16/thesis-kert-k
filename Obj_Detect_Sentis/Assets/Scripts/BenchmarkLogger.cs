using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class BenchmarkLogger : MonoBehaviour {
    public static BenchmarkLogger Instance { get; private set; }

    [Header("benchmark id")]
    public string modelName = "ModelName_Resolution_Backend";

    [Header("time")]
    public float benchmarkDurationSec = 60f;

    [Tooltip("warmup")]
    public float warmupSec = 5f;

    [Header("display")]
    public bool showOverlay = true;

    readonly List<float> _inferenceTimes = new();
    readonly List<int> _detectionCounts = new();
    readonly List<float> _renderFpsSamples = new();
    readonly List<float> _inferenceFpsSamples = new();

    float _sessionStart;
    bool _warmedUp = false;
    bool _running = true;
    bool _saved = false;
    string _logPath;

    int _renderFrameCount;
    int _inferenceFrameCount;
    float _lastFpsWindowStart;

    float _liveInferenceMs;
    float _liveAvgInferenceMs;
    float _liveRenderFps;
    float _liveInferenceFps;
    int _liveDets;
    string _statusMsg = "WARMUP";

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;

        _sessionStart = Time.time;
        _lastFpsWindowStart = Time.time;

        string safeName = modelName.Replace(" ", "_").Replace("/", "-");
        string fileName = $"bench_{safeName}_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
        _logPath = Path.Combine(Application.persistentDataPath, fileName);

        File.WriteAllText(_logPath, "timestamp_s,inference_ms,render_fps,inference_fps,detection_count,memory_mb\n");
    }

    void Update() {
        float now = Time.time;
        float elapsed = now - _sessionStart;
        _renderFrameCount++;

        //warmup
        if (!_warmedUp && elapsed >= warmupSec)
        {
            _warmedUp = true;
            _statusMsg = "REC";
            _renderFrameCount = 0;
            _inferenceFrameCount = 0;
            _lastFpsWindowStart = now;
        }

        //sample FPS
        float window = now - _lastFpsWindowStart;
        if (window >= 1.0f) {
            float rFps = _renderFrameCount / window;
            float iFps = _inferenceFrameCount / window;

            if (_warmedUp && _running) {
                _renderFpsSamples.Add(rFps);
                _inferenceFpsSamples.Add(iFps);
            }

            _liveRenderFps = rFps;
            _liveInferenceFps = iFps;
            _renderFrameCount = 0;
            _inferenceFrameCount = 0;
            _lastFpsWindowStart = now;
        }

        if (_running && _warmedUp && benchmarkDurationSec > 0 && elapsed >= warmupSec + benchmarkDurationSec) {
            _running = false;
            _statusMsg = "DONE";
            SaveResults();
        }
    }

    public void LogFrame(float inferenceMs, int detectionCount) {
        if (!_running || !_warmedUp) return;

        _inferenceFrameCount++;
        _inferenceTimes.Add(inferenceMs);
        _detectionCounts.Add(detectionCount);

        _liveInferenceMs = inferenceMs;
        _liveAvgInferenceMs = Average(_inferenceTimes);
        _liveDets = detectionCount;

        float memMB = (float)System.GC.GetTotalMemory(false) / (1024f * 1024f);
        float ts = Time.time - _sessionStart;
        string row = $"{ts:F2},{inferenceMs:F1},{_liveRenderFps:F1}," + $"{_liveInferenceFps:F1},{detectionCount},{memMB:F1}\n";
        try { File.AppendAllText(_logPath, row); }
        catch { }
    }

    public void SaveResults() {
        if (_saved) return; _saved = true;
        if (_inferenceTimes.Count == 0) { Debug.LogWarning("[Benchmark] No data � was LogFrame() called from the runner?"); return; }

        var sorted = new List<float>(_inferenceTimes);
        sorted.Sort();

        float avgInf = Average(_inferenceTimes);
        float minInf = sorted[0];
        float maxInf = sorted[sorted.Count - 1];
        float p50Inf = Percentile(sorted, 50);
        float p95Inf = Percentile(sorted, 95);
        float p99Inf = Percentile(sorted, 99);
        float avgRFps = Average(_renderFpsSamples);
        float minRFps = _renderFpsSamples.Count > 0 ? Min(_renderFpsSamples) : 0;
        float avgIFps = Average(_inferenceFpsSamples);
        float avgDets = Average(_detectionCounts);
        float memMB = (float)System.GC.GetTotalMemory(false) / (1024f * 1024f);

        string summary =
            $"\n# BENCHMARK SUMMARY\n" +
            $"# Model,{modelName}\n" +
            $"# Inference frames,{_inferenceTimes.Count}\n" +
            $"# Avg inference ms,{avgInf:F1}\n" +
            $"# Min inference ms,{minInf:F1}\n" +
            $"# Max inference ms,{maxInf:F1}\n" +
            $"# P50 inference ms,{p50Inf:F1}\n" +
            $"# P95 inference ms,{p95Inf:F1}\n" +
            $"# P99 inference ms,{p99Inf:F1}\n" +
            $"# Avg render FPS,{avgRFps:F1}\n" +
            $"# Min render FPS,{minRFps:F1}\n" +
            $"# Avg inference per sec,{avgIFps:F1}\n" +
            $"# Avg detections per frame,{avgDets:F2}\n" +
            $"# Memory MB,{memMB:F1}\n";

        try { File.AppendAllText(_logPath, summary); } catch { }
    }

    void OnDestroy() => SaveResults();
    void OnApplicationQuit() => SaveResults();

    void OnGUI() {
        if (!showOverlay) return;

        var style = new GUIStyle(GUI.skin.box) { fontSize = 12, alignment = TextAnchor.UpperLeft };
        style.normal.textColor = _statusMsg == "REC" ? Color.green : _statusMsg == "DONE" ? Color.yellow : Color.cyan;

        float elapsed = Time.time - _sessionStart;
        float remaining = Mathf.Max(0, warmupSec + benchmarkDurationSec - elapsed);

        string text =
            $"[{_statusMsg}] {modelName}\n" +
            $"Inf:      {_liveInferenceMs:F0} ms  avg {_liveAvgInferenceMs:F0}\n" +
            $"Render:   {_liveRenderFps:F1} FPS\n" +
            $"Inf/s:    {_liveInferenceFps:F1}\n" +
            $"Dets:     {_liveDets}\n" +
            $"Frames:   {_inferenceTimes.Count}\n" +
            (_statusMsg == "WARMUP" ? $"Warmup:   {Mathf.Max(0, warmupSec - elapsed):F0}s left" : $"Rec left: {remaining:F0}s");
        GUI.Box(new Rect(10, 10, 280, 175), text, style);
    }

    static float Average(List<float> l) {
        if (l.Count == 0) return 0f;
        float s = 0f; foreach (var v in l) s += v; return s / l.Count;
    }
    static float Average(List<int> l) {
        if (l.Count == 0) return 0f;
        float s = 0f; foreach (var v in l) s += v; return s / l.Count;
    }
    static float Min(List<float> l) {
        float m = float.MaxValue; foreach (var v in l) if (v < m) m = v; return m;
    }
    static float Percentile(List<float> sorted, float p) {
        if (sorted.Count == 0) return 0f;
        int idx = Mathf.Clamp(Mathf.RoundToInt(p / 100f * sorted.Count), 0, sorted.Count - 1);
        return sorted[idx];
    }
}