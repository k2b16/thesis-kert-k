using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DetectionOverlay : MonoBehaviour {
    //plain data passed in by runners
    //image space [0,1]
    [System.Serializable]
    public struct Detection {
        public float left, top, right, bottom;
        public float score;
        public int classId;
        public string label;
    }

    public RectTransform overlayRoot; //parent canvas
    public RectTransform boxPrefab; //prefab
    public int maxBoxes = 30; //hard cap (pool size)

    // pre allocated pool of RectTransforms.
    readonly List<RectTransform> _pool = new();

    void Awake() {
        //default overlay root
        if (overlayRoot == null) overlayRoot = (RectTransform)transform;
        WarmPool();
    }
    //instantiates pool entries, safe to call multiple times
    void WarmPool() {
        while (_pool.Count < maxBoxes) {
            var rt = Instantiate(boxPrefab, overlayRoot);
            rt.gameObject.SetActive(false);
            _pool.Add(rt);
        }
    }
    // position and label up to _pool.Count det boxes
    public void Render(List<Detection> dets, float scoreThreshold) {
        WarmPool();

        //hide everything
        for (int i = 0; i < _pool.Count; i++)
            _pool[i].gameObject.SetActive(false);

        int shown = 0;
        for (int i = 0; i < dets.Count && shown < _pool.Count; i++) {
            var d = dets[i];
            if (d.score < scoreThreshold) continue;

            // min/max, zero cost
            float l = Mathf.Min(d.left, d.right);
            float r = Mathf.Max(d.left, d.right);
            float t = Mathf.Min(d.top, d.bottom);
            float b = Mathf.Max(d.top, d.bottom);

            //cord system conversion
            // input image converion, output UGUI convention
            float xMin = Mathf.Clamp01(l);
            float xMax = Mathf.Clamp01(r);
            float yMin = Mathf.Clamp01(1f - b);
            float yMax = Mathf.Clamp01(1f - t);

            //pos the pooled box by normalized anchors inside overlayRoot
            var box = _pool[shown++];
            box.pivot = new Vector2(0.5f, 0.5f);
            box.anchorMin = new Vector2(xMin, yMin);
            box.anchorMax = new Vector2(xMax, yMax);
            box.offsetMin = Vector2.zero;
            box.offsetMax = Vector2.zero;

            //update label
            var txt = box.GetComponentInChildren<TMP_Text>(true);
            if (txt != null) txt.text = $"{d.label}  {(d.score * 100f):0}%";
            box.gameObject.SetActive(true);
        }
    }
}