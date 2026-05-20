using UnityEngine;
using Unity.Sentis;

// DEVELOPER UTILITY
public class ModelDebugger : MonoBehaviour {
    [Header("assing model")]
    public ModelAsset modelAsset;

    void Start() {
        if (modelAsset == null){ return; }
        var model = ModelLoader.Load(modelAsset);

        Debug.Log($"input ({model.inputs.Count}):");
        foreach (var inp in model.inputs) Debug.Log($"name='{inp.name}'");

        Debug.Log($"output ({model.outputs.Count}):");
        foreach (var outp in model.outputs) Debug.Log($"name='{outp.name}'");
    }
}