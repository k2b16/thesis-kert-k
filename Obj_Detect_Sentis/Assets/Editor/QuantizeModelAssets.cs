using System.IO;
using UnityEditor;
using UnityEngine;
using Unity.Sentis;

public static class QuantizeModelAssets
{
    const string MenuRoot = "Thesis/Quantize Selected Model Assets/";

    [MenuItem(MenuRoot + "Uint8")]
    static void QuantizeUint8() => QuantizeSelected(QuantizationType.Uint8);

    [MenuItem(MenuRoot + "Float16")]
    static void QuantizeFloat16() => QuantizeSelected(QuantizationType.Float16);

    static void QuantizeSelected(QuantizationType quantizationType)
    {
        if (Selection.objects == null || Selection.objects.Length == 0)
        {
            EditorUtility.DisplayDialog(
                "Quantize Model Assets",
                "Select one or more Sentis ModelAsset files in the Project window first.",
                "OK");
            return;
        }

        var suffix = quantizationType == QuantizationType.Uint8 ? "_uint8" : "_fp16";
        var converted = 0;

        foreach (var selected in Selection.objects)
        {
            if (selected is not ModelAsset modelAsset)
                continue;

            var sourcePath = AssetDatabase.GetAssetPath(modelAsset);
            if (string.IsNullOrEmpty(sourcePath))
                continue;

            var outputPath = Path.Combine(
                Path.GetDirectoryName(sourcePath) ?? "Assets",
                Path.GetFileNameWithoutExtension(sourcePath) + suffix + ".sentis");

            if (File.Exists(outputPath))
            {
                Debug.LogWarning($"Skipping existing file: {outputPath}");
                continue;
            }

            var model = ModelLoader.Load(modelAsset);
            ModelQuantizer.QuantizeWeights(quantizationType, ref model);
            ModelWriter.Save(outputPath, model);
            converted++;
            Debug.Log($"Quantized {sourcePath} -> {outputPath} ({quantizationType})");
        }

        AssetDatabase.Refresh();

        EditorUtility.DisplayDialog(
            "Quantize Model Assets",
            converted == 0
                ? "No ModelAsset files were converted. Select ONNX/Sentis assets in Assets/Models."
                : $"Created {converted} quantized .sentis file(s).\nAssign them to runner Model Asset fields and benchmark on device.",
            "OK");
    }

    [MenuItem(MenuRoot + "Uint8", true)]
    [MenuItem(MenuRoot + "Float16", true)]
    static bool ValidateQuantizeSelected() => Selection.objects != null && Selection.objects.Length > 0;
}
