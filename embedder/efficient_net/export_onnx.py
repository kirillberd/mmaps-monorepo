from model import EfficientNetB0Embed
from pathlib import Path
import torch


def export_onnx(
    out_path: str | Path = ".onnx/model.onnx",
    l2_normalize: bool = True,
    opset: int = 17,
) -> Path:
    """
    Exports EfficientNetB0Embed to ONNX at `.onnx/model.onnx`.

    Assumptions:
      - input is a ready tensor shaped [1, 3, 224, 224] (NCHW, float32)
      - model in eval mode
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = EfficientNetB0Embed(l2_normalize=l2_normalize).eval()

    # Dummy input matching your runtime tensor shape
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Export
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes=None,  # fixed [1,3,224,224] as requested
        )

    return out_path


if __name__ == "__main__":
    p = export_onnx()
    print(f"Saved: {p.resolve()}")
