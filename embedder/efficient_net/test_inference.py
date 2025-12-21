import argparse
import numpy as np
import torch
from PIL import Image
from model import EfficientNetB0Embed
from torchvision import transforms


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    """
    Возвращает тензор float32 NCHW: [1, 3, image_size, image_size],
    отмасштабированный и нормализованный под ImageNet.
    """
    img = Image.open(path).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # H x W
            transforms.ToTensor(),  # [0, 1], CHW
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    x = preprocess(img)  # [3, H, W]
    x = x.unsqueeze(0)  # [1, 3, H, W]

    return x


@torch.no_grad()
def embed_efficientnet_b0(x: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
    """
    Возвращает эмбеддинг [B, 1280].
    """

    model = EfficientNetB0Embed(l2_normalize=l2_normalize)
    embed = model.forward(x)
    return embed


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    a,b: (d,)
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img1", required=True)
    p.add_argument("--img2", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    x1 = load_image(args.img1).to(device)
    x2 = load_image(args.img2).to(device)

    e1 = embed_efficientnet_b0(x1).squeeze(0).cpu().numpy()  # (1280,)
    e2 = embed_efficientnet_b0(x2).squeeze(0).cpu().numpy()

    sim = cosine_similarity(e1, e2)
    print(f"Embedding dim: {e1.shape[0]}")
    print(f"Cosine similarity: {sim:.6f}")


if __name__ == "__main__":
    main()
