from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .train import (
    BASE_DIRECTORY,
    IMAGE_SIZE,
    MODEL_PATH,
    AxialCNN,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Folder "tesst" is at the same level as "src"
TEST_DIRECTORY = BASE_DIRECTORY.parents[1] / "tesst"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Predictions below this confidence are marked with "?"
LOW_CONFIDENCE = 0.70

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(device: torch.device) -> tuple[AxialCNN, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun train.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = AxialCNN(number_of_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


# ============================================================
# IMAGE PIPELINE
# ============================================================

def create_transform() -> transforms.Compose:
    # Same as the evaluation transform in train.py
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_image(
    model: AxialCNN,
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """Returns the probability of every class, shape [num_classes]."""
    # Add batch dimension: 1 x 1 x 64 x 64
    inputs = transform(image).unsqueeze(0).to(device)

    logits = model(inputs)

    return torch.softmax(logits, dim=1)[0].cpu()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Prediction device: {device}")

    if not TEST_DIRECTORY.exists():
        print(f"Folder not found: {TEST_DIRECTORY}")
        return

    image_paths = find_images(TEST_DIRECTORY)

    if not image_paths:
        print(f"No images found in: {TEST_DIRECTORY}")
        return

    model, class_names = load_model(device)
    transform = create_transform()

    counts = {name: 0 for name in class_names}
    low_confidence_count = 0
    unreadable_count = 0

    print()
    print(f"{'Image':<36}{'Prediction':<12}{'Confidence':>11}  Probabilities")
    print("-" * 100)

    for image_path in image_paths:
        relative_path = image_path.relative_to(TEST_DIRECTORY)

        try:
            image = Image.open(image_path)
            probabilities = predict_image(model, image, transform, device)
        except OSError as error:
            print(f"{str(relative_path):<36}cannot read image ({error})")
            unreadable_count += 1
            continue

        confidence, class_index = probabilities.max(dim=0)
        predicted_name = class_names[class_index.item()]
        counts[predicted_name] += 1

        is_low_confidence = confidence.item() < LOW_CONFIDENCE
        low_confidence_count += int(is_low_confidence)

        probability_text = " | ".join(
            f"{name}={probability:.3f}"
            for name, probability in zip(class_names, probabilities.tolist())
        )

        print(
            f"{str(relative_path):<36}"
            f"{predicted_name:<12}"
            f"{confidence.item() * 100:>10.2f}%"
            f"{'?' if is_low_confidence else ' '} "
            f"{probability_text}"
        )

    predicted_total = sum(counts.values())

    print()
    print("========================================")
    print(f"Total images: {len(image_paths)}")

    for name, count in counts.items():
        print(f"{name}: {count}")

    print(f"Low confidence (< {LOW_CONFIDENCE * 100:.0f}%, marked ?): {low_confidence_count}")

    if unreadable_count:
        print(f"Unreadable: {unreadable_count}")

    print(f"Predicted: {predicted_total}")
    print("========================================")


if __name__ == "__main__":
    main()