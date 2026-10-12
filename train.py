from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIRECTORY = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIRECTORY / "datasets"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
MODEL_PATH = MODEL_DIRECTORY / "axial_classifier_cnn.pth"

IMAGE_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
RANDOM_SEED = 42

VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ImageFolder sorts folders alphabetically:
# axial -> 0, negative -> 1, sagittal -> 2
FOLDER_NAMES = ["axial", "negative", "sagittal"]
CLASS_NAMES = ["axial", "none", "sagittal"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# AXIAL CNN MODEL ARCHITECTURE
# ============================================================

class AxialCNN(nn.Module):
    def __init__(self, number_of_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: 1 x 64 x 64
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(inplace=False),

            # Output: 8 x 32 x 32
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=False),

            # Output: 16 x 16 x 16
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=False),

            # Output: 32 x 8 x 8
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=False),

            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=False),

            # Output: 16 x 4 x 4
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            # 16 x 4 x 4 = 256 inputs
            nn.Linear(256, 32, bias=True),
            nn.ReLU(inplace=False),

            # Output layer: no ReLU because these are logits
            nn.Linear(32, number_of_classes, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        logits = self.classifier(flattened)

        return logits


# ============================================================
# DATA PIPELINE
# ============================================================

def describe_split(
    split_name: str,
    indices: list[int],
    targets: list[int],
) -> None:
    counts = Counter(targets[index] for index in indices)
    summary = ", ".join(
        f"{CLASS_NAMES[class_index]}={counts[class_index]}"
        for class_index in range(NUM_CLASSES)
    )

    print(f"{split_name} images: {len(indices)} ({summary})")


def create_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    base_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    normalize_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]

    training_transform = transforms.Compose(
        base_transforms
        + [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
        + normalize_transforms
    )
    evaluation_transform = transforms.Compose(
        base_transforms + normalize_transforms
    )

    # Same files, two transforms: augmentation only for training
    training_source = datasets.ImageFolder(
        DATASET_ROOT, transform=training_transform
    )
    evaluation_source = datasets.ImageFolder(
        DATASET_ROOT, transform=evaluation_transform
    )

    if training_source.classes != FOLDER_NAMES:
        raise ValueError(
            f"Expected folders {FOLDER_NAMES}, found {training_source.classes}"
        )

    total_size = len(training_source)
    validation_size = int(VALIDATION_RATIO * total_size)
    test_size = int(TEST_RATIO * total_size)

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_size, generator=split_generator).tolist()

    validation_indices = indices[:validation_size]
    test_indices = indices[validation_size:validation_size + test_size]
    training_indices = indices[validation_size + test_size:]

    training_dataset = Subset(training_source, training_indices)
    validation_dataset = Subset(evaluation_source, validation_indices)
    test_dataset = Subset(evaluation_source, test_indices)

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    describe_split("Training", training_indices, training_source.targets)
    describe_split("Validation", validation_indices, training_source.targets)
    describe_split("Test", test_indices, training_source.targets)

    return training_loader, validation_loader, test_loader


# ============================================================
# TRAINING & EVALUATION LOOP
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_random_seed(RANDOM_SEED)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    training_loader, validation_loader, test_loader = (
        create_data_loaders()
    )

    model = AxialCNN(number_of_classes=NUM_CLASSES).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        training_loss, training_accuracy = train_one_epoch(
            model=model,
            loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={training_loss:.4f} | "
            f"train accuracy={training_accuracy * 100:.2f}% | "
            f"validation loss={validation_loss:.4f} | "
            f"validation accuracy={validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "architecture": {
                        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
                        "conv_filters": [8, 16, 32, 32, 16],
                        "fc_outputs": [32, NUM_CLASSES],
                    },
                },
                MODEL_PATH,
            )

            print(f"Saved improved model to: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate(
        model=model,
        loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print()
    print("========================================")
    print("Training complete")
    print(f"Best validation accuracy: "
          f"{best_validation_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Model: {MODEL_PATH}")
    print("========================================")


if __name__ == "__main__":
    main()