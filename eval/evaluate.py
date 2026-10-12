from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..train import (
    BATCH_SIZE,
    DATASET_ROOT,
    FOLDER_NAMES,
    IMAGE_SIZE,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_RATIO,
    VALIDATION_RATIO,
    AxialCNN,
)

# ============================================================
# CONFIGURATION
# ============================================================

# The report is saved next to this file
REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.txt"

# At most this many misclassified images are listed per split
MAX_LISTED_ERRORS = 20

# ============================================================
# DATA PIPELINE
# ============================================================

def create_evaluation_splits() -> tuple[datasets.ImageFolder, dict[str, Subset]]:
    # No augmentation for any split, so train metrics are comparable
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)

    if dataset.classes != FOLDER_NAMES:
        raise ValueError(
            f"Expected folders {FOLDER_NAMES}, found {dataset.classes}"
        )

    # Same split as train.py (same seed, same ratios)
    total_size = len(dataset)
    validation_size = int(VALIDATION_RATIO * total_size)
    test_size = int(TEST_RATIO * total_size)

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_size, generator=split_generator).tolist()

    splits = {
        "train": Subset(dataset, indices[validation_size + test_size:]),
        "validation": Subset(dataset, indices[:validation_size]),
        "test": Subset(dataset, indices[validation_size:validation_size + test_size]),
    }

    return dataset, splits


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
# METRICS
# ============================================================

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        total_loss += loss.item() * images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    average_loss = total_loss / len(labels)

    return average_loss, labels, predictions


def compute_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    # Rows: true class, columns: predicted class
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)

    return matrix


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_class_metrics(matrix: np.ndarray) -> list[dict[str, float]]:
    metrics = []

    for class_index in range(matrix.shape[0]):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1_score = safe_divide(2 * precision * recall, precision + recall)

        metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": int(matrix[class_index, :].sum()),
            }
        )

    return metrics


# ============================================================
# REPORT FORMATTING
# ============================================================

def format_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
) -> list[str]:
    column_width = max(len(name) for name in class_names) + 2

    lines = ["Confusion matrix (rows: true, columns: predicted)"]
    lines.append(
        " " * column_width + "".join(f"{name:>{column_width}}" for name in class_names)
    )

    for class_index, name in enumerate(class_names):
        row = "".join(f"{value:>{column_width}}" for value in matrix[class_index])
        lines.append(f"{name:<{column_width}}{row}")

    return lines


def format_split_report(
    split_name: str,
    average_loss: float,
    matrix: np.ndarray,
    class_names: list[str],
) -> tuple[list[str], dict[str, float]]:
    accuracy = np.trace(matrix) / matrix.sum()
    class_metrics = compute_class_metrics(matrix)
    macro_f1 = float(np.mean([metric["f1"] for metric in class_metrics]))

    lines = [
        "========================================",
        f"{split_name.upper()} ({matrix.sum()} images)",
        "========================================",
        f"Loss:     {average_loss:.4f}",
        f"Accuracy: {accuracy * 100:.2f}%",
        f"Macro F1: {macro_f1:.4f}",
        "",
        f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}",
    ]

    for name, metric in zip(class_names, class_metrics):
        lines.append(
            f"{name:<12}"
            f"{metric['precision']:>10.4f}"
            f"{metric['recall']:>10.4f}"
            f"{metric['f1']:>10.4f}"
            f"{metric['support']:>10}"
        )

    lines.append("")
    lines.extend(format_confusion_matrix(matrix, class_names))

    summary = {"loss": average_loss, "accuracy": accuracy, "f1": macro_f1}

    return lines, summary


def format_misclassified(
    file_paths: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> list[str]:
    wrong_indices = np.nonzero(labels != predictions)[0]

    lines = [f"Misclassified images: {len(wrong_indices)}"]

    for index in wrong_indices[:MAX_LISTED_ERRORS]:
        relative_path = os.path.relpath(file_paths[index], DATASET_ROOT)

        lines.append(
            f"  {relative_path}  "
            f"true={class_names[labels[index]]} -> "
            f"predicted={class_names[predictions[index]]}"
        )

    if len(wrong_indices) > MAX_LISTED_ERRORS:
        lines.append(f"  ... and {len(wrong_indices) - MAX_LISTED_ERRORS} more")

    lines.append("")

    return lines


def format_summary(summary: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "========================================",
        "SUMMARY",
        "========================================",
        f"{'Split':<12}{'Loss':>10}{'Accuracy':>12}{'Macro F1':>10}",
    ]

    for split_name, result in summary.items():
        lines.append(
            f"{split_name:<12}"
            f"{result['loss']:>10.4f}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['f1']:>10.4f}"
        )

    return lines


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, class_names = load_model(device)
    dataset, splits = create_evaluation_splits()
    loss_function = nn.CrossEntropyLoss()

    report_lines = [
        f"Evaluation report ({datetime.now():%Y-%m-%d %H:%M})",
        f"Model: {MODEL_PATH}",
        f"Dataset: {DATASET_ROOT}",
        "Classes: "
        + ", ".join(f"{index}={name}" for index, name in enumerate(class_names)),
        "Train images are evaluated WITHOUT augmentation.",
        "",
    ]

    summary: dict[str, dict[str, float]] = {}

    for split_name, subset in splits.items():
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

        average_loss, labels, predictions = collect_predictions(
            model=model,
            loader=loader,
            loss_function=loss_function,
            device=device,
        )

        matrix = compute_confusion_matrix(labels, predictions, len(class_names))

        split_lines, summary[split_name] = format_split_report(
            split_name, average_loss, matrix, class_names
        )
        report_lines.extend(split_lines)
        report_lines.append("")

        file_paths = [dataset.samples[index][0] for index in subset.indices]
        report_lines.extend(
            format_misclassified(file_paths, labels, predictions, class_names)
        )

    report_lines.extend(format_summary(summary))

    report_text = "\n".join(report_lines) + "\n"

    print(report_text)

    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()