from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..train import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_PATH,
    AxialCNN,
    create_data_loaders,
)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIRECTORY: Path = MODEL_PATH.parent / "quantized"
OUTPUT_MODEL_PATH = OUTPUT_DIRECTORY / "axial_classifier_int8.npz"
OUTPUT_CONFIG_PATH = OUTPUT_DIRECTORY / "quantization_config.json"

# Input images are normalized to -1..1, so int8 input = round(pixel * 127)
INPUT_SCALE = 1.0 / 127.0

# Layers that get int8 weights:
# (layer name, module path in the model, input activation, output activation)
QUANTIZED_LAYERS = [
    ("conv1", "features.0", "input", "conv1"),
    ("conv2", "features.3", "pool1", "conv2"),
    ("conv3", "features.6", "pool2", "conv3"),
    ("conv4", "features.9", "pool3", "conv4"),
    ("conv5", "features.11", "conv4", "conv5"),
    ("fc6", "classifier.0", "pool5", "fc6"),
    ("fc7", "classifier.2", "fc6", "fc7"),
]

LAYER_ORDER = [
    "conv1", "pool1", "conv2", "pool2", "conv3", "pool3",
    "conv4", "conv5", "pool5", "fc6", "fc7",
]

# ============================================================
# CALIBRATION
# ============================================================

def calculate_symmetric_scale(minimum: float, maximum: float) -> float:
    maximum_absolute = max(abs(minimum), abs(maximum))

    if maximum_absolute == 0.0:
        return 1.0

    return maximum_absolute / 127.0


def get_monitored_layers(model: AxialCNN) -> dict[str, nn.Module]:
    # Activation = output after the ReLU (or pool / last linear layer)
    return {
        "conv1": model.features[1],
        "pool1": model.features[2],
        "conv2": model.features[4],
        "pool2": model.features[5],
        "conv3": model.features[7],
        "pool3": model.features[8],
        "conv4": model.features[10],
        "conv5": model.features[12],
        "pool5": model.features[13],
        "fc6": model.classifier[1],
        "fc7": model.classifier[2],
    }


def run_activation_calibration(
    model: AxialCNN,
    calibration_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    activation_ranges: dict[str, dict[str, float]] = defaultdict(
        lambda: {"minimum": float("inf"), "maximum": float("-inf")}
    )

    hooks = []

    def register_activation_hook(
        module: nn.Module,
        layer_name: str,
    ) -> None:
        def hook(current_module, inputs, output) -> None:
            current_minimum = float(output.min().item())
            current_maximum = float(output.max().item())

            activation_ranges[layer_name]["minimum"] = min(
                activation_ranges[layer_name]["minimum"],
                current_minimum,
            )
            activation_ranges[layer_name]["maximum"] = max(
                activation_ranges[layer_name]["maximum"],
                current_maximum,
            )

        hooks.append(module.register_forward_hook(hook))

    for name, layer in get_monitored_layers(model).items():
        register_activation_hook(layer, name)

    model.eval()

    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images.to(device))

    for hook in hooks:
        hook.remove()

    activation_scales: dict[str, float] = {"input": INPUT_SCALE}

    print()
    print("========================================")
    print("Calibration: activation ranges and scales")
    print("========================================")

    for name, values in activation_ranges.items():
        scale = calculate_symmetric_scale(
            values["minimum"],
            values["maximum"],
        )
        activation_scales[name] = scale

        print(
            f"{name:8s} "
            f"min={values['minimum']:10.6f} "
            f"max={values['maximum']:10.6f} "
            f"scale={scale:.10f}"
        )

    return activation_scales


def calculate_weight_scales(model: AxialCNN) -> dict[str, float]:
    weight_scales: dict[str, float] = {}

    print()
    print("========================================")
    print("Calibration: weight ranges and scales")
    print("========================================")

    for layer_name, module_path, _, _ in QUANTIZED_LAYERS:
        weight = model.get_submodule(module_path).weight

        minimum = float(weight.min().item())
        maximum = float(weight.max().item())
        scale = calculate_symmetric_scale(minimum, maximum)

        weight_scales[layer_name] = scale

        print(
            f"{layer_name:5s} "
            f"min={minimum:10.6f} "
            f"max={maximum:10.6f} "
            f"scale={scale:.10f}"
        )

    return weight_scales


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_int8(values: torch.Tensor, scale: float) -> np.ndarray:
    quantized = torch.round(values / scale)
    quantized = torch.clamp(quantized, -128, 127)

    return quantized.to(torch.int8).cpu().numpy()


def quantize_bias_int32(
    bias: torch.Tensor,
    input_scale: float,
    weight_scale: float,
) -> np.ndarray:
    bias_scale = input_scale * weight_scale

    quantized = torch.round(bias / bias_scale)

    int32_minimum = -(2**31)
    int32_maximum = (2**31) - 1

    quantized = torch.clamp(quantized, int32_minimum, int32_maximum)

    return quantized.to(torch.int32).cpu().numpy()


def add_quantized_layer(
    exported_data: dict[str, np.ndarray],
    layer_name: str,
    layer: nn.Module,
    input_scale: float,
    output_scale: float,
    weight_scale: float,
) -> None:
    exported_data[f"{layer_name}_weight"] = quantize_int8(
        layer.weight.detach(),
        weight_scale,
    )

    exported_data[f"{layer_name}_bias"] = quantize_bias_int32(
        layer.bias.detach(),
        input_scale,
        weight_scale,
    )

    multiplier = input_scale * weight_scale / output_scale

    exported_data[f"{layer_name}_multiplier"] = np.asarray(
        multiplier,
        dtype=np.float64,
    )

    print(
        f"{layer_name:5s} | "
        f"input scale={input_scale:.10f} | "
        f"weight scale={weight_scale:.10f} | "
        f"output scale={output_scale:.10f} | "
        f"multiplier={multiplier:.10f}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = AxialCNN(number_of_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Calibrate on the validation images (no augmentation, not the test set)
    _, validation_loader, _ = create_data_loaders()

    activation_scales = run_activation_calibration(
        model,
        validation_loader,
        device,
    )
    weight_scales = calculate_weight_scales(model)

    print()
    print("========================================")
    print("Quantizing layers")
    print("========================================")

    exported_data: dict[str, np.ndarray] = {}

    for layer_name, module_path, input_name, output_name in QUANTIZED_LAYERS:
        add_quantized_layer(
            exported_data,
            layer_name,
            model.get_submodule(module_path),
            activation_scales[input_name],
            activation_scales[output_name],
            weight_scales[layer_name],
        )

    np.savez(OUTPUT_MODEL_PATH, **exported_data)

    configuration = {
        "input_scale": INPUT_SCALE,
        "activation_scales": activation_scales,
        "weight_scales": weight_scales,
        "layer_order": LAYER_ORDER,
        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
        "class_names": CLASS_NAMES,
    }

    with OUTPUT_CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    print()
    print("========================================")
    print("Quantization export complete")
    print(f"Model:  {OUTPUT_MODEL_PATH}")
    print(f"Config: {OUTPUT_CONFIG_PATH}")
    print("========================================")


if __name__ == "__main__":
    main()