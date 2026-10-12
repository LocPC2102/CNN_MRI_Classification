from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .quantization.quantize import OUTPUT_CONFIG_PATH, OUTPUT_MODEL_PATH
from .train import CLASS_NAMES, MODEL_PATH, AxialCNN, create_data_loaders

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "integer_results"

# Index of the test image whose layer outputs are saved for the RTL testbench
TEST_IMAGE_INDEX = 0

# Requantization: output = (accumulator * integer_multiplier) >> REQUANT_SHIFT
# (must match the shift used in the RTL)
REQUANT_SHIFT = 24

# The network in execution order:
# (layer name, kind, kernel size, padding, apply ReLU)
INTEGER_NETWORK = [
    ("conv1", "conv", 5, 2, True),
    ("pool1", "pool", 0, 0, False),
    ("conv2", "conv", 3, 1, True),
    ("pool2", "pool", 0, 0, False),
    ("conv3", "conv", 3, 1, True),
    ("pool3", "pool", 0, 0, False),
    ("conv4", "conv", 3, 1, True),
    ("conv5", "conv", 3, 1, True),
    ("pool5", "pool", 0, 0, False),
    ("flatten", "flatten", 0, 0, False),
    ("fc6", "linear", 0, 0, True),
    ("fc7", "linear", 0, 0, False),
]

# Layers that have weights (and therefore a multiplier)
WEIGHT_LAYER_NAMES = [
    name for name, kind, _, _, _ in INTEGER_NETWORK if kind in ("conv", "linear")
]

# ============================================================
# QUANTIZATION HELPERS
# ============================================================

def quantize_tensor(values: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError(f"Scale must be positive, received {scale}")

    quantized = torch.round(values.to(torch.float64) / scale)
    quantized = torch.clamp(quantized, -128, 127)

    return quantized.to(torch.int8)


def float_multiplier_to_q_int(floating_multiplier: float, shift: int) -> int:
    if floating_multiplier < 0:
        raise ValueError("Requantization multiplier must be non-negative")

    integer_multiplier = int(round(floating_multiplier * (1 << shift)))

    if integer_multiplier > (2**31 - 1):
        raise OverflowError(
            f"Multiplier {integer_multiplier} does not fit in signed 32 bits"
        )

    return integer_multiplier


def requantize_accumulator(
    accumulator: torch.Tensor,
    integer_multiplier: int,
    shift: int,
    apply_relu: bool,
) -> torch.Tensor:
    """
    int32 accumulator -> int8 output.

    Multiply, shift right with round-half-to-even on the magnitude,
    optional ReLU, then saturate to -128..127.
    """
    product = accumulator.to(torch.int64) * int(integer_multiplier)

    sign = torch.sign(product)
    absolute_product = torch.abs(product)

    if shift == 0:
        rounded_magnitude = absolute_product
    else:
        divisor = 1 << shift
        half = 1 << (shift - 1)

        quotient = absolute_product // divisor
        remainder = absolute_product % divisor

        round_up = (remainder > half) | (
            (remainder == half) & (quotient % 2 == 1)
        )

        rounded_magnitude = torch.where(round_up, quotient + 1, quotient)

    scaled = sign * rounded_magnitude

    if apply_relu:
        scaled = torch.clamp(scaled, min=0)

    scaled = torch.clamp(scaled, -128, 127)

    return scaled.to(torch.int8)


# ============================================================
# INTEGER LAYERS
# ============================================================

def integer_conv2d(
    inputs: torch.Tensor,
    weights: np.ndarray,
    biases: np.ndarray,
    integer_multiplier: int,
    shift: int,
    kernel_size: int,
    stride: int,
    padding: int,
    apply_relu: bool,
) -> torch.Tensor:
    if inputs.dtype != torch.int8:
        raise TypeError(f"Expected INT8 convolution input, got {inputs.dtype}")

    batch_size, _, input_height, input_width = inputs.shape
    output_channels = weights.shape[0]

    # float64 holds every integer below 2^53 exactly, so this sum is exact
    unfolded = functional.unfold(
        inputs.to(torch.float64),
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )

    weight_matrix = torch.from_numpy(
        weights.astype(np.int64)
    ).to(torch.float64).reshape(output_channels, -1)

    accumulators = torch.matmul(weight_matrix.unsqueeze(0), unfolded)

    bias_tensor = torch.from_numpy(
        biases.astype(np.int64)
    ).to(torch.float64).reshape(1, output_channels, 1)

    accumulators = accumulators + bias_tensor

    output_height = (input_height + (2 * padding) - kernel_size) // stride + 1
    output_width = (input_width + (2 * padding) - kernel_size) // stride + 1

    accumulators = accumulators.reshape(
        batch_size, output_channels, output_height, output_width
    ).to(torch.int64)

    return requantize_accumulator(
        accumulators, integer_multiplier, shift, apply_relu
    )


def integer_max_pool2d(
    inputs: torch.Tensor,
    kernel_size: int = 2,
    stride: int = 2,
) -> torch.Tensor:
    if inputs.dtype != torch.int8:
        raise TypeError(f"Expected INT8 pooling input, got {inputs.dtype}")

    pooled = functional.max_pool2d(
        inputs.to(torch.float64), kernel_size=kernel_size, stride=stride
    )

    return pooled.to(torch.int8)


def integer_linear(
    inputs: torch.Tensor,
    weights: np.ndarray,
    biases: np.ndarray,
    integer_multiplier: int,
    shift: int,
    apply_relu: bool,
) -> torch.Tensor:
    if inputs.dtype != torch.int8:
        raise TypeError(f"Expected INT8 linear input, got {inputs.dtype}")

    input_int64 = inputs.to(torch.int64)
    weight_int64 = torch.from_numpy(weights.astype(np.int64))
    bias_int64 = torch.from_numpy(biases.astype(np.int64))

    accumulators = torch.matmul(input_int64, weight_int64.transpose(0, 1))
    accumulators = accumulators + bias_int64.reshape(1, -1)

    return requantize_accumulator(
        accumulators, integer_multiplier, shift, apply_relu
    )


def run_integer_network(
    quantized_inputs: torch.Tensor,
    quantized_model: np.lib.npyio.NpzFile,
    integer_multipliers: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Run every layer in integers. Returns the int8 output of each layer."""
    layer_outputs: dict[str, torch.Tensor] = {}
    x = quantized_inputs

    for name, kind, kernel_size, padding, apply_relu in INTEGER_NETWORK:
        if kind == "conv":
            x = integer_conv2d(
                inputs=x,
                weights=quantized_model[f"{name}_weight"],
                biases=quantized_model[f"{name}_bias"],
                integer_multiplier=integer_multipliers[name],
                shift=REQUANT_SHIFT,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                apply_relu=apply_relu,
            )
        elif kind == "pool":
            x = integer_max_pool2d(x)
        elif kind == "flatten":
            x = torch.flatten(x, start_dim=1)
        else:
            x = integer_linear(
                inputs=x,
                weights=quantized_model[f"{name}_weight"],
                biases=quantized_model[f"{name}_bias"],
                integer_multiplier=integer_multipliers[name],
                shift=REQUANT_SHIFT,
                apply_relu=apply_relu,
            )

        layer_outputs[name] = x

    return layer_outputs


# ============================================================
# LOADING
# ============================================================

def load_float_model() -> AxialCNN:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    model = AxialCNN(number_of_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_quantization() -> tuple[np.lib.npyio.NpzFile, dict, dict[str, int]]:
    if not OUTPUT_MODEL_PATH.exists() or not OUTPUT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Quantized model not found in {OUTPUT_MODEL_PATH.parent}\n"
            "Run quantization/quantize.py first."
        )

    quantized_model = np.load(OUTPUT_MODEL_PATH)

    with OUTPUT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        configuration = json.load(file)

    integer_multipliers = {
        name: float_multiplier_to_q_int(
            float(quantized_model[f"{name}_multiplier"]), REQUANT_SHIFT
        )
        for name in WEIGHT_LAYER_NAMES
    }

    return quantized_model, configuration, integer_multipliers


# ============================================================
# FLOAT vs INT8 COMPARISON ON A WHOLE SPLIT
# ============================================================

@torch.no_grad()
def evaluate_split(
    split_name: str,
    loader: DataLoader,
    float_model: AxialCNN,
    quantized_model: np.lib.npyio.NpzFile,
    configuration: dict,
    integer_multipliers: dict[str, int],
) -> None:
    input_scale = float(configuration["input_scale"])
    output_scale = float(configuration["activation_scales"]["fc7"])

    all_labels = []
    all_float_logits = []
    all_integer_logits = []

    for images, labels in loader:
        layer_outputs = run_integer_network(
            quantize_tensor(images, input_scale),
            quantized_model,
            integer_multipliers,
        )

        all_labels.append(labels)
        all_float_logits.append(float_model(images).to(torch.float64))
        all_integer_logits.append(layer_outputs["fc7"])

    labels = torch.cat(all_labels)
    float_logits = torch.cat(all_float_logits)
    integer_logits = torch.cat(all_integer_logits)

    # INT8 logits converted back to real numbers, to compare with float
    dequantized_logits = integer_logits.to(torch.float64) * output_scale

    float_predictions = float_logits.argmax(dim=1)
    integer_predictions = integer_logits.argmax(dim=1)

    total = len(labels)
    float_correct = (float_predictions == labels).sum().item()
    integer_correct = (integer_predictions == labels).sum().item()
    agreement = (float_predictions == integer_predictions).sum().item()

    # Logit error between INT8 and float
    logit_error = (dequantized_logits - float_logits).abs()
    maximum_error = logit_error.max().item()

    # Margin = top-1 logit minus top-2 logit of the float model.
    # An image can only flip if its margin is smaller than about 2x the error.
    top_two = float_logits.topk(2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    at_risk = (margin < 2 * maximum_error).sum().item()

    saturated = ((integer_logits >= 127) | (integer_logits <= -128)).sum().item()

    print()
    print("========================================")
    print(f"{split_name.upper()}: float vs INT8 ({total} images)")
    print("========================================")
    print(f"Float accuracy:   {float_correct / total * 100:.2f}%")
    print(f"INT8 accuracy:    {integer_correct / total * 100:.2f}%")
    print(f"Accuracy drop:    {(float_correct - integer_correct) / total * 100:+.2f}%")
    print(
        f"Same prediction:  {agreement}/{total} "
        f"({agreement / total * 100:.2f}%)"
    )
    print(
        f"Logit error:      mean={logit_error.mean().item():.4f} | "
        f"max={maximum_error:.4f} "
        f"(= {maximum_error / output_scale:.1f} steps of {output_scale:.4f})"
    )
    print(
        f"Smallest margin:  {margin.min().item():.4f} | "
        f"images with margin < 2x max error: {at_risk}"
    )
    print(f"Saturated INT8 logits: {saturated}")


# ============================================================
# SINGLE IMAGE (layer dump for the RTL testbench)
# ============================================================

def print_tensor_summary(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name:8s} "
        f"shape={str(tuple(tensor.shape)):18s} "
        f"min={int(tensor.min().item()):4d} "
        f"max={int(tensor.max().item()):4d}"
    )


def save_tensor(name: str, tensor: torch.Tensor) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIRECTORY / f"{name}.npy", tensor.cpu().numpy())


@torch.no_grad()
def run_single_image(
    loader: DataLoader,
    float_model: AxialCNN,
    quantized_model: np.lib.npyio.NpzFile,
    configuration: dict,
    integer_multipliers: dict[str, int],
) -> None:
    image, true_class = loader.dataset[TEST_IMAGE_INDEX]
    image = image.unsqueeze(0)

    float_logits = float_model(image)
    float_prediction = int(float_logits.argmax(dim=1).item())

    quantized_input = quantize_tensor(image, float(configuration["input_scale"]))
    layer_outputs = run_integer_network(
        quantized_input, quantized_model, integer_multipliers
    )

    print()
    print("========================================")
    print("Integer layer outputs (single image)")
    print("========================================")

    print_tensor_summary("Input", quantized_input)
    save_tensor("input", quantized_input)

    for name, output in layer_outputs.items():
        print_tensor_summary(name, output)
        save_tensor(name, output)

    integer_logits = layer_outputs["fc7"]
    integer_prediction = int(integer_logits.argmax(dim=1).item())

    fc7_scale = float(configuration["activation_scales"]["fc7"])
    dequantized_logits = integer_logits.to(torch.float64) * fc7_scale

    print()
    print("========================================")
    print("Integer inference result")
    print("========================================")
    print(f"Test image index: {TEST_IMAGE_INDEX}")
    print(f"True class: {true_class} - {CLASS_NAMES[true_class]}")
    print()
    print(f"Float logits:       {float_logits.numpy().flatten()}")
    print(f"INT8 logits:        {integer_logits.numpy().flatten()}")
    print(f"Dequantized logits: {dequantized_logits.numpy().flatten()}")
    print()
    print(f"Float prediction: {float_prediction} - {CLASS_NAMES[float_prediction]}")
    print(f"INT8 prediction:  {integer_prediction} - {CLASS_NAMES[integer_prediction]}")

    print(
        "Prediction agreement: "
        + ("PASS" if integer_prediction == float_prediction else "FAIL")
    )
    print(
        "INT8 classification: "
        + ("PASS" if integer_prediction == true_class else "INCORRECT")
    )
    print(f"Layer outputs saved to: {OUTPUT_DIRECTORY}")
    print("========================================")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    quantized_model, configuration, integer_multipliers = load_quantization()

    print()
    print("========================================")
    print("Q-format integer multipliers (must match the RTL)")
    print("========================================")

    for name in WEIGHT_LAYER_NAMES:
        print(f"{name:5s} | Q{REQUANT_SHIFT}={integer_multipliers[name]}")

    float_model = load_float_model()

    # Validation and test images only: no augmentation, same split as train.py
    _, validation_loader, test_loader = create_data_loaders()

    for split_name, loader in (
        ("validation", validation_loader),
        ("test", test_loader),
    ):
        evaluate_split(
            split_name,
            loader,
            float_model,
            quantized_model,
            configuration,
            integer_multipliers,
        )

    run_single_image(
        test_loader,
        float_model,
        quantized_model,
        configuration,
        integer_multipliers,
    )


if __name__ == "__main__":
    main()