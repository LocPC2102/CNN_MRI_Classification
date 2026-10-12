# CNN Classification: Axial / Sagittal / None (Lumbar MRI)

## Overview

The problem is set up as a **3-class image classification** task on lumbar spine MRI:

| Index | Class | Description |
|---|---|---|
| 0 | `axial` | Axial-plane MRI slices of the lumbar spine (T1 / T2) |
| 1 | `none` | Images that are neither axial nor sagittal spine (brain, heart, shoulder, knee, neck and blank images) |
| 2 | `sagittal` | Sagittal-plane MRI slices of the spine |

The model is a small CNN designed for FPGA deployment. It is trained in FP32, then converted to **INT8** with post-training quantization.

## Dataset

The dataset has **3,282 images**, stored in one folder per class (`axial`, `negative`, `sagittal`). The `negative` folder is the `none` class.

The data is split randomly with a fixed seed (42) into 70% Train / 15% Validation / 15% Test:

| Split | Images | axial | none | sagittal |
|---|---|---|---|---|
| Train | 2,298 | 624 | 902 | 772 |
| Validation | 492 | 131 | 203 | 158 |
| Test | 492 | 145 | 177 | 170 |
| **Total** | **3,282** | **900** | **1,282** | **1,100** |

## Preprocessing

All images go through the same steps:

1. Convert to grayscale (1 channel).
2. Resize to 64×64 pixels.
3. Convert to a tensor in [0, 1], then normalize with mean 0.5 and std 0.5, giving values in [-1, 1].

The symmetric range [-1, 1] makes INT8 quantization simple: the input scale is fixed at 1/127, so `int8 = round(pixel × 127)`.

Data augmentation is applied **only to the Train set**: random rotation (±10°) and random translation (up to 5%). Flips and large rotations are not used, so axial and sagittal views are never confused by the augmentation. Validation and Test use no augmentation.

## CNN Architecture

The network has 5 convolution layers and 2 fully connected layers. It uses ReLU and 2×2 max pooling. It does not use BatchNorm or Dropout, which keeps the hardware simple. Every layer has a bias.

| Layer | Operation | Output shape | Parameters (weights + bias) | MACs |
|---|---|---|---|---|
| Input | grayscale image | 1×64×64 | | |
| conv1 | Conv 1→8, k5, pad 2, ReLU | 8×64×64 | 200 + 8 | 819,200 |
| pool1 | MaxPool 2×2 | 8×32×32 | | |
| conv2 | Conv 8→16, k3, pad 1, ReLU | 16×32×32 | 1,152 + 16 | 1,179,648 |
| pool2 | MaxPool 2×2 | 16×16×16 | | |
| conv3 | Conv 16→32, k3, pad 1, ReLU | 32×16×16 | 4,608 + 32 | 1,179,648 |
| pool3 | MaxPool 2×2 | 32×8×8 | | |
| conv4 | Conv 32→32, k3, pad 1, ReLU | 32×8×8 | 9,216 + 32 | 589,824 |
| conv5 | Conv 32→16, k3, pad 1, ReLU | 16×8×8 | 4,608 + 16 | 294,912 |
| pool5 | MaxPool 2×2 | 16×4×4 | | |
| flatten | order C, H, W | 256 | | |
| fc6 | Linear 256→32, ReLU | 32 | 8,192 + 32 | 8,192 |
| fc7 | Linear 32→3 (logits) | 3 | 96 + 3 | 96 |

- Total parameters: **28,211** (28,072 weights and 139 biases).
- Total compute: about **4.07 million MACs** per image.
- INT8 model size: about **28.6 KB** (28,072 bytes of int8 weights and 556 bytes of int32 biases).

## Training

| Setting | Value |
|---|---|
| Hardware | GPU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 20 |
| Loss | CrossEntropyLoss |
| Input | 1×64×64 |
| Random seed | 42 |

The checkpoint with the highest validation accuracy is kept (99.19%, reached at epoch 18). The final model is then evaluated once on the Test set.

## Inference

For each image the model applies the same preprocessing as in training (grayscale, resize to 64×64, normalize) and outputs 3 logits. The class with the largest logit is the prediction. A softmax over the logits gives a confidence score for each class.

## Results (FP32)

### Overall

| Metric | Train | Validation | Test |
|---|---|---|---|
| Images | 2,298 | 492 | 492 |
| Accuracy | 99.22% | 99.19% | **98.98%** |
| Macro-F1 | 0.9922 | 0.9918 | **0.9897** |
| Loss | 0.0267 | 0.0493 | 0.0306 |

Train images are evaluated without augmentation. The Test accuracy is 487 correct out of 492 images, with 5 mistakes.

### Per-class results on the Test set

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| axial | 0.9862 | 0.9862 | 0.9862 | 145 |
| none | 0.9831 | 0.9887 | 0.9859 | 177 |
| sagittal | 1.0000 | 0.9941 | 0.9971 | 170 |

### Confusion matrix on the Test set (rows: true, columns: predicted)

| | axial | none | sagittal |
|---|---|---|---|
| **axial** | 143 | 2 | 0 |
| **none** | 2 | 175 | 0 |
| **sagittal** | 0 | 1 | 169 |

Axial and sagittal images are never confused with each other. All 5 test mistakes involve the `none` class. The same pattern holds in the Train and Validation sets.

## INT8 Quantization

### Method

The trained FP32 model is converted with **post-training quantization**:

- **Weights and activations:** INT8, symmetric, one scale per tensor. `scale = max(|min|, |max|) / 127`.
- **Input:** INT8 with a fixed scale of 1/127.
- **Bias:** INT32 with scale `input_scale × weight_scale`, so it adds directly to the accumulator.
- **Accumulation:** 32-bit integers.
- **Requantization:** after each layer, `output = (accumulator × M) >> 24`, where `M` is an integer multiplier equal to `input_scale × weight_scale / output_scale` in Q24 format. The shift rounds half to even, then ReLU (when present) and saturation to [-128, 127] are applied.
- **Max pooling:** works directly on INT8 values, so a pooled tensor keeps the scale of the layer before it.
- **Calibration:** activation ranges are measured on the Validation images (492, no augmentation). The Test set is not used.

### Scales and multipliers

| Layer | Calibrated range (FP32 output) | Input scale | Weight scale | Output scale | Multiplier | Q24 integer |
|---|---|---|---|---|---|---|
| conv1 | 0 to 2.831 | 0.0078740 | 0.0021526 | 0.0222919 | 0.00076036 | 12,757 |
| conv2 | 0 to 3.523 | 0.0222919 | 0.0019204 | 0.0277390 | 0.00154332 | 25,893 |
| conv3 | 0 to 7.857 | 0.0277390 | 0.0017227 | 0.0618639 | 0.00077243 | 12,959 |
| conv4 | 0 to 23.837 | 0.0618639 | 0.0015613 | 0.1876892 | 0.00051463 | 8,634 |
| conv5 | 0 to 47.761 | 0.1876892 | 0.0017223 | 0.3760746 | 0.00085957 | 14,421 |
| fc6 | 0 to 51.330 | 0.3760746 | 0.0013152 | 0.4041711 | 0.00122376 | 20,531 |
| fc7 | -69.145 to 50.069 | 0.4041711 | 0.0019319 | 0.5444505 | 0.00143413 | 24,061 |

### Accuracy: FP32 vs INT8

The integer model was run in Python (integer arithmetic only, as designed for the hardware) on the Validation and Test sets:

| Set | FP32 accuracy | INT8 accuracy | Accuracy drop | Same prediction as FP32 | Mean logit error | Max logit error |
|---|---|---|---|---|---|---|
| Validation | 99.19% | 99.19% | 0.00% | 492 / 492 | 0.179 | 0.748 (1.4 steps) |
| Test | 98.98% | 98.98% | 0.00% | 492 / 492 | 0.188 | 8.768 (see note) |

- One step of the final layer is 0.5445. The mean logit error is about 0.3 step, so the INT8 model follows the FP32 logits closely.
- The Test maximum error of 8.77 comes from one logit that was clipped to 127. The final-layer scale was calibrated on the Validation set, and one Test image has a larger logit. The prediction for that image did not change.
- Example (Test image 0, true class `axial`): FP32 logits `[15.19, 4.18, -26.71]`, INT8 logits dequantized `[15.24, 4.36, -26.68]`. Both predict `axial`.

The INT8 outputs of every layer for one test image are saved as `.npy` files, to be used as the reference when checking the RTL implementation.

## Conclusion

The lightweight CNN (5 convolution layers, 2 fully connected layers, 28,211 parameters, about 4.07 M MACs per image) classifies lumbar MRI images into `axial`, `sagittal` and `none` with **98.98% accuracy and 0.9897 Macro-F1** on the Test set (487 of 492 images correct). The 5 mistakes all involve the `none` class, and axial and sagittal images are never confused.

After post-training INT8 quantization, the model keeps exactly the same predictions as FP32 on all 984 Validation and Test images (no accuracy drop), with a compact 28.6 KB parameter size. This makes it suitable for FPGA implementation.
