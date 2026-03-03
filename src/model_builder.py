"""
Model Architecture Builder for CIFAR-10 Image Classification.

This module defines:
- A custom CNN architecture designed for 32x32 CIFAR-10 images
- A transfer learning model using MobileNetV2 pretrained on ImageNet
"""

from keras.api.models import Sequential, Model
from keras.api.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Input,
    UpSampling2D
)
from keras.api.applications import MobileNetV2

from .data_loader import NUM_CLASSES, IMG_SHAPE


def build_custom_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """
    Build a custom CNN architecture for CIFAR-10 classification.

    Architecture:
        Block 1: Conv2D(32) -> BatchNorm -> Conv2D(32) -> BatchNorm -> MaxPool -> Dropout(0.25)
        Block 2: Conv2D(64) -> BatchNorm -> Conv2D(64) -> BatchNorm -> MaxPool -> Dropout(0.25)
        Block 3: Conv2D(128) -> BatchNorm -> Conv2D(128) -> BatchNorm -> MaxPool -> Dropout(0.25)
        Dense: Flatten -> Dense(256) -> BatchNorm -> Dropout(0.5) -> Dense(10, softmax)

    Args:
        input_shape (tuple): Shape of input images (H, W, C). Default: (32, 32, 3)
        num_classes (int): Number of output classes. Default: 10

    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        # --- Block 1 ---
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Block 2 ---
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Block 3 ---
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Classifier ---
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def build_transfer_learning_model(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES,
                                   upscale=True):
    """
    Build a transfer learning model using MobileNetV2.

    MobileNetV2 is chosen because:
    - Lightweight and efficient (good for training on CPU/limited GPU)
    - Strong feature extraction from ImageNet pretraining
    - Keras docs: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
    - Requires minimum 96×96×3 input for efficient performance

    Architecture (GPU / upscale=True):
        Input(32,32,3) -> UpSampling2D(3x) -> MobileNetV2(frozen) ->
        GlobalAveragePooling2D -> Dense(256) -> Dropout(0.5) -> Dense(10, softmax)

    Architecture (CPU / upscale=False):
        Input(96,96,3) [pre-resized with cv2] -> MobileNetV2(frozen) ->
        GlobalAveragePooling2D -> Dense(256) -> Dropout(0.5) -> Dense(10, softmax)

    Args:
        input_shape (tuple): Shape of input images.
                             GPU: (32, 32, 3) — UpSampling2D handles resize.
                             CPU: (96, 96, 3) — images pre-resized with cv2.
        num_classes (int):  Number of output classes. Default: 10
        upscale (bool):     If True, adds UpSampling2D(3,3) inside the model
                            (GPU version, 03_transfer_learning_gpu.py).
                            If False, skips it — images must already be 96×96
                            (CPU version, 03_transfer_learning_cpu.py).

    Returns:
        tuple: (keras.Model, base_model) — full model and MobileNetV2 base
    """
    # Input layer
    inputs = Input(shape=input_shape)

    if upscale:
        # GPU version: upscale 32×32 → 96×96 inside the graph (runs each batch)
        x = UpSampling2D(size=(3, 3))(inputs)
        mobilenet_input_shape = (96, 96, 3)
    else:
        # CPU version: images already 96×96 (pre-resized with cv2 before training)
        # This eliminates per-batch upscaling — the main CPU bottleneck
        x = inputs
        mobilenet_input_shape = input_shape

    # Load MobileNetV2 pretrained on ImageNet, without top classification layers
    # Minimum efficient input: 96×96×3 (see Keras docs link above)
    base_model = MobileNetV2(
        input_shape=mobilenet_input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze all base model layers (we only train the new head)
    base_model.trainable = False

    x = base_model(x, training=False)

    # Custom classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, base_model


def get_model_summary(model):
    """
    Get a string summary of the model architecture.

    Args:
        model: Keras model

    Returns:
        str: Model summary string
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)
