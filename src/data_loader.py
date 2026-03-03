"""
Data Loading and Preprocessing Module for CIFAR-10 Image Classification.

This module handles:
- Loading the CIFAR-10 dataset from Keras
- Normalizing pixel values to [0, 1]
- One-hot encoding labels
- Data augmentation via ImageDataGenerator
"""

import numpy as np
from tensorflow import keras
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# CIFAR-10 class names (in order of label index 0-9)
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

NUM_CLASSES = 10
IMG_SHAPE = (32, 32, 3)


def load_cifar10_data():
    """
    Load the CIFAR-10 dataset and return raw train/test splits.

    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
            - x: uint8 images of shape (N, 32, 32, 3)
            - y: integer labels of shape (N, 1)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Normalize images and one-hot encode labels.

    Args:
        x_train: Training images (uint8).
        y_train: Training labels (int).
        x_test: Test images (uint8).
        y_test: Test labels (int).

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
            - x: float32 images normalized to [0, 1]
            - y: one-hot encoded labels of shape (N, 10)
    """
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test


def create_data_augmentation_generator():
    """
    Create an ImageDataGenerator with augmentation transformations.

    Augmentations applied:
    - Random rotation up to 15 degrees
    - Random width/height shift up to 10%
    - Random horizontal flip
    - Random zoom up to 10%

    Returns:
        ImageDataGenerator: configured generator for training data augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    return datagen


def get_prepared_data():
    """
    Full pipeline: load CIFAR-10, preprocess, and return ready-to-use data.

    Returns:
        dict with keys:
            'x_train', 'y_train', 'x_test', 'y_test': processed arrays
            'datagen': ImageDataGenerator for augmented training
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    datagen = create_data_augmentation_generator()
    datagen.fit(x_train)

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'datagen': datagen
    }
