"""
Training Pipeline for CIFAR-10 Image Classification.

This module provides functions to compile and train Keras models
with appropriate callbacks for preventing overfitting.
"""

import os
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Default training configuration
DEFAULT_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,          # max epochs (early stopping will cut short)
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
}


def compile_model(model, learning_rate=DEFAULT_CONFIG['learning_rate']):
    """
    Compile a Keras model with Adam optimizer and categorical crossentropy loss.

    Args:
        model: Keras model to compile.
        learning_rate (float): Initial learning rate for Adam optimizer.

    Returns:
        Compiled Keras model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_save_path, patience=DEFAULT_CONFIG['early_stopping_patience']):
    """
    Create a list of training callbacks for robust training.

    Callbacks:
    - EarlyStopping: Stops training if val_loss doesn't improve for `patience` epochs.
                     Restores the best weights.
    - ModelCheckpoint: Saves the best model based on val_loss.
    - ReduceLROnPlateau: Reduces learning rate by half if val_loss plateaus for 5 epochs.

    Args:
        model_save_path (str): File path to save the best model.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        list: List of Keras callback instances.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=DEFAULT_CONFIG['reduce_lr_factor'],
            patience=DEFAULT_CONFIG['reduce_lr_patience'],
            min_lr=DEFAULT_CONFIG['min_lr'],
            verbose=1
        )
    ]
    return callbacks


def train_model(model, x_train, y_train, datagen=None,
                validation_split=0.1, x_val=None, y_val=None,
                batch_size=DEFAULT_CONFIG['batch_size'],
                epochs=DEFAULT_CONFIG['epochs'],
                callbacks=None):
    """
    Train a Keras model with optional data augmentation.

    If `datagen` is provided, training uses the augmentation generator.
    Otherwise, trains directly on the raw data.

    Args:
        model: Compiled Keras model.
        x_train: Training images array.
        y_train: Training labels array (one-hot).
        datagen: Optional ImageDataGenerator for data augmentation.
        validation_split: Fraction of training data for validation (used if x_val is None).
        x_val: Optional separate validation images.
        y_val: Optional separate validation labels.
        batch_size (int): Training batch size.
        epochs (int): Maximum number of training epochs.
        callbacks (list): List of Keras callbacks.

    Returns:
        keras.callbacks.History: Training history object.
    """
    # Determine validation data
    if x_val is not None and y_val is not None:
        validation_data = (x_val, y_val)
    else:
        # Split training data for validation
        split_idx = int(len(x_train) * (1 - validation_split))
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        validation_data = (x_val, y_val)

    if datagen is not None:
        # Train with data augmentation
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks or [],
            steps_per_epoch=len(x_train) // batch_size,
            verbose=1
        )
    else:
        # Train without augmentation
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks or [],
            verbose=1
        )

    return history
