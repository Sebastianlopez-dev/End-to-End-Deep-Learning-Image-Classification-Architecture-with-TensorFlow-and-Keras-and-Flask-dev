"""
Evaluation Module for CIFAR-10 Image Classification.

Provides functions to:
- Evaluate model accuracy on test data
- Generate classification reports (precision, recall, F1)
- Plot and save confusion matrices
- Plot training history curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from .data_loader import CLASS_NAMES


def evaluate_model(model, x_test, y_test):
    """
    Evaluate a trained model on the test set.

    Args:
        model: Trained Keras model.
        x_test: Test images (normalized).
        y_test: Test labels (one-hot encoded).

    Returns:
        dict: Dictionary with loss, accuracy, precision, recall, f1,
              y_true (int labels), y_pred (int predictions),
              and the full classification_report string.
    """
    # Get test loss and accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'classification_report': report
    }


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES,
                          save_path=None, title='Confusion Matrix'):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class name strings.
        save_path: Optional file path to save the plot.
        title: Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)


def plot_training_history(history, save_path=None, title_prefix=''):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras History object from model.fit().
        save_path: Optional file path to save the plot.
        title_prefix: Optional prefix for plot titles (e.g., 'Custom CNN').
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{title_prefix} Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{title_prefix} Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")

    plt.close(fig)


def print_evaluation_summary(metrics, model_name='Model'):
    """
    Print a formatted evaluation summary.

    Args:
        metrics: Dictionary returned by evaluate_model().
        model_name: Name of the model for display.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Test Loss:      {metrics['loss']:.4f}")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1-Score:       {metrics['f1_score']:.4f}")
    print(f"{'='*60}")
    print(f"\nClassification Report:\n")
    print(metrics['classification_report'])
