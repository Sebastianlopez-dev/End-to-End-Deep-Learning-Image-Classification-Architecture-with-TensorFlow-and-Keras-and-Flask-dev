# Functions Stored in `src` Folder

The `src` folder contains **14 functions** distributed across 4 Python modules. These functions handle data processing, model building, training, and evaluation, allowing the notebooks to remain clean and focused on high-level workflows.

Here is the detailed list of functions by file:

## 1. `data_loader.py` (4 functions)
Handles loading and preprocessing the CIFAR-10 dataset.
- `load_cifar10_data()`: Loads the CIFAR-10 dataset from Keras and returns raw train/test splits.
- `preprocess_data(x_train, y_train, x_test, y_test)`: Normalizes pixel values to [0, 1] and one-hot encodes the labels.
- `create_data_augmentation_generator()`: Creates an `ImageDataGenerator` configured with augmentation transformations for training.
- `get_prepared_data()`: Full data pipeline that loads CIFAR-10, preprocesses it, and returns a dictionary with the ready-to-use data and generator.

## 2. `model_builder.py` (3 functions)
Defines the neural network architectures used in the project.
- `build_custom_cnn(input_shape, num_classes)`: Builds and returns a custom Convolutional Neural Network (CNN) architecture designed for 32x32 images.
- `build_transfer_learning_model(input_shape, num_classes, upscale)`: Builds a transfer learning model using MobileNetV2 pretrained on ImageNet.
- `get_model_summary(model)`: Returns a string representation of the Keras model architecture summary.

## 3. `train.py` (3 functions)
Provides pipelines to compile and train models while preventing overfitting.
- `compile_model(model, learning_rate)`: Compiles a Keras model with the Adam optimizer and categorical crossentropy loss.
- `get_callbacks(model_save_path, patience)`: Creates a list of training callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) to ensure robust training.
- `train_model(model, x_train, y_train, ...)`: Trains a compiled Keras model, supporting optional data augmentation and dynamic validation splitting.

## 4. `evaluate.py` (4 functions)
Provides utilities to evaluate trained models and visualize results.
- `evaluate_model(model, x_test, y_test)`: Evaluates a model on the test set, returning metrics like loss, accuracy, precision, recall, and a full classification report.
- `plot_confusion_matrix(y_true, y_pred, ...)`: Plots and optionally saves a confusion matrix heatmap.
- `plot_training_history(history, ...)`: Plots and optionally saves the training and validation accuracy/loss curves over epochs.
- `print_evaluation_summary(metrics, model_name)`: Prints a nicely formatted summary of the model's evaluation metrics.
