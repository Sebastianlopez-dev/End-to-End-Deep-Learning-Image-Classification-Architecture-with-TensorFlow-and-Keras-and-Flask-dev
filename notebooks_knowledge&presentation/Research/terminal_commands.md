# Important Terminal Commands

This document contains a summary of the most important terminal commands for the CIFAR-10 Image Classification project, organized by workflow.

## 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n ironhack.nn python=3.10
conda activate ironhack.nn

# Install dependencies
pip install -r requirements.txt
```

## 2. Training the Model

```bash
# Run the training notebook (or script)
jupyter notebook notebooks/

# Or run training scripts directly
python src/train.py          # if you have a training script
```

## 3. Running the Flask Web App

```bash
# Activate environment first
conda activate ironhack.nn

# Start the Flask server
python app/app.py
# or with a custom port:
python app/app.py (or flask run --port 5001)
```

## 4. Docker (Containerized Deployment)

```bash
# Build the Docker image
docker build -t cifar10-classifier .

# Run the container
docker run -p 5001:5001 cifar10-classifier
```

## 5. Git (Version Control)

```bash
# Check status of changes
git status

# Stage and commit changes
git add .
git commit -m "your message here"

# Push to remote
git push origin main
```

## 6. Useful Utility Commands

```bash
# Check installed packages
pip list

# Check Python version
python --version

# Check GPU availability (if using TensorFlow)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check model file sizes
ls -lh models/
```
