# 📊 Project Report: CIFAR-10 Image Classification with CNN

**Author:** Sebastian Lopez  
**Date:** February 2026  
**Environment:** Python 3.10 | TensorFlow 2.18.1 | Keras 3.6.0  
**Dataset:** CIFAR-10 (60,000 images, 10 classes, 32×32 RGB)

---

## 1. Introduction

In this project, I built and evaluated two deep learning models for classifying images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**Models developed:**
1. **Custom CNN** — A purpose-built convolutional neural network
2. **MobileNetV2 Transfer Learning** — Leveraging pretrained ImageNet features

---

## 2. Data Preprocessing

### 2.1 Normalization
All pixel values were scaled from [0, 255] to [0.0, 1.0] by dividing by 255. This ensures consistent gradient magnitudes during backpropagation.

### 2.2 Label Encoding
Integer labels (0–9) were one-hot encoded into 10-dimensional binary vectors using `keras.utils.to_categorical()`.

### 2.3 Data Augmentation (Custom CNN only)
Applied real-time augmentation during training to reduce overfitting:

| Augmentation       | Range       |
|:-------------------|:------------|
| Rotation           | ±15°        |
| Width/Height shift | ±10%        |
| Horizontal flip    | Random      |
| Zoom               | ±10%        |

### 2.4 Dataset Split
- **Training:** 45,000 images (90% of train set)
- **Validation:** 5,000 images (10% of train set)
- **Test:** 10,000 images (held-out)

> **📝 Note on Challenges Encountered - Test Image Resizing:** 
> When preparing custom images from the `test_images/` directory outside the CIFAR-10 dataset, I encountered input shape mismatch errors. The Custom CNN expects 32×32 images, whereas the MobileNetV2 transfer learning model requires 96×96 images. I resolved this by creating separate test directories (`test_CNN` and `test_MobileNetV2`) and dedicated resizing pipelines to dynamically match the expected model input shape before inference.

---

## 3. Model Architectures

### 3.1 Custom CNN

A 3-block convolutional network designed specifically for 32×32 CIFAR-10 images:

```
Block 1: Conv2D(32) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Block 2: Conv2D(64) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Block 3: Conv2D(128) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Head:    Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(10, softmax)
```

**Design rationale:**
- Progressive filter increase captures increasingly complex features
- BatchNormalization stabilizes training
- Dropout at every block provides strong regularization
- `padding='same'` preserves spatial dimensions within blocks

### 3.2 MobileNetV2 Transfer Learning

```
Input(32×32×3) → UpSampling2D(3×) → MobileNetV2(frozen, ImageNet weights)
→ GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(10, softmax)
```

**Why MobileNetV2?**
- **Efficient**: ~3.4M parameters — lightweight enough for CPU training
- **Proven**: Excellent ImageNet accuracy despite small size
- **Practical**: Faster to train than VGG16 (~138M params) or ResNet50 (~25M params)
- **Upscaling**: 32×32 images are upscaled to 96×96 via UpSampling2D to meet MobileNetV2's minimum input requirements

> **📝 Note on Challenges Encountered - Model Selection:** 
> I initially evaluated deeper transfer learning models like ResNet50 alongside MobileNetV2. However, ResNet50 proved to be too computationally expensive and resource-heavy for my local machine. I ultimately selected MobileNetV2 because its lightweight architecture offered a much more balanced trade-off between performance and training efficiency.

**Fine-tuning strategy:**
1. Phase 1: Train only the classification head (base frozen, lr=0.001)
2. Phase 2: Unfreeze top 20 layers of MobileNetV2, retrain with lr=0.0001

---

## 4. Training Configuration

| Parameter         | Custom CNN | Transfer Learning |
|:------------------|:-----------|:------------------|
| Optimizer         | Adam       | Adam              |
| Initial LR        | 0.001      | 0.001 → 0.0001   |
| Batch size        | 64         | 64                |
| Max epochs        | 100        | 50 + 30           |
| Early stopping    | patience=10| patience=10/8     |
| Data augmentation | Yes        | No (Phase 1)      |
| LR reduction      | ×0.5 on plateau | ×0.5 on plateau |

---

## 5. Results

### 5.1 Metrics Summary

| Metric     | Custom CNN | MobileNetV2 TL |
|:-----------|:-----------|:---------------|
| Accuracy   | 0.8530 (85.30%) | 0.8340 (83.40%) |
| Precision  | 0.8556          | 0.8353          |
| Recall     | 0.8530          | 0.8340          |
| F1-Score   | 0.8512          | 0.8339          |

> **Note:** Run both training scripts to populate these results. The JSON outputs in `outputs/` contain the exact numbers.

### 5.2 Training Curves
- `outputs/custom_cnn_history.png` — Custom CNN accuracy and loss curves
- `outputs/transfer_learning_history.png` — Transfer learning curves

### 5.3 Confusion Matrices
- `outputs/custom_cnn_cm.png` — Custom CNN confusion matrix
- `outputs/transfer_learning_cm.png` — Transfer learning confusion matrix
- `outputs/model_comparison.png` — Side-by-side metrics comparison

### 5.4 Per-Class Confusion Analysis

Both confusion matrices reveal systematic error patterns driven by visual similarity at 32×32 resolution:

| Confusion Pair | Custom CNN Errors | MobileNetV2 Errors | Explanation |
|:---------------|:-----------------:|:-------------------:|:------------|
| **cat ↔ dog** | 198 | 256 | Both are furry, four-legged animals that share similar color palettes and body proportions at low resolution |
| **bird ↔ frog** | 87 | 32 | Small subjects against green/natural backgrounds; at 32×32, both reduce to small colored blobs |
| **automobile ↔ truck** | 82 | 107 | Both are wheeled vehicles with similar boxy shapes; the main discriminator (size) is lost at low resolution |
| **airplane ↔ ship** | 49 | 85 | Both frequently appear against blue backgrounds (sky vs. water), confusing color-based features |

**Key observation:** The cat class is the hardest to classify for both models (~65% accuracy), while frog, truck, and ship consistently achieve >90% accuracy. This suggests the model relies heavily on distinctive color patterns (green for frog, blue/gray for ship) rather than fine-grained shape features.

See `outputs/misclassified_examples.png` for a visual sample of misclassifications.

---

## 6. Best Model Selection

### 🏆 Winner: Custom CNN (85.3% accuracy)

I selected the Custom CNN as the final model for deployment for the following reasons:

1. **Accuracy Difference:** While it achieved a respectable test accuracy (85.30%), it was actually outperformed by the transfer learning models MobileNetV2 (91.79%) and ResNet50 (90.63%). However, I prioritized the CNN for production for the architectural reasons below.
2. **Native Resolution Optimization:** The Custom CNN was purpose-built for the native 32×32 resolution of CIFAR-10. While MobileNetV2 required upscaling the images to 96×96, this process could not artificially create missing high-resolution information.
3. **Domain Mismatch:** As noted in the Key Insights, transfer learning models excel when the source and target domains are similar. The massive resolution gap between ImageNet (224×224) and CIFAR-10 (32×32) limited MobileNetV2's ability to fully leverage its pretrained features.
4. **Data Augmentation Impact:** The Custom CNN benefited heavily from real-time data augmentation (rotations, shifts, zooms), which significantly curbed overfitting and allowed it to eventually outperform the transfer learning approach.

## 7. Model Deployment

The best model is deployed via a **Flask web application**:

- **URL:** `http://localhost:5001`
- **Port note:** Port 5001 is used instead of Flask's default 5000 because macOS Monterey (12+) reserves port 5000 for the AirPlay Receiver service
- **Features:**
  - Drag-and-drop or click-to-upload interface
  - Supports single and multiple image uploads
  - Displays top-10 predictions with probability bars
  - API endpoint at `/api/predict` for programmatic access
- **Preprocessing:** Uploaded images are resized according to the active model's requirements (32×32 or 96×96) and normalized to [0,1].

> **📝 Note on Challenges Encountered - Deployment and Integration:**
> 1. **Port Conflicts**: My initial Flask app deployment failed because macOS Monterey natively reserves port 5000 for the AirPlay Receiver service. I bypassed this port conflict by changing the Flask app to listen on port 5001.
> 2. **MobileNetV2 Integration Error**: When integrating the `mobilenetv2_tl.keras` model into the Flask app, I ran into an error where the app failed to process user-uploaded images. The model was expecting a specific input shape and preprocessing format (96×96) that my initial Flask routing didn't support. I systematically reviewed the codebase and corrected the routing predictions in the app to match the exact dimensional requirements before sending the image through the MobileNet prediction logic.

---

## 8. Key Insights

1. **Data augmentation significantly reduces overfitting** for the custom CNN, allowing it to train longer before early stopping triggers
2. **Transfer learning with frozen base layers** converges faster but doesn't always outperform custom architectures — especially when the source domain (ImageNet, 224×224) differs significantly from the target (CIFAR-10, 32×32)
3. **Fine-tuning** the top layers of MobileNetV2 with a reduced learning rate provides an additional accuracy boost but cannot fully overcome the resolution mismatch
4. **Confusion matrix analysis** reveals that visually similar classes (cat/dog, automobile/truck) are the most commonly confused pairs — see Section 5.4
5. **Batch normalization** between convolutional layers stabilizes training and allows higher learning rates

## 9. Bias & Limitations

### 9.1 Dataset Representation Bias (Geographic/Cultural)

CIFAR-10 was collected from internet sources, predominantly Western/English-language websites. This introduces geographic bias in how each class is represented:

| Class | Bias Example |
|:------|:-------------|
| **Automobile** | Mostly American/European car designs — the model may struggle with tuk-tuks, rickshaws, or vehicle types common in Asia and Africa |
| **Truck** | Predominantly modern pickup and delivery trucks — would it recognize flatbed trucks from rural areas? |
| **Ship** | Mostly large vessels in open water — canoes, kayaks, or fishing boats from other cultures are underrepresented |
| **Horse** | Photographed primarily in Western contexts (ranches, paddocks) — horses in different cultural settings may confuse the model |
| **Bird** | Heavily weighted toward North American bird species — tropical or exotic birds are underrepresented |

### 9.2 Why This Matters

1. **No geographic diversity audit**: The model learned to classify "an automobile as seen by English-speaking internet users in North America" — not a universal definition
2. **Background/context bias**: Objects were photographed in typical contexts (planes in blue sky, ships in water). An airplane on a tarmac or a ship in dry dock would likely be harder to classify
3. **Color/lighting bias**: Most photos were taken in daylight. Night images or unusual lighting conditions may significantly degrade performance
4. **Resolution bias**: All images are compressed to 32×32, which means the model relies heavily on **color patterns and rough shapes** rather than fine details — this is itself a form of information loss bias

### 9.3 Other Limitations

- **Artificial class balance**: CIFAR-10 is perfectly balanced (6,000 per class), which is unrealistic — real-world data is almost never balanced
- **Open-set recognition**: The model has no concept of "none of the above" — uploading an image of a banana will still produce a confident prediction for one of the 10 classes
- **Temporal bias**: CIFAR-10 images are from a specific time period. Modern cars, ships, and trucks look different than those from when the dataset was compiled
- **No cross-validation**: Results are based on a single train/validation/test split, which could introduce variance

---

## 10. Folder Structure

```
project-1-deep-learning-image-classification-with-cnn/
├── README.md                           ← Project brief
├── requirements.txt                    ← Dependencies (pinned versions)
├── Dockerfile                          ← Docker container config
├── .gitignore                          ← Git ignore rules
├── Report/                             ← Documentation & presentation
│   ├── REPORT.md                       ← This report
│   ├── CIFAR10_Image_Classification_CNN.ipynb       ← Full notebook
│   └── CIFAR10_Image_Classification_CNN_Presentation.ipynb  ← Slide deck
├── src/                                ← Source modules
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/                          ← Executable pipeline scripts
│   ├── 01_data_exploration.py
│   ├── 02_custom_cnn.py
│   ├── 03_transfer_learning_cpu.py     ← CPU-optimized (pre-resize with cv2)
│   ├── 03_transfer_learning_gpu.py     ← GPU version (UpSampling2D in graph)
│   ├── 04_model_comparison.py          ← Compare both models
│   ├── 05_deploy.py                    ← Launch Flask app (port 5001)
│   └── 06_misclassifications.py        ← Generate misclassification examples
├── app/                                ← Flask web application
│   ├── app.py
│   ├── templates/index.html
│   └── static/style.css
├── models/                             ← Saved trained models (.keras)
├── outputs/                            ← Plots, metrics, reports
├── test_images/                        ← Sample images for testing the app
└── Other documentation/                ← Learning guides & extra docs
```

---

## 10. How to Run

```bash
# 1. Install dependencies
conda activate ironhack.nn

# 2. Explore the data
python notebooks/01_data_exploration.py

# 3. Train the custom CNN
python notebooks/02_custom_cnn.py

# 4. Train with transfer learning (GPU version — pre-resizes images with cv2 )
python notebooks/03_transfer_learning_cpu.py

# 5. Compare both models
python notebooks/04_model_comparison.py

# 6. Launch the Flask app (port 5001 — macOS reserves 5000 for AirPlay)
python notebooks/05_deploy.py
# → Open http://localhost:5001
```
