---
title: "End-to-End Deep Learning: Image Classification Architecture and Flask Integration"
emoji: 🌖
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: An end-to-end computer vision project featuring custom CNN architectures and professional Flask deployment.
---

# 📊 End-to-End Deep Learning: Image Classification Architecture and Flask Integration

**Author:** Sebastian Lopez  
**Environment:** Python 3.10 | TensorFlow 2.18.1 | Keras 3.6.0  
**Dataset:** CIFAR-10 (60,000 images, 10 classes, 32×32 RGB)

---

## 🚀 Project Overview

In this project, I engineered and evaluated several deep learning architectures to classify images into 10 distinct categories. The goal was to compare a **Custom CNN** built from scratch against state-of-the-art **Transfer Learning** models like **MobileNetV2** and **ResNet50**.

### 🔗 Quick Links
- **Deployed App**: [Hugging Face Space](https://huggingface.co/spaces/basstianlopez/CIFAR-10-Image-Classification) (Docker)
- **Original Brief**: [IRONHACK_BRIEF.md](./IRONHACK_BRIEF.md)
- **Source Code**: [`src/`](./src/)

---

## 🔬 The Scientific Approach

### 1. Data Analysis & Preprocessing
Before modeling, I addressed the inherent biases in the CIFAR-10 dataset (60,000 balanced images).
- **Normalization**: Scaled pixel values to [0.0, 1.0].
- **Data Augmentation**: For the Custom CNN, I implemented real-time augmentation (rotations, shifts, zooms) to force the model to learn invariant features and prevent overfitting.

### 2. Model Architectures & Comparison
I benchmarked three distinct architectures to find the optimal balance between accuracy and reliability:

#### 🟢 **Custom CNN**
A 3-block convolutional network (Conv2D × 2 → BatchNorm → MaxPool → Dropout) specifically designed for 32×32 resolution.
- **Accuracy**: 85.30%
- **Strength**: Highly reliable on native low-resolution shapes without artificial upscaling.

#### 🟡 **MobileNetV2 (Transfer Learning)**
Selected for its efficiency and speed, leveraging pretrained ImageNet features.
- **Accuracy**: 91.79%
- **Challenge**: Requires upscaling 32×32 images to 96×96, which can introduce blur-based confusion.

#### 🔴 **ResNet50 (Transfer Learning)**
The deepest architecture tested, used to extract highly complex spatial patterns.
- **Accuracy**: 90.63%
- **Challenge**: High memory footprint and similar upscaling sensitivities as MobileNet.

> **💡 Final Model Selection**: Despite mathematically higher raw accuracy from MobileNetV2, I selected the **Custom CNN** for production. It showed superior reliability on difficult edge cases because it processes the data at its native resolution.

![Model Selection Criteria](./outputs/model_comparison_mobilenetv2_cnn.png)
*Figure 1. Comparison demonstrates that while MobileNetV2 has higher raw accuracy, the Custom CNN is more stable across resolution-sensitive inputs.*

---

## 📈 Performance Analysis

| Metric | Custom CNN | MobileNetV2 TL | ResNet50 TL |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 85.30% | **91.79%** | 90.63% |

![Confusion Matrix Comparison](./outputs/transfer_learning_cm.png)
*Figure 2. Visual analysis of misclassifications (Cats ↔ Dogs) helped drive the decision towards the more 'shape-aware' Custom CNN.*

### Key Confusion Clusters
Visible similarities at 32x32 resolution drive most errors:
- **Cat ↔ Dog**: Furry features are indistinguishable at low res.
- **Airplane ↔ Ship**: Often confused due to similar blue backgrounds (sky vs. water).

---

## 🛠️ Deployment & Engineering
The final model is deployed via a **Flask web application** containerized with **Docker**.

- **Hugging Face Environment**: Optimized to run the lightweight Custom CNN to satisfy free-tier memory constraints.
- **Port Management**: Configured to bypass macOS native port conflicts (running on 5001 locally).

---

## ⚖️ Ethics & Limitations (Project Insight)
A critical part of this project was auditing the **Geographic and Cultural bias** in the dataset:
- **Representation Bias**: Vehicles like tuk-tuks or rural flatbed trucks are underrepresented compared to Western designs.
- **Open-Set Limitations**: The model assumes every image belongs to one of its 10 classes (e.g., an apple would be forced into a "Frog" or "Truck" category).

---

## 📖 How to Explore
1. **Source Code**: Check the [`src/`](./src/) directory for modularized Python scripts.
2. **Notebooks**: View the experimental journey in [`notebooks_knowledge&presentation/`](./notebooks_knowledge&presentation/).
3. **Report**: Detailed analysis is available in the full [`Report.md`](./Report.md).

---
*Developed during Ironhack's IA engineering Bootcamp - 2026*
