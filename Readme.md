# 🌌 Deep Learning-Based Astronomical Object Classification

## 📄 Abstract

This project presents a deep learning-based multi-class classification system for astronomical objects using transfer learning. A convolutional neural network (EfficientNet-B0) is fine-tuned on a curated dataset of astronomical imagery to classify objects such as asteroids, asteroid belts, comets, and Kuiper belt structures. The trained model is deployed via a FastAPI-based inference service, enabling real-time prediction from user-uploaded images. The system demonstrates the effectiveness of transfer learning for domain-specific astronomical image classification.

---

## 🛰 1. Introduction

Astronomical object classification plays a crucial role in planetary science, astrophysics, and observational astronomy. With the growing availability of telescope imagery, automated classification methods are increasingly necessary.

### 🎯 Objectives

- Train a robust CNN-based classifier using transfer learning  
- Evaluate performance on a multi-class dataset  
- Deploy the model via REST API  
- Integrate structured scientific metadata for interpretability  

---

## 📊 2. Dataset

The dataset consists of labeled astronomical images organized by category:

- 🪨 `asteroids`
- 🌀 `asteroid_belt`
- ☄️ `comets`
- 🌠 `kuiper_belt`

### 🔧 Preprocessing Steps

- Aspect-ratio preserving resizing  
- Zero-padding to 224 × 224 resolution  
- RGB normalization  
- Data augmentation (horizontal flip, rotation)  

Scientific metadata is stored separately in CSV format and used for informational enrichment during inference.

---

## 🧠 3. Methodology

### 🏗 3.1 Model Architecture

The classification model is based on **EfficientNet-B0**, which employs compound scaling to balance:

- Network depth  
- Network width  
- Input resolution  

Transfer learning strategy:

1. Initialize pretrained ImageNet weights  
2. Freeze feature extraction layers  
3. Replace final classifier layer  
4. Fine-tune classification head  

---

### ⚙️ 3.2 Training Configuration

- 🐍 Framework: PyTorch  
- 🔁 Optimizer: Adam  
- 📉 Loss Function: Cross-Entropy Loss  
- 📦 Batch Size: 32  
- 🖼 Input Resolution: 224 × 224  
- 🧪 Validation Split: 80/20  
- 💻 Device: GPU (CUDA) when available  

Model weights and class mappings are saved for reproducible inference.

---

## 🏛 4. System Architecture
