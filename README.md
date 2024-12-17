# SequentialTransformer: Transformer Architectures for Vision

This project explores the adaptation of transformer architectures, originally designed for Natural Language Processing (NLP), to computer vision tasks. Specifically, this implementation focuses on using transformers for image recognition, with the CIFAR-10 dataset as the primary benchmark.

## 📚 Overview

Transformers leverage attention mechanisms to process sequences effectively. In this project:
- Images are treated as sequences of pixels.
- Each pixel (or RGB triplet for color images) is embedded into a higher-dimensional space using linear layers.
- A transformer model classifies the images into their respective categories, with CIFAR-10's 10 classes as the target.

---

## 🛠️ Features

1. **Pixel-Based Sequence Input**:
   - Images (e.g., 32x32) are transformed into sequences of pixel embeddings.
   - Black-and-white images use 1-channel inputs, while color images use 3-channel RGB inputs.

2. **Transformer Architecture**:
   - Implemented using PyTorch.
   - Incorporates rotary positional embeddings for improved performance.

3. **Adaptation of NLP Techniques**:
   - Includes classification token embedding, a method inspired by BERT and other NLP transformers.

4. **Dataset Support**:
   - CIFAR-10 (both grayscale and color).
   - Expandable to MNIST and other datasets with minor adjustments.

5. **Rotary Positional Embeddings**:
   - Enhances transformer performance by replacing absolute positional encodings.

---

## 🗂️ Project Structure

```
SequentialTransformer/
├── data/                  # Dataset storage
├── models/                # Transformer and embedding code
│   ├── Rotary_Embedding_torch.py
│   ├── SimpleTransformer.py
├── checkpoint/            # Model checkpoints
├── Utils.py               # Utilities for data loading and preprocessing
└── SequentialTransformerMain.py
```

---

## ⚙️ Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/SequentialTransformer.git
   cd SequentialTransformer
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision 
   ```

3. **Download CIFAR-10 Dataset**:
   The dataset is automatically downloaded when the script runs.

---

## 🚀 How to Run

1. **Train the Transformer**:
   ```bash
   python SequentialTransformerMain.py
   ```

2. **Parameters**:
   Adjust key parameters in `SequentialTransformerMain.py`, such as:
   - `BATCH_SIZE`
   - `NUM_EPOCHS`
   - `EMBEDDING_SIZE`

3. **Evaluate the Model**:
   Accuracy is computed on both validation and test datasets during training.

---

## 🧪 Experiments

1. **Baseline Accuracy**:
   - Model achieves ~57% accuracy on CIFAR-10 grayscale images.
   
2. **Enhancements**:
   - Using color images improves classification accuracy.
   - Adding a classification token and using `index 0` for classification further boosts results.

3. **Future Experiments**:
   - Modify for MNIST dataset and compare performance.
   - Test advanced positional embeddings for performance gains.

---

## ✨ Key Learnings

- Adapting transformers for vision tasks requires embedding pixel sequences into a higher-dimensional space.
- Rotary embeddings outperform traditional absolute positional encodings.
- Leveraging techniques from NLP, like classification tokens, improves performance in vision tasks.

---

## 🛡️ Acknowledgments

- Inspired by Andrej Karpathy's and Phil Wang's works on transformer architectures.
- Uses CIFAR-10 dataset for benchmarking.
- Built with PyTorch and torchvision.

---
