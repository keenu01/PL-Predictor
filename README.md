# 🧠 KMNIST CNN Classifier

A simple convolutional neural network built with PyTorch to classify Japanese Hiragana characters from the [KMNIST dataset](http://codh.rois.ac.jp/kmnist/). This project demonstrates basic image classification on a non-Latin dataset using a custom CNN.

---

## 📌 Project Overview

- 🔤 10-class classification task (Hiragana characters)
- 📈 Trained with PyTorch on CPU
- 🧪 Evaluates accuracy and visualizes correct vs. incorrect predictions
- 🖼️ Uses Matplotlib to plot prediction results

---

## 🧱 Model Summary

The CNN model is defined in `model.py` and includes:

- 5 convolutional layers with ReLU activation
- Max pooling after each conv layer
- Fully connected layers with ReLU
- `nn.LazyLinear` used for dynamic input sizing

```python
model = CNN(input_layer=1, hidden_layer=200, output_layer=10)
