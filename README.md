KMNIST CNN Classifier 🧠🈁

A convolutional neural network (CNN) built with PyTorch to classify handwritten Japanese characters from the KMNIST dataset (a modern alternative to MNIST).
📚 Dataset

The KMNIST dataset contains 10 classes of Hiragana characters, with 60,000 training and 10,000 test images. Each image is grayscale and 28×28 in size.
🏗️ Model Architecture

Implemented in model.py:

    5 Convolutional layers with ReLU activation

    MaxPooling after each conv layer

    Flatten + 2 Linear layers

    Optional: LazyLinear for automatic input size inference

CNN(1, 200, 10)  # input_channels=1, hidden_channels=200, output_classes=10

🖥️ Training

python main.py

Key features:

    CPU-only support (runs fast on high-end CPUs like Intel Core i9)

    Progress printed per epoch

    Evaluation accuracy on the test set

    Optionally visualizes correct vs. incorrect predictions using Matplotlib

🔍 Evaluation Example

A 5×5 grid of predictions is shown after training:

    ✅ Green title: Correct prediction

    ❌ Red title: Incorrect prediction

<!-- Replace with your own image path -->
🛠 Requirements

Install dependencies with:

pip install torch torchvision matplotlib

For Windows users with SSL issues:

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

📁 File Structure

├── main.py           # Training and evaluation loop
├── model.py          # CNN model definition
├── images/           # Output plots and prediction grids
├── data/             # KMNIST dataset (downloaded automatically)
└── README.md

🧠 Credits

    Dataset by: ROIS-DS Center for Open Data in the Humanities

    Built with ❤️ using PyTorch

📜 License

MIT License – use freely and contribute if you'd like!
