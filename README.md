<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg" alt="Premier League Logo" width="150"/>
</p>

# Football Match Outcome Prediction

This project builds a binary classifier to predict football match results (Home Win or Not) using match statistics and metadata. The model is implemented with PyTorch and trained on a real-world dataset.

---

## 🧾 Dataset

- Dataset file: `2021-2022.csv`
- Features used:
  - Division (`Div`)
  - Home Team (`HomeTeam`)
  - Away Team (`AwayTeam`)
  - Home Shots on Target (`HST`)
  - Home Shots (`HS`)
  - Away Shots on Target (`AST`)
  - Away Shots (`AS`)
  - Home Corners (`HC`)
  - Away Corners (`AC`)
  - Referee (`Referee`)
- Target:
  - Full Time Result (`FTR`), converted to a binary label: Home Win = 1, else 0

---

## 🔧 Data Preprocessing

- Categorical features are label-encoded to convert text labels to numeric.
- Missing values are imputed using the mean strategy.
- Dataset is split into train/test sets (80/20).
- Features and targets are converted to PyTorch tensors.

---

## 🧠 Model Architecture

- Fully connected feed-forward neural network using PyTorch.
- 4 hidden layers, each with ReLU activation.
- Output layer uses Sigmoid activation for binary classification.
- Model input size equals number of features (10 in this case).
- Output size is 1 (probability of Home Win).

---

## ⚙️ Training Details

- Loss function: Binary Cross Entropy Loss (`BCELoss`)
- Optimizer: Adam with learning rate 0.001
- Training epochs: 100
- Accuracy calculated during training every 100 epochs.

---

## 📈 Evaluation

- After training, the model is evaluated on the test set.
- Metrics:
  - Test loss
  - Test accuracy

---

## 🚀 How to Run

1. Install dependencies

```bash
pip install torch pandas scikit-learn
