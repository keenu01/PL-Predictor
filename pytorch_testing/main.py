from model import NeuralNetwork
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


path = r"C:\Users\keenu\Downloads\archive\2021-2022.csv"
data = pd.read_csv(path)


features = ["Div", "HomeTeam", "AwayTeam", "HST", "HS", "AST", "AS", "HC", "AC","Referee"]
x = data[features].copy()
y = data.FTR.copy()  # Full Time Result (e.g., H, D, A)

label_encoders = {}
for col in features:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))
    label_encoders[col] = le


x = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(x), columns=features)


y = (y == "H").astype(int)  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


x_train_tensor = t.tensor(x_train.values, dtype=t.float32)
y_train_tensor = t.tensor(y_train.values, dtype=t.float32).unsqueeze(1)

x_test_tensor = t.tensor(x_test.values, dtype=t.float32)
y_test_tensor = t.tensor(y_test.values, dtype=t.float32).unsqueeze(1)


model = NeuralNetwork(input_size=len(features), hidden_layers=30, output_size=1)
loss_fn = t.nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)


epochs = 100
for epoch in range(epochs):
    model.train()

    preds = model(x_train_tensor)
    loss = loss_fn(preds, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        acc = ((preds > 0.5) == y_train_tensor).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {acc.item() * 100:.2f}%")


model.eval()
with t.no_grad():
    test_preds = model(x_test_tensor)
    test_loss = loss_fn(test_preds, y_test_tensor)
    predicted_classes = (test_preds > 0.5).float()
    accuracy = (predicted_classes == y_test_tensor).float().mean()

print("\n=== Evaluation on Test Set ===")
print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
