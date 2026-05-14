import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

# here data would be taken from the SQLite database
X = torch.rand(20000, 3) # the input data
y = torch.rand(20000, 1) # the target

X_train, X_test = X[:10000], X[10001:]
y_train, y_test = y[:10000], y[10001:]

model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
print('Started learning...')
net = NeuralNet(
    module=model,
    criterion=nn.SoftMarginLoss,
    optimizer=optim.SGD,
    lr=0.01,
    max_epochs=1000,
    batch_size=1,
    train_split=None
)

net.fit(X_train, y_train)

losses = net.history[:, 'train_loss']

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    z_test = model(X_test_tensor)

    preds = torch.where(z_test > 0, 1.0, -1.0)

    acc = (preds == y_test_tensor).float().mean()

print(f"\nTest accuracy: {acc.item() * 100:.2f}%")

plt.figure(figsize=(8, 4))
plt.plot(losses, color='blue', linewidth=2)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Logistic Loss")
plt.grid(True, linestyle='--', alpha=0.6)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Test Set Predictions (10 Samples)", fontsize=16, y=1.05)
axes = axes.flatten()

sample_X = torch.tensor(X_test[:100])
sample_y_true = y_test[:100]

for x in range(100):
    print(f'Batch {x}:')
    print(f'Test result: {sample_X[x]}, correct: {sample_y_true[x]}')
    if abs(sample_X - sample_y_true) < 5:
        print(colored('Correct!', 'green'))
    else:
        print(colored('Wrong prediction!', 'red'))
    