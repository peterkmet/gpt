import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

np.random.seed(1337)
torch.manual_seed(1337)

# data: same normalization and split as demo_digits.py (same np seed -> same permutation)
digits = load_digits()
X = digits.data / 16.0
y = digits.target

perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]
X_train = torch.tensor(X[:1000], dtype=torch.float32)
y_train = torch.tensor(y[:1000])
X_test = torch.tensor(X[1000:], dtype=torch.float32)
y_test = torch.tensor(y[1000:])

#network = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
network = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
print("number of parameters", sum(p.numel() for p in network.parameters()))

loss_fn = nn.CrossEntropyLoss()  # softmax + cross-entropy in one

steps = 1000
batch = 32
for k in range(steps):
    idx = torch.randint(0, len(X_train), (batch,))
    scores = network(X_train[idx])
    total_loss = loss_fn(scores, y_train[idx])

    network.zero_grad()
    total_loss.backward()

    learning_rate = 0.5 * (1 - 0.9*k/steps)
    with torch.no_grad():
        for p in network.parameters():
            p -= learning_rate * (p.grad + 2e-4 * p)  # 2e-4*p = grad of alpha*sum(p^2), alpha=1e-4

    acc = (scores.argmax(dim=1) == y_train[idx]).float().mean().item()
    print(f"step {k} loss {total_loss.item():.4f}, accuracy {acc*100:.0f}%")

with torch.no_grad():
    pred_train = network(X_train).argmax(dim=1)
    pred = network(X_test).argmax(dim=1)
print(f"train accuracy {(pred_train == y_train).float().mean()*100:.1f}%")

correct = (pred == y_test).sum().item()
print(f"test accuracy {correct/len(y_test)*100:.1f}% ({correct}/{len(y_test)})")