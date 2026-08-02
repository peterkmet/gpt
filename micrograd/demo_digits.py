import random
import numpy as np
from sklearn.datasets import load_digits

from micrograd.grad import Value
from micrograd.nn import MLP
from micrograd.visualize import draw_digit

np.random.seed(1337)
random.seed(1337)

# data: 1797 samples, 64 features (8x8 pixels, values 0..16), labels 0..9
digits = load_digits()
X = digits.data / 16.0  # normalize to 0..1
y = digits.target

# shuffle, then split: 500 train, rest test
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]
X_train, y_train = X[:500], y[:500]
X_test, y_test = X[500:], y[500:]

network = MLP(64, [(16, 'relu'), (10, None)])  # 64 inputs, 16 hidden, 10 class scores
print("number of parameters", len(network.parameters()))

def loss(X_batch, y_batch):
    inputs = [[Value(v) for v in xrow] for xrow in X_batch]
    scores = [network(x) for x in inputs]  # 10 class scores per sample

    # softmax + cross-entropy: -log( exp(s_correct) / sum(exp(s)) )
    losses = []
    for s, ygt in zip(scores, y_batch):
        m = max(si.data for si in s)          # shift scores so exp doesn't overflow
        exps = [(si - m).exp() for si in s]
        losses.append(-(exps[ygt] / sum(exps)).log())
    data_loss = sum(losses) / len(losses)

    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum(p*p for p in network.parameters())
    total_loss = data_loss + reg_loss

    # accuracy: predicted class = argmax of scores
    accuracy = [max(range(10), key=lambda i: s[i].data) == ygt
                for s, ygt in zip(scores, y_batch)]

    return total_loss, sum(accuracy) / len(accuracy)

# training: minibatch sgd
steps = 100
batch = 32
for k in range(steps):
    idx = np.random.randint(0, len(X_train), batch)
    total_loss, acc = loss(X_train[idx], y_train[idx])

    network.zero_grad()
    total_loss.backward()

    learning_rate = 0.5 * (1 - 0.9*k/steps)
    for p in network.parameters():
        p.data -= learning_rate * p.grad

    print(f"step {k} loss {total_loss.data:.4f}, accuracy {acc*100:.0f}%")

# test accuracy, one sample at a time so each graph gets freed
correct = 0
for xrow, ygt in zip(X_test, y_test):
    s = network([Value(v) for v in xrow])
    correct += max(range(10), key=lambda i: s[i].data) == ygt
print(f"test accuracy {correct/len(X_test)*100:.1f}% ({correct}/{len(X_test)})")

#draw_digit(X_train.images).savefig('svg/digits_all.svg')
