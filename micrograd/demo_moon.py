import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from micrograd.grad import Value
from micrograd.nn import MLP

np.random.seed(1337)
random.seed(1337)

X, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1  # make y be -1 or 1

print(X[:5], y[:5])

network = MLP(2, [(16, 'relu'), (16, 'relu'), (1, None)])  # 2 inputs, two hidden layers of 16, 1 output
print("number of parameters", len(network.parameters()))

def loss():
    inputs = [[Value(v) for v in xrow] for xrow in X] # create Value objects for every coordinate
    ypred = [network(x) for x in inputs] # run network to get predictions 

    # svm "max-margin" loss
    losses = [(1 - ygt*yout).relu() for ygt, yout in zip(y, ypred)] # margin ygt*yout: <0 wrong side, 0..1 correct but close, >=1 loss 0
    data_loss = sum(losses) / len(losses) # AVG composed of Value supported operations

    # L2 regularization, penalty for large parameters (w, b)
    alpha = 1e-4
    reg_loss = alpha * sum(p*p for p in network.parameters())
    total_loss = data_loss + reg_loss

    # accuracy: fraction of points with correct sign
    accuracy = [(ygt > 0) == (yout.data > 0) for ygt, yout in zip(y, ypred)]

    return total_loss, sum(accuracy) / len(accuracy)

for k in range(100):

    # forward
    total_loss, acc = loss()

    # backward
    network.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in network.parameters():
        p.data -= learning_rate * p.grad

    print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [[Value(v) for v in xrow] for xrow in Xmesh]
scores = [network(x) for x in inputs]
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig('svg/moon_boundary.svg')