# micrograd

A scalar autograd engine and a small neural network library built on top of it,
following Andrej Karpathy's micrograd lecture. Exercise code, not a package for reuse.

## Layout

- `micrograd/grad.py` — `Value`: a scalar that records the operations applied to it
  (`+`, `*`, `**`, `tanh`, `relu`, `exp`, ...) and computes gradients for all
  inputs via `backward()` (reverse-mode autodiff over the recorded graph).
- `micrograd/nn.py` — `Neuron`, `Layer`, `MLP` built from `Value` objects.
  Activation is set per layer: `MLP(2, [(16, 'relu'), (16, 'relu'), (1, None)])`.
- `micrograd/visualize.py` — renders the computation graph and network diagrams
  to SVG via graphviz.

## Demos

- `demo_grad.py` — builds a single-neuron expression, runs `backward()`, and
  draws the computation graph with data and gradient at each node.
- `demo_grad_torch.py` — the same expression in PyTorch; gradients match the
  `Value` implementation.
- `demo_nn.py` — trains a 3-input MLP on four data points with squared-error
  loss (40 steps, decaying learning rate).
- `demo_moon.py` — classifies the sklearn moons dataset with a 2-16-16-1 MLP,
  max-margin loss and L2 regularization, and plots the decision boundary.

SVG output goes to `svg/` (git-ignored).

## Running

Requires Python 3.12+ with graphviz, numpy, matplotlib, scikit-learn, and torch
(torch only for `demo_grad_torch.py`).

    python demo_moon.py

## Notes

Everything is scalar: one `Value` object per number, so a forward pass through
a 337-parameter network builds a graph of a few thousand nodes. Fine for these
demos, and the reason real frameworks use tensors. Planned exercises are in
`TODO.md`.
