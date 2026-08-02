# Changelog

## 2026-08-02

- Per-layer activation functions: `MLP(3, [(4, 'tanh'), (1, None)])`; Neuron and Layer take an activation name ('relu' | 'tanh' | None) instead of hardcoded tanh.
- grad.py: added `relu()`, `__rsub__`, `__rtruediv__`.
- nn.py: added `parameters()` and `zero_grad()`.
- demo_nn.py: training loop (40 steps, decaying learning rate) instead of a single forward/backward pass.
- New demo_moon.py: moon dataset classification with max-margin loss, L2 regularization, and a decision-boundary plot.
- New TODO.md with planned exercises.
- Moved grad.py, nn.py, and visualize.py into the micrograd/ package.
- Demos import from the package; nn.py imports Value via a relative import.
- grad.py: added `log()`; backward() builds the topological order iteratively (the recursive version hit Python's recursion limit on graphs with >1000 chained nodes).
- visualize.py: new `draw_digit()` tiles one or more 8x8 digit images into a square grid and returns the figure.
- New demo_digits.py: digit classification on the sklearn 8x8 digits dataset with softmax + cross-entropy loss and minibatch SGD; 79.5% test accuracy (200 training samples, 100 steps).
- New demo_digits_torch.py: the same model and loss in PyTorch; 95.4% test accuracy (1000 training samples, 1000 steps), runtime under a second versus minutes.
- test_engine.py: test_tanh_exp also covers `log()` against torch.
