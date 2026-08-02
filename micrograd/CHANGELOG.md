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
