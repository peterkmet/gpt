# TODO

- Batch processing in demo_moon.py, like Karpathy's demo.ipynb: `loss(batch_size)` trains on a random subset per step instead of the full dataset.
- Handwritten digit classification demo. Candidate dataset: sklearn `load_digits` (8x8, 64 inputs); full MNIST (28x28) is likely too slow for scalar micrograd.
