# Changelog

## 2026-08-09

- New mlp.py: MLP language model over a 3-char context (Bengio-style, makemore part 2) — 10-dim char embeddings, 200-neuron tanh hidden layer, 20k training steps on mps (batch 16384, lr 1 decayed to 0.1 at the halfway point). Kaiming init with tanh gain 5/3 plus a near-zero output layer start the loss at 3.30 (uniform over 27 chars); final train loss 1.93, val 2.01. Samples 20 names; with 2D embeddings it also prints an ASCII scatter of the learned character space.
- data.py: lowercases the input, maps ĺ and ŕ in the diacritics table, adds build_dataset — rolling block_size context windows padded with `.`, used by mlp.py.
- names.txt: dataset grown from 11738 to 24890 names.

## 2026-08-08

- New data.py: shared preprocessing — reads names.txt, strips diacritics via a translation table, builds stoi/itos over 27 symbols (`.` + a-z).
- New bigrams.py: count-based bigram model; builds the 27x27 bigram matrix with add-one smoothing, samples names, reports negative log-likelihood (2.5228 nats per bigram).
- New bigrams_nn.py: the same bigram model as a single 27x27 linear layer trained by gradient descent (one-hot input, softmax, cross-entropy with L2), reaching a comparable loss.
- New visualize.py: graphviz diagram of the model structure — 27 input characters fully connected to 27 neurons, each neuron's logit mapped to one next-character output.
