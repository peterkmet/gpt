# Changelog

## 2026-08-08

- New data.py: shared preprocessing — reads names.txt, strips diacritics via a translation table, builds stoi/itos over 27 symbols (`.` + a-z).
- New bigrams.py: count-based bigram model; builds the 27x27 bigram matrix with add-one smoothing, samples names, reports negative log-likelihood (2.5228 nats per bigram).
- New bigrams_nn.py: the same bigram model as a single 27x27 linear layer trained by gradient descent (one-hot input, softmax, cross-entropy with L2), reaching a comparable loss.
- New visualize.py: graphviz diagram of the model structure — 27 input characters fully connected to 27 neurons, each neuron's logit mapped to one next-character output.
