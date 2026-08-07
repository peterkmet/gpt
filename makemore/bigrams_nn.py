import torch
import torch.nn.functional as F
from data import words, stoi, itos

# create dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs=torch.tensor(xs)
ys=torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize network
g = torch.Generator().manual_seed(314)
V = len(stoi)
W = torch.randn((V,V), generator=g, requires_grad=True) # random weights

# gradient descent to update weights
for k in range(100):
  # forward pass
  xenc = F.one_hot(xs, num_classes=V).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims = True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() # loss for wrong prediction
  loss += 0.01*(W**2).mean() # loss for large non-uniform W.data
  print(loss.item())

  # backward pass
  W.grad = None
  loss.backward()

  # update
  W.data += -50 * W.grad

# sampling
for i in range(10):
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float() # one_hot for first char
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next char

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # get next char
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

loss = -probs[torch.arange(num), ys].log().mean() # final loss without regularization term
print('loss: ',loss.item())