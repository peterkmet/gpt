import torch
import torch.nn.functional as F
import random
import time
from data import words, itos, build_dataset

# config
device = torch.device("mps")
block_size = 3      # context length
emb_dim = 10         # size of C embedding
hidden = 200        # hidden layer size
steps = 20000       # training steps
batch_size = 16384  # examples per minibatch
lr_start = 1        # lr multiplier start
lr_decay = 0.1      # lr multiplier after decay point
decay_at = 0.5      # fraction of steps after which lr decays
vocab_size = len(itos)  # 27

print("device:       ", device)
print("block_size:   ", block_size)
print("emb_dim:      ", emb_dim)
print("hidden:       ", hidden)
print("steps:        ", steps)
print("batch_size:   ", batch_size)
print("lr_start:     ", lr_start)
print("lr_decay:     ", lr_decay)
print("decay_at:     ", decay_at)

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = map(torch.tensor, build_dataset(words[:n1], block_size))
print("training data:", Xtr.size()[0])
Xdev, Ydev = map(torch.tensor, build_dataset(words[n1:n2], block_size))
Xte, Yte = map(torch.tensor, build_dataset(words[n2:], block_size))

Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
Xte, Yte = Xte.to(device), Yte.to(device)

g = torch.Generator(device=device).manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size, emb_dim), generator=g, device=device) # embedding lookup table: char index -> emb_dim vector
W1 = torch.randn((block_size*emb_dim, hidden), generator=g, device=device) * (5/3)/(block_size*emb_dim)**0.5 # Kaiming init, tanh gain 5/3
b1 = torch.randn(hidden, generator=g, device=device) * 0.01 # small init, keep tanh in active region
W2 = torch.randn((hidden, vocab_size), generator=g, device=device) * 0.01 # near-zero logits at init -> uniform softmax
b2 = torch.zeros(vocab_size, device=device) # zero bias -> exactly uniform predictions at init
parameters = [C, W1, b1, W2, b2]
for p in parameters:
  p.requires_grad = True

start_time = time.time()

# training
for i in range(steps):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g, device=device) # create minibatch, not run on whole dataset

  # forward pass
  emb = C[Xtr[ix]] # embedding lookup: (batch_size, block_size, emb_dim)
  h = torch.tanh(emb.view(-1, block_size*emb_dim) @ W1 + b1) # concat context embeddings, hidden layer: (batch_size, hidden)
  logits = h @ W2 + b2 # output layer, raw scores per char: (batch_size, vocab_size)
  loss = F.cross_entropy(logits, Ytr[ix]) # softmax + negative log likelihood of the correct char, numerically stable

  # backward pass
  for p in parameters:
    p.grad = None # grads accumulate by default - without reset each step would mix in all previous gradients
  loss.backward() # compute for every parameter: which direction increases the loss

  # update
  lr = lr_start if i < decay_at*steps else lr_start*lr_decay # big steps early to move fast, small steps late to settle into the minimum
  for p in parameters:
    p.data += -lr * p.grad # nudge each parameter against its gradient = decrease the loss

  if i % 1000 == 0:
    print(f"{i}: {loss.item():.4f}")

end_time = time.time()

# evaluation
@torch.no_grad() # inference only: skip building the computational graph - faster and no memory wasted on grads
def evaluate(X, Y):
  emb = C[X] # whole dataset at once, not minibatches - we want the exact loss, not a noisy estimate
  h = torch.tanh(emb.view(-1, block_size*emb_dim) @ W1 + b1)
  logits = h @ W2 + b2
  return F.cross_entropy(logits, Y).item()

print(f"Training execution time: {end_time - start_time:.2f} seconds")
print("train loss", evaluate(Xtr, Ytr)) # gap between these two shows overfitting:
print("val loss", evaluate(Xdev, Ydev)) # val >> train means the model memorizes instead of generalizing

# sampling
g = torch.Generator(device=device).manual_seed(2147483647 + 10)

for _ in range(20):
  out = []
  context = [0] * block_size # same start as training data: model only ever saw words beginning after "..."
  while True:
    emb = C[torch.tensor([context], device=device)] # run the trained network on current context...
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1) # ...to get a probability for each possible next char
    ix = torch.multinomial(probs, num_samples=1, generator=g).item() # sample randomly, not argmax - argmax would print the same word 20 times
    context = context[1:] + [ix] # model sees only last block_size chars, so the new char must push the oldest one out
    if ix == 0:
      break # model decided the word is complete
    out.append(ix)
  print(''.join(itos[i] for i in out))

if emb_dim == 2: # ASCII plot only possible for 2D embeddings
  # detach() is necessary to remove the tensor from the computational graph
  # move tensor to CPU for processing
  c_cpu = C.detach().cpu()

  grid_width = 80
  grid_height = 30

  x_min = c_cpu[:, 0].min().item()
  x_max = c_cpu[:, 0].max().item()
  y_min = c_cpu[:, 1].min().item()
  y_max = c_cpu[:, 1].max().item()

  x_span = x_max - x_min if x_max > x_min else 1.0
  y_span = y_max - y_min if y_max > y_min else 1.0

  grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

  # annotate every point with corresponding character
  for i in range(c_cpu.shape[0]):
    x = c_cpu[i, 0].item()
    y = c_cpu[i, 1].item()

    col = int((x - x_min) / x_span * (grid_width - 1))
    row = int((y - y_min) / y_span * (grid_height - 1))

    # invert row index because terminal prints from top to bottom
    row = (grid_height - 1) - row

    grid[row][col] = itos[i]

  for row in grid:
    print("".join(row))