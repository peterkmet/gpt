import torch
from data import words, stoi, itos

# model creation, combination of first and second char in sample data with number of occurrences
V = len(stoi)
N = torch.zeros(V,V,dtype=torch.int32)
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] +=1

P = (N+1).float() # model smoothing, every combination has at least one occurrence so that loss function is not infinity
P /= P.sum(1, keepdim=True) # normalize rows to probabilities

# get some model predictions
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
  out = []
  ix = 0
  while True:
    p = P[ix]

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # get item from a hat
    out.append(itos[ix])
    if ix == 0: # until stop character
      break
  print(''.join(out))

log_likelihood = 0.0
n=0

# loss function
# we compare real bigram occurrences from training data, with calculated probabilities to determine loss
for w in words:
  chs=['.'] + list(w) + ['.']
  for ch1, ch2 in zip (chs, chs[1:]): # we add computed likelihoods of all REAL bigrams in training data words (not simple P matrix traversal)
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2] # we test what probability model gives to these real combinations
    logprob = torch.log(prob)
    log_likelihood += logprob # error is accumulated
    n += 1

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') # normalized to not increase with n