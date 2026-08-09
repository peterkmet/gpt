# data preparation
words = open("names.txt","r").read().lower().splitlines()
trans = str.maketrans({
    "á":"a", "â":"a", "ä":"a", "æ":"ae",
    "ç":"c", "ć":"c", "č":"c",
    "ď":"d", "đ":"d",
    "é":"e", "ë":"e", "ě":"e",
    "í":"i", "î":"i", "ï":"i",
    "ľ":"l", "ł":"l", "ĺ":"l",
    "ň":"n",
    "ó":"o", "ô":"o", "ö":"o", "ø":"o",
    "ř":"r", "ŕ":"r",
    "ş":"s", "š":"s",
    "ť":"t",
    "ú":"u", "û":"u", "ü":"u", "ů":"u", "ű":"u",
    "ý":"y",
    "ź":"z", "ż":"z", "ž":"z",
    "*":""," ":"", "'":"","-":"","ʼ":""
})
words = [w.translate(trans) for w in words]

chars = sorted(set(''.join(words)))
#string to integer
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 # stop char
#integer to string
itos = {i:s for s, i in stoi.items()}

def build_dataset(words, block_size):
  X, Y = [], []
  for w in words:

    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append
  return X, Y