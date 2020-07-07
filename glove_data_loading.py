import numpy as np
import pickle

words = []
idx = 0
word2idx = {}

glove_path = 'data'
vectors = np.zeros(shape=(400001,50))

with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        vectors[idx] = np.array(line[1:]).astype(np.float)
        idx += 1
    

pickle.dump(vectors,  open(f'{glove_path}/6B.50_vectors.pkl', 'wb'))
pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))