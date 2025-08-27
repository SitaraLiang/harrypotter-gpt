"""
Using a character-level tokenizer from scratch.
"""

import os
from pathlib import Path
import pickle # for saving/loading Python objects (like dicts).
import numpy as np

base_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(base_dir)
input_dir = Path(os.path.join(parent_dir, 'clean'))
input_file_path = input_dir / "input.txt"

# Reads entire harry potter series into memory as a single string data.
with open(input_file_path, 'r') as f:
    data = f.read()

# get all the unique characters that occur in this text
# set(data) → get unique characters in the dataset.
# Convert set to sorted list → ensures consistent ordering.
chars = sorted(list(set(data)))
# vocab_size = number of unique characters (65 in this dataset).
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)] 
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
# First convert lists to numpy arrays (uint16 = 2-byte integers).
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# Then saves them to compact binary files (.bin).
# This format is much faster and smaller than storing as text.
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size, # vocab_size (needed for model initialization, embedding matrix and the last layer),
    'itos': itos, # itos & stoi (needed for encoding/decoding during training or generation).
    'stoi': stoi,
}
with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

"""
vocab size: 106
train has 5,650,993 tokens
val has 627,889 tokens
"""

