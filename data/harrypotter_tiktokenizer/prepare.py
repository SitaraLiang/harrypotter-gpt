"""
Using tiktoken (GPT-2 byte-level BPE) for fine tuning the GPT2 pre-trained model
"""
import os
from pathlib import Path
import numpy as np
import tiktoken # OpenAI’s tokenizer library (used in GPT models). 
                # It splits text into subword tokens using Byte Pair Encoding (BPE).

base_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(base_dir)
input_dir = Path(os.path.join(parent_dir, 'clean'))
input_file_path = input_dir / "input.txt"

# Reads entire harry potter series into memory as a single string data.
with open(input_file_path, 'r') as f:
    data = f.read() # data: a single string

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Loads GPT-2’s BPE tokenizer. This tokenizer splits text into subword tokens (not just characters).
enc = tiktoken.get_encoding("gpt2")

# Converts train and val strings into lists of integers (token IDs).
# Each token is an integer from GPT-2’s vocab (size = 50,257).
# Example: enc.encode_ordinary("Hello world!") might return: [15496, 995, 0] (depends on GPT-2’s vocab)
# These IDs are what the model will actually see.
# Note: Notice that tokens < characters. That’s because BPE merges common chunks of text 
# (e.g., "the", "ing") into single tokens.
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
# Uses uint16 (16-bit unsigned integers):
# GPT-2 vocab = 50,257 < 65,535 (fits in 16 bits).
# Saves half the memory compared to int32.
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

"""
train has 1,500,234 tokens
val has 167,297 tokens
"""