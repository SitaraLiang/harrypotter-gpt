"""
Train a custom Hugging Face BPE tokenizer.
"""

import os
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

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

# Initialize a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Pre-tokenizer
tokenizer.pre_tokenizer = ByteLevel()


# Trainer configuration
trainer = BpeTrainer(
    vocab_size=5000,  # adjust depending on desired size
    min_frequency=2,  # ignore rare tokens,
    show_progress=True,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train the tokenizer
tokenizer.train([str(input_file_path)], trainer)

tokenizer.decoder = ByteLevelDecoder()

# Encode train/val
train_ids = tokenizer.encode(train_data).ids
val_ids = tokenizer.encode(val_data).ids
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(base_dir, 'train.bin')) 
val_ids.tofile(os.path.join(base_dir, 'val.bin')) 

meta = {
    "vocab_size": tokenizer.get_vocab_size()
}

print(f"vocab size : {tokenizer.get_vocab_size()}")

with open(os.path.join(base_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# Save tokenizer object itself for later use
tokenizer.save(os.path.join(base_dir, "tokenizer.json"))

"""
train has 1,406,003 tokens
val has 156,821 tokens
vocab size : 5000
"""