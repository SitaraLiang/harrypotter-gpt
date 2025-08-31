"""
Train a custom SentencePiece tokenizer.
"""
import os
import sentencepiece as spm
from pathlib import Path
import numpy as np
import pickle

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

options = dict(
  # input spec
  input=str(input_file_path),
  input_format="text",
  # output spec
  model_prefix=os.path.join(base_dir, "tok400"),
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=400,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(base_dir, 'tok400.model'))
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]

# Encode to token IDs
train_ids = sp.encode(train_data, out_type=int)
val_ids = sp.encode(val_data, out_type=int)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

vocab_file = os.path.join(base_dir, 'meta.pkl')
meta = {
    'vocab_size': sp.get_piece_size(),
}

print(f"vocab size: {sp.get_piece_size()}")

with open(vocab_file, 'wb') as f:
    pickle.dump(meta, f)


"""
train has 3,654,586 tokens
val has 403,328 tokens
vocab size: 400
""" 