"""
Sample from a trained model with generalized load_encoding
"""
import os
import time
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPT
from model_configs import GPTConfig

def setup_device_and_context(device, dtype):
    """Set seeds, TF32 flags, and autocast context for given device/dtype."""
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed(int(time.time()))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return device_type, ptdtype, ctx


def load_model(init_from, out_dir, device, compile=False):
    """Load GPT model from checkpoint or Hugging Face pretrained weights."""
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
        checkpoint = None
    else:
        raise ValueError(f"Unsupported init_from: {init_from}")

    model.eval().to(device)
    if compile:
        model = torch.compile(model)
    return model, checkpoint


def load_encoding(init_from, checkpoint, tokenizer_type=None):
    """Load encoding (stoi/itos if available, otherwise GPT-2 tokenizer)."""
    # Check if meta exists from a previous dataset
    if init_from == 'resume' and checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset_dir = checkpoint['config']['dataset']
        base_path = os.path.join('data', dataset_dir)

        if tokenizer_type == "bpe":
            from tokenizers import Tokenizer
            tokenizer_file = os.path.join(base_path, "tokenizer.json")
            assert os.path.exists(tokenizer_file), f"Tokenizer file doesn't exist: {tokenizer_file}"
            tokenizer = Tokenizer.from_file(tokenizer_file)

            encode_fn = lambda s: tokenizer.encode(s).ids
            decode_fn = lambda l: tokenizer.decode(l)
            return encode_fn, decode_fn

        elif tokenizer_type == "sp":
            import sentencepiece as spm
            model_file = os.path.join(base_path, "tok5000.model")
            assert os.path.exists(model_file), f"Model file doesn't exist: {model_file}"
            sp = spm.SentencePieceProcessor()
            sp.load(model_file)

            encode_fn = lambda s: sp.encode(s, out_type=int)
            decode_fn = lambda l: sp.decode(l)
            return encode_fn, decode_fn
            
        elif tokenizer_type == "char":
            meta_path = os.path.join(base_path, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                stoi, itos = meta['stoi'], meta['itos']
                encode_fn = lambda s: [stoi[c] for c in s]
                decode_fn = lambda l: ''.join([itos[i] for i in l])
                return encode_fn, decode_fn
            else:
                raise ValueError(f"Meta.pkl doesn't exist: {meta_path}")
    
     # Fallback to GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode_fn = lambda l: enc.decode(l)
    return encode_fn, decode_fn


def prepare_prompt(start, encode, device):
    """Turn a start string (or file) into tensor of token IDs."""
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    return torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]


def generate_text(model, x, decode, ctx, max_new_tokens, temperature, top_k):
    """Generate text from model given input x."""
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            return decode(y[0].tolist())


# -------------------- MAIN --------------------
base_dir = os.path.dirname(__file__)
init_from = "resume"
out_dir = "out"
start = "\n"
max_new_tokens = 1000
temperature = 0.8
top_k = 200
device = "mps"
dtype = "float16"
compile = False
tokenizer_type = 'tiktoken'
exec(open(os.path.join(base_dir, 'configurator.py')).read()) # overrides from command line or config file

device_type, ptdtype, ctx = setup_device_and_context(device, dtype)
model, checkpoint = load_model(init_from, out_dir, device, compile)
encode, decode = load_encoding(init_from, checkpoint, tokenizer_type)
x = prepare_prompt(start, encode, device)
y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
# y[0] -> tensor([50256, 72, 95, 423, 198])
# y[0].tolist() -> [50256, 72, 95, 423, 198]
print(decode(y[0].tolist()))