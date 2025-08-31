"""
A training script can be run on a single GPU or MPs or CPU.
Example:
$ python train.py --batch_size=32 --compile=False
"""
from contextlib import nullcontext
import os
import pickle
import time
import numpy as np
import torch
from model import GPT
from model_configs import GPTConfig
import math

def init_model(model_args, init_from, out_dir=None, device='cpu', dropout=None, context_length=None):
    """
    Initialize a GPT or LLaMA model either from scratch, a checkpoint, or pretrained Hugging Face weights.

    Args:
        model_args (dict): Hyperparameters for GPTConfig.
        init_from (str): 'scratch', 'resume', or Hugging Face model name like 'gpt2', 'llama-7b'.
        out_dir (str, optional): Directory to load checkpoint from if init_from=='resume'.
        device (str): 'cpu' or 'cuda'.
        dropout (float, optional): Dropout override.
        context_length (int, optional): Override context length.

    Returns:
        model (nn.Module): Initialized GPT model.
        checkpoint (dict or None): Loaded checkpoint if resuming, else None.
    """
    checkpoint = None

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == 'resume':
        # resume training from a checkpoint
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']

        # Force core model attributes to match checkpoint
        for k in ['n_layer', 'n_head', 'n_embd', 'context_length', 'includeBias', 'vocab_size']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']

        # Strip unwanted prefixes if needed (NanoGPT convention)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Load weights
        model.load_state_dict(state_dict)

    elif init_from.startswith('gpt2') or init_from.startswith('llama'):
        print(f"Initializing from pretrained Hugging Face weights: {init_from}")

        override_args = {}
        if dropout is not None:
            override_args['dropout'] = dropout

        # Always returns a local GPT instance with pretrained weights
        model = GPT.from_pretrained(init_from, override_args)

        # Sync model_args with the model config
        for k, v in vars(model.config).items():
            model_args[k] = v

    else:
        raise ValueError(f"Unsupported init_from value: {init_from}")

    # Crop model block size if smaller than default
    if context_length is not None and context_length < model.config.context_length:
        print(f"Cropping context length to {context_length}")
        model.crop_context_length(context_length)
        model_args['context_length'] = context_length

    return model, checkpoint


def get_batch(split, data_dir, context_length, batch_size, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # np.memmap maps a file (or part of a file) into memory, so you can treat it like a regular NumPy array.
    # Only the parts of the file you actually access are read into RAM.
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Sample batch_size random start indices so that start+context_length stays in range.
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    # For each start index i, take a context window of context_length -> input tokens x.
    # Cast to int64 to match PyTorch’s LongTensor (token IDs type).
    # Stack to shape (batch_size, context_length).
    x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
    # Targets y are shifted by 1 (next-token prediction). Same shape.
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # Return a batch: x (inputs), y (labels), both (B=batch_size, T=context_length).
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters):
    """ Estimates an arbitrarily accurate loss over either split using many batches """
    
    # Dict to store results
    out = {} 

    # Set model to eval mode (disables dropout, uses eval behavior for LayerNorm/Dropout).
    model.eval() 

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, context_length, batch_size, device)
            with ctx: # runs under autocast (bf16/fp16) if CUDA; CPU uses nullcontext.
                logits, loss = model(X, Y) # loss is a float tensor
            losses[k] = loss.item() # Collect scalar loss values (loss.item() detaches and moves to host).
        out[split] = losses.mean() # Average them for a stable estimate (out[split]).

    # Restore train mode
    model.train()

    # return mean losses for both splits.
    return out

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters):
    """ learning rate decay scheduler (cosine with warmup) """

    # Case 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        # Start small, linearly ramp to learning_rate over warmup_iters steps (use +1 to avoid 0 exactly).
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Case 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # Case 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    
    # Cosine schedule from learning_rate → min_lr across the decay window.
    return min_lr + coeff * (learning_rate - min_lr)


# ----------- Default configs -----------
base_dir = os.path.dirname(__file__)
out_dir = "out" # where to save checkpoints, logs, etc.
eval_interval = 2000 # Run evaluation every 2000 optimization steps.
log_interval = 1 # Print training logs every step (very chatty).
eval_iters = 200 # During each evaluation, run 200 minibatches to estimate metrics.
eval_only = False # If True, run a single eval then exit (useful for sanity checks on a checkpoint).
always_save_checkpoint = False # if True, always save a checkpoint after each eval (even if the metric didn’t improve)
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' or 'llama*'

# ----------- Model -----------
n_layer = 12  # 12 transformer blocks
n_head = 12   # 12 heads
n_embd = 768  # 768 embedding dim
head_size = n_embd // n_head
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
includeBias = False # do we use bias inside LayerNorm and Linear layers? Set False -> a common simplification in some GPT variants

# ----------- Optimizer (AdamW) & training length -----------
# Classic GPT-2-ish schedule: high LR with decay; Adam betas; weight decay; 
#                             clip gradients to 1.0 to stabilize training (0 disables clipping).
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
                # If grad_clip > 0, then after backprop but before optimizer.step(), 
                # the code will rescale gradients so their global norm ≤ 1.0.

# ----------- Learning Rate decay settings -----------
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# ----------- System / dtype / compile -----------
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = 'mps'
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# use PyTorch 2.0 to compile the model to be faster
compile = True

# ----------- Data & batching -----------
dataset = 'harrypotter_bpe'
gradient_accumulation_steps = 42 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size 
                # per accumulation step (per process). If you accumulate 40 times, 
                # the effective batch per process is 12 * 40.
context_length = 1024

# Choose initialization mode: 'scratch', 'resume', 'gpt2', 'llama'
init_from = 'scratch'  

# ----------- Config ingestion and exposure -----------
# globals() = dictionary of all variables/functions in global scope.
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))] # Collect the names of all top-level simple config variables (ints/floats/bools/strings) for later export/logging.
exec(open(os.path.join(base_dir, 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # Snapshot the (possibly overridden) config into a dict for logging or checkpoint metadata.

# ----------- Tokens-per-iteration accounting -----------
tokens_per_iter = gradient_accumulation_steps * batch_size * context_length
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True) # Create the output directory only once (by rank 0).
torch.manual_seed(1337) # Set a deterministic seed per rank (1337 base + rank offset).
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul, speeds up training with minimal accuracy loss.
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn, speeds up training with minimal accuracy loss.

# ----------- Autocast / dtype plumbing -----------
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# Build a context manager for mixed-precision autocasting:
# On CPU: do nothing (no autocast).
# On CUDA: automatically cast ops to bf16/fp16 where safe for speed & memory savings.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------- poor man's data loader -----------
data_dir = os.path.join('data', dataset) # Builds the data folder path, e.g. data/openwebtext.

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# If a meta.pkl exists, load it and read vocab_size. 
# This helps match tokenizer/training artifacts.
meta_path = os.path.join(data_dir, 'meta.pkl') 
meta_vocab_size = 50304
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f: # b -> binary mode (you read raw bytes, not text).
        meta = pickle.load(f) # reads the file in binary mode and reconstructs the original Python object (could be a dict, list, model weights, etc.).
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ----------- Model init -----------
# Assemble constructor args for GPTConfig. 
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    context_length=context_length,
    includeBias=includeBias,
    vocab_size=meta_vocab_size,
    dropout=dropout
)

# Initialize model and optionally load checkpoint
model, checkpoint = init_model(
    model_args=model_args,
    init_from=init_from,
    out_dir=out_dir,        
    device=device
)

# Debug
print("model_args =", {k: v for k, v in model_args.items()})

# Now model_args contains all updated hyperparameters for logging/checkpointing
print(f"Model initialized from '{init_from}' with vocab_size={model_args['vocab_size']} and context_length={model_args['context_length']}")

# Put the model (parameters/buffers) on CPU/GPU as configured.
model.to(device)

# Restore training counters
if init_from == 'resume':
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

# initialize a GradScaler. If enabled=False scaler is a no-op
# It scales the loss to avoid fp16 underflow when backpropagating.
# Enabled only if you’re training in fp16; for bf16/fp32 it’s a no-op (bf16 typically doesn’t need scaling).
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
# Asks the model to build AdamW optimizer with proper parameter groups (decayed vs non-decayed) and any fused kernel options.
optimizer = model.get_optimizer(weight_decay, learning_rate, (beta1, beta2))
# If resuming, restore optimizer state (momentum, second moments, etc.) so training continues seamlessly.
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Drop the large checkpoint dict reference so Python can GC it and free RAM/VRAM.
checkpoint = None # free up memory

# ----------- compile the model -----------
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model # Keeps a handle to the original model (unoptimized_model) just in case you need to fall back.
    model = torch.compile(model) # requires PyTorch 2.0 
                                 # torch.compile captures/optimizes the model graph for faster execution.


# ----------- Main training loop -----------
# Adapted most of the code from NanoGPT

X, Y = get_batch('train', data_dir, context_length, batch_size, device) # fetch the very first batch
t0 = time.time()  # start a wall-clock timer
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0 # EMA of MFU; -1 means “not initialized”

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters) if decay_lr else learning_rate # Compute LR (cosine schedule + warmup if enabled)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # Set it on all optimizer param groups.

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model, eval_iters) # estimate_loss() averages many minibatches for stable estimates.
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            # iter_num > 0 avoids saving a “just-started” model at step 0.
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")

                # Save a checkpoint if: validation loss improved, or always_save_checkpoint is True.
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # If we only wanted an eval pass, exit after the first evaluation.
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx: # autocast for bf16/fp16 on CUDA, no-op on CPU
            logits, loss = model(X, Y)
            # Divide loss so that summing grads over gradient_accumulation_steps yields the mean (keeps update magnitude stable).
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', data_dir, context_length, batch_size, device)
        # backward pass, with gradient scaling if training in fp16, otherwise this acts like a normal .backward().
        # Scales the loss up before backward to avoid underflow:
        scaler.scale(loss).backward() # loss.backward(): compute/add up gradients

    # clip the gradient
    if grad_clip != 0.0:
        # bring grads back to real scale, otherwise you’d clip the scaled gradients.
        scaler.unscale_(optimizer)
        # clip global grad norm to grad_clip (e.g., 1.0) to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    # scaler.step() does two things:
    #   Unscales gradients by the same factor used in scaler.scale(loss).
    #   Checks for NaNs/Infs. If any, it skips the optimizer step to avoid corrupting weights.
    #   Calls optimizer.step() only if gradients are valid.
    # optimizer.step(): update parameters
    scaler.step(optimizer)
    scaler.update() # adjust scale factor for next iteration (fp16 safety)

    # flush the gradients as soon as we can, no need for this memory anymore
    # set_to_none=True reduces memory footprint and can be a tiny bit faster than zeroing to 0.
    optimizer.zero_grad(set_to_none=True) # free grad memory aggressively

    # timing and logging
    t1 = time.time()
    dt = t1 - t0 # time per optimizer step.
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        # multiply the last micro-loss by gradient_accumulation_steps to approximate the un-averaged total (exact would sum each micro-loss before division).
        lossf = loss.item() * gradient_accumulation_steps

        # MFU (Model FLOPs Utilization): estimate how close you are to A100 peak FLOPs; 
        # running_mfu is an exponential moving average (stabilizes readout after ~5 iters).
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1 # Global iteration counter
    local_iter_num += 1 # Per-process counter (for local warmup of stats).

    # termination conditions
    if iter_num > max_iters:
        break




