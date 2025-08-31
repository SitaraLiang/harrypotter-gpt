# train a miniature character-level harry potter model
# good for debugging and playing on macbooks and such

out_dir = 'out-harrypotter-bpe' # where to save the checkpoints & logs
eval_interval = 250 # Run evaluation every 250 optimization steps.
                    # keep frequent because we'll overfit
eval_iters = 20 # # During each evaluation, run 200 minibatches to estimate metrics.
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'harrypotter_bpe'
gradient_accumulation_steps = 4
batch_size = 12
context_length = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.01

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'
compile = False
