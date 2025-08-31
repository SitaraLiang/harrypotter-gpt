"""
Build a GPT version decoder from scratch.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model_configs import GPTConfig

"""
At first I separated single-head and multi-head into different classes, each single-head module 
needs its own causal mask and softmax. Repeating this for every head introduces redundant 
computation. Separate single-head modules also allocate separate intermediate tensors for 
Q, K, V for each head. The model ran too slow even for a small harrypotter-char model.

So then I switched to CausalSelfAttention implementation, combined these two classes: 
one vectorized, batched operation for all heads -> much faster and memory-efficient.


class SingleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd // GPTConfig.n_head, bias=GPTConfig.includeBias)
        self.query = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd // GPTConfig.n_head,bias=GPTConfig.includeBias)
        self.value = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd // GPTConfig.n_head, bias=GPTConfig.includeBias)
        self.dropout = nn.Dropout(GPTConfig.dropout)
        # non-trainable tensors
        self.register_buffer('tril', torch.tril(torch.ones(GPTConfig.context_length, GPTConfig.context_length)))
                    
    
    def forward(self, x):
        _, seq_len, _ = x.shape

        # The reason why the output of a single head is (B, T, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        mask = self.tril[:seq_len, :seq_len] == 0

        # att[b,i,j] represents the raw attention score from query position i to key position j for head h in batch b.
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, T, C) @ (B, C, T) -> (B, T, T)
        att = att.masked_fill(mask, float('-inf'))# att.shape=(B,T,T)
        att = F.softmax(att, dim=-1) # dim=-1: apply softmax over the key positions for each query
        att = self.dropout(att)

        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList(
            [SingleHeadAttention() for _ in range(GPTConfig.n_head)])
        self.w_o = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd)
        self.dropout = nn.Dropout(GPTConfig.dropout)

    def forward(self, x):
        # Concatenates these tensors along the last dimension, i.e., the feature dimension
        # After concatenating: (B,T,num_heads×head_size)
        # Recall: num_heads×head_size=n_embd
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.w_o(out)
        out = self.dropout(out)

        return out

"""

class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention (QKV in one matmul, efficient GPT-style)."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # One big projection for Q, K, V (B, T, n_embd) -> (B, T, 3*n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.includeBias)
        # Output projection back to (B, T, n_embd)
        self.w_o = nn.Linear(config.n_embd, config.n_embd, bias=config.includeBias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Register a causal mask shaped (1,1,T,T), so it broadcasts over batch & heads
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.context_length, config.context_length))
                 .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Project once, then split into q, k, v
        qkv = self.c_attn(x)                           # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)                 # each (B, T, C)

        # Reshape into heads: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention + causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))  # (B, n_head, T, T)
        
        # Apply causal mask (broadcasts from (1,1,T,T))
        mask = self.tril[:, :, :T, :T] == 0
        att = att.masked_fill(mask, float('-inf'))

        # Softmax + dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v # (B, n_head, T, head_size)        

        # Recombine heads                    
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_head*head_size) == (B, T, C)

        # Final linear + dropout
        y = self.w_o(y)                                 # (B, T, C)
        y = self.resid_dropout(y)
        return y

        
class NormLayer(nn.Module):
    """ Normalization Layer """
    def __init__(self, n_embd, includeBias):
        super().__init__()
        # gamma=1, beta=0: pure normalization (zero mean, unit variance).
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if includeBias else None

    def forward(self, x):
        """
        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """
        return F.layer_norm(x, 
                            normalized_shape=self.weight.shape,
                            weight=self.weight,
                            bias=self.bias,
                            eps=1e-5) # a small constant to avoid division by zero.

class FFN(nn.Module):
    """ Position-wise Feed-Forward Networks """
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            # maps from d_model → d_ff
            nn.Linear(config.n_embd, 4*config.n_embd, bias=config.includeBias),
            
            # GELU = Gaussian Error Linear Unit.
            nn.GELU(),
            
            # maps from d_ff → d_model
            nn.Linear(4*config.n_embd, config.n_embd, bias=config.includeBias),

            # Residual Dropout
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        """
        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """

        # No residual conenction here. It is conceptually part of the Transformer block, 
        # not part of the FFN itself.
        return self.ffn(x)

class Block(nn.Module):
    """ A single layer inside the model architecture which will be applied N times"""
    def __init__(self, config):
        super().__init__()
        self.norm1 = NormLayer(config.n_embd, config.includeBias)
        self.att = CausalSelfAttention(config)
        self.norm2 = NormLayer(config.n_embd, config.includeBias)
        self.ffn = FFN(config)

    def forward(self, x):
        """
        Add residual connections.

        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """
        x = x + self.att(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
        
class GPT(nn.Module):
    """ Main architecture of a GPT-style decoder """
    def __init__(self, gptconfig):
        super().__init__()
        assert gptconfig.context_length is not None
        assert gptconfig.vocab_size is not None

        self.config = gptconfig

        self.transformer = nn.ModuleDict(dict(
            token_embd_table = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            pos_embd_table = nn.Embedding(self.config.context_length, self.config.n_embd),
            dropout = nn.Dropout(self.config.dropout),
            blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),

            # an additional layer normalization was added after the final self-attention block
            # source: GPT-2 paper
            norm = NormLayer(self.config.n_embd, self.config.includeBias),
        ))

        # Takes the final hidden states and produces logits over the vocabulary.
        # can be replaced for different tasks without changing the transformer:
        # E.g., use the same GPT backbone for language modeling, masked LM, or classification.
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=self.config.includeBias)

        # Weight tying: makes the embedding matrix and output projection share weights (saves memory, improves generalization).
        #               -> The same matrix is used for both token embeddings (input) and the final projection to logits (output).
        self.transformer.token_embd_table.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # recursively applies init_weights to all submodules in the model.
        self.apply(self.init_weights)

        # scale the weights of residual layers at initialization by a factor of 1/ N 
        # where N is the number of residual layers.
        # source: GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w_o.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def init_weights(self, module):
        """
        A seperate helper method that takes a PyTorch module (like a Linear layer, Embedding, etc.) 
        and initializes its weights in a specific way. This ensures that all weights start 
        small and centered around 0, which helps prevent exploding activations early in 
        training, while zero biases avoid introducing unintended shifts.
        Works recursively, even nested Blocks, attention layers, and embeddings get properly 
        initialized.
        Follows GPT-2’s initialization scheme.

        Args:
            module (pytorch module): a PyTorch module (like a Linear layer, Embedding, etc.)
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # torch.numel() is a function that returns the total number of elements in a tensor. 
        # It's an efficient way to get the size of the tensor, regardless of its shape or dimensionality.
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_embd_table.weight.numel() 
        return n_params
    
    def forward(self, x, y=None):
        """
        Args:
            x (tensor): shape (batch_size, seq_length), input token indices, integers in [0, vocab_size]
            y (tensor): shape (batch_size, seq_length), next-token ground truth labels, used only in training to compute cross-entropy loss.
        """
        # Grab which device (cpu or cuda) the data is on.
        device = x.device
        _, seq_len = x.shape

        # Ensure the sequence length doesn’t exceed the maximum context window.
        assert seq_len <= self.config.context_length, f"Cannot forward sequence of length {seq_len}, context length is only {self.config.context_length}"
        
        token_embd = self.transformer.token_embd_table(x) # (B, T, n_embd)
        pos = torch.arange(seq_len, dtype=torch.long, device=device) 
        pos_embd = self.transformer.pos_embd_table(pos) # (B, T, n_embd)

        # Use dropout here:
        #   1. Prevents overfitting: otherwise, the model could memorize exact word–position 
        #      combinations instead of generalizing.
        #   2. Encourages robustness: with dropout, the model learns representations that 
        #      work even if some embedding dimensions are missing.
        #   3. Regularizes the embedding space: forces the network to not rely on one 
        #      dimension too heavily.
        x = self.transformer.dropout(token_embd + pos_embd)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.norm(x) # x.shape = (B, T, n_embd)

        if y is None:
            # Case 1 - If inference (no targets):
            # Only compute logits for the last token (x[:, [-1], :]).
            # This is efficient: when generating one token at a time, you don’t need logits for earlier positions.
            # Shape = (batch, 1, vocab_size).
            # note: using list [-1] to preserve the seq_length dim
            # NOTE: even though we only output logits for the last token, the hidden state at 
            #       that last token already encodes the context of all previous tokens (thanks to self-attention).
            logits = self.lm_head(x[:, [-1], :]) # logts.shape=(B, 1, vocab_size)
            loss = None
        else:
            # Case 2 - If training (targets provided):
            # Pass the hidden states x through the final linear layer (lm_head).
            # logits.shape = (batch, time, vocab_size).
            logits = self.lm_head(x) # logts.shape=(B, T, vocab_size)

            # F.cross_entropy expects inputs of shape:
            #   input: (N, C) -> predictions for N samples across C classes.
            #   target: (N,) -> integer class labels for those N samples.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        # Always return (logits, loss);
        # During training: (logits over all positions, loss value).
        # During inference: (logits for last token, None).
        return logits, loss

    def get_optimizer(self, weight_decay, learning_rate, betas):
        """
        Create an optimizer with fine-grained control decay/non-decay
        """
        params = dict((name, param) 
            for name, param in self.named_parameters() 
            if param.requires_grad
        )
        params_decay = [p for _, p in params.items() if p.dim() >= 2]
        params_non_decay = [p for _, p in params.items() if p.dim() < 2]

        # Helps debug and confirm that weight decay is applied to the correct parameters.
        num_decay_params = sum(p.numel() for p in params_decay)
        num_nodecay_params = sum(p.numel() for p in params_non_decay)
        print(f"num decayed parameter tensors: {len(params_decay)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(params_non_decay)}, with {num_nodecay_params:,} parameters")

        optimize_params = [
            {'params': params_decay, 'weight_decay': weight_decay},
            {'params': params_non_decay, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimize_params, lr=learning_rate, betas=betas)
        
        return optimizer

    def crop_context_length(self, new_context_length):
        """ Reduce the context_length if the new_context_length < GPTConfig.context_length
            e.g. we may load the GPT2 pretrained model checkpoint (context_length 1024)
            but want to use a smaller context_length for some smaller, simpler model
        """
        assert new_context_length <= self.config.context_length, f"Error: Model maximal context length is only {self.config.context_length}"
        self.config.context_length = new_context_length

        # Recall: pos_embd_table.shape = (context_length, n_embd)
        self.transformer.pos_embd_table.weight = nn.Parameter(self.transformer.pos_embd_table.weight[:new_context_length])

        for block in self.transformer.blocks:
                if hasattr(block.attn, "tril"):
                    block.attn.tril = block.attn.tril[:new_context_length, :new_context_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load pretrained weights from Hugging Face GPT-2 or LLaMA into our local GPT class.
        Always returns a local GPT instance with weights copied over.
        """

        override_args = override_args or {}

        # Define supported models
        supported_models = {
            'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
            'llama-7b', 'llama-13b'
        }
        assert model_type in supported_models, f"Unsupported model: {model_type}"

        # -----------------
        # Handle GPT-2 family
        # -----------------
        if model_type.startswith("gpt2"):
            from transformers import GPT2LMHeadModel
            print(f"Loading pretrained weights from Hugging Face {model_type}...")

            # config mapping for GPT-2 sizes
            config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
            }[model_type]

            print("Forcing GPT-2 defaults: vocab_size=50257, block_size=1024, bias=True")
            config_args['vocab_size'] = 50257
            config_args['context_length'] = 1024
            config_args['includeBias'] = True

            # optional dropout override
            if 'dropout' in override_args:
                print(f"Overriding dropout to {override_args['dropout']}")
                config_args['dropout'] = override_args['dropout']

            # init local GPT
            config = GPTConfig(**config_args)
            model = GPT(config)

            # load Hugging Face model
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

            # local state dict
            sd = model.state_dict()
            sd_hf = model_hf.state_dict()

            # Keys in our model
            sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

            # Keys in Hugging Face model
            sd_keys_hf = [k for k in sd_hf.keys()
                        if not k.endswith('.attn.bias')
                        and not k.endswith('.attn.masked_bias')]

            # keys that need transposition
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

            # copy weights
            for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

            print(f"Loaded GPT-2 {model_type} into local GPT successfully.")
            return model

        # -----------------
        # Handle LLaMA family
        # -----------------
        elif model_type.startswith("llama"):
            from transformers import LlamaForCausalLM
            print(f"Loading pretrained weights from Hugging Face {model_type}...")

            # config mapping (simplified; you can extend with exact values)
            config_args = {
                'llama-7b':  dict(n_layer=32, n_head=32, n_embd=4096),
                'llama-13b': dict(n_layer=40, n_head=40, n_embd=5120),
            }[model_type]

            # Force LLaMA defaults
            config_args['bias'] = None  # LLaMA doesn’t use bias
            config_args['block_size'] = 2048  # typical context length, adjust as needed

            # allow dropout override
            if 'dropout' in override_args:
                print(f"Overriding dropout to {override_args['dropout']}")
                config_args['dropout'] = override_args['dropout']

            # init local GPT
            config = GPTConfig(**config_args)
            model = GPT(config)

            # load Hugging Face model
            model_hf = LlamaForCausalLM.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

            # local state dict
            sd = model.state_dict()
            sd_keys = [k for k in sd.keys()]
            sd_keys_hf = [k for k in sd_hf.keys()]

            # NOTE: LLaMA weight mapping differs; you’ll need a mapping function here.
            # Example stub:
            mapping = {hf_k: local_k for hf_k, local_k in zip(sd_keys_hf, sd_keys)}

            for hf_k, local_k in mapping.items():
                assert sd_hf[hf_k].shape == sd[local_k].shape, f"Shape mismatch for {hf_k} vs {local_k}"
                with torch.no_grad():
                    sd[local_k].copy_(sd_hf[hf_k])

            print(f"Loaded LLaMA {model_type} into local GPT successfully.")
            return model


    @torch.no_grad() # not to track gradients inside this function
    def generate(self, context, max_new_tokens, temperature=1.0, top_k=0):
        """
        Auto-regressive decoding, predict one token at a time, feed it back, repeat.

        Args: 
            context (tensor): shape (batch_size, seq_length) input tensor of token IDs.
            max_new_tokens: how many new tokens to generates.
            temperature: controls randomness in sampling:
                - 1.0: normal randomness.
                - < 1: makes distribution sharper (less random).
                - > 1: makes distribution flatter (more random).
            top_k: optional. If set, restricts sampling only to the top k most likely tokens.
        """
        assert max_new_tokens > 0 and top_k >= 0 and temperature != 0

        n_token = 0

        while True:
            if n_token == max_new_tokens:
                break
            
            start = context.shape[1] - self.config.context_length
            current_context = context if context.shape[1] <= self.config.context_length else context[:, start:]

            # We don't care about loss
            logits, loss = self(current_context)

            # Only need the logits for the last token in the sequence
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                # extracts the top-k logits per batch row.
                # Recall: logts.shape=(B, T, vocab_size)
                v, pos = torch.topk(logits, min(top_k, logits.shape[-1]))
                # With v[:, [-1]] -> shape [2,1], it broadcasts correctly across the vocab dimension, masking each row properly.
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax over vocab dimension.
            probs = F.softmax(logits, dim=-1)

            # torch.multinomial(probs, num_samples=1): Randomly pick 1 token ID according to the probability distribution in probs.
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)

            n_token += 1

        # tensor of shape [1, sequence_length]
        return context
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Source: NanoGPT

        This function estimates model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS:
        - How many FLOPs the model needs per iteration.
        - How many FLOPs per second it achieves in practice.
        - Divides by the GPU’s theoretical max (312 TFLOPS on A100).
        - This gives MFU (Model FLOPs Utilization), a standard metric for training efficiency.

        fwdbwd_per_iter: how many forward+backward passes happen in one training iteration. (Sometimes training steps are broken into micro-batches → multiple fwdbwd per iter).
        dt: time (in seconds) it took to complete one iteration.
        Returns: an estimate of Model FLOPs Utilization (MFU), i.e., what fraction of the GPU’s max compute was used.
        """

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        # Calls a helper method that counts all trainable parameters in the model.
        # Denoted as N (number of parameters).
        N = self.get_num_params()

        # Extracts model hyperparameters from config:
        #   - L: number of Transformer layers.
        #   - H: number of attention heads.
        #   - Q: head dimension (embedding_size ÷ number_of_heads).
        #   - T: block size (sequence length).
        # So now we know the architecture shape.
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.context_length

        # This formula is from the PaLM paper (Appendix B). Rough breakdown:
        # 6*N: FLOPs to update weights per token (forward + backward passes for parameters).
        # 12*L*H*Q*T: FLOPs for multi-head self-attention and feed-forward computations (depends on layers, heads, head size, and sequence length).
        # So this gives how many FLOPs are needed for processing a single token.
        flops_per_token = 6*N + 12*L*H*Q*T

        # Each forward+backward pass processes a sequence of T tokens.
        # Multiply per-token FLOPs by T.
        flops_per_fwdbwd = flops_per_token * T

        # If we do multiple forward+backward passes per iteration (micro-batching), scale up accordingly.
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of A100 bfloat16 peak flops
        # FLOPs per iteration ÷ time per iteration = FLOPs achieved per second.
        flops_achieved = flops_per_iter * (1.0/dt)
        # NVIDIA A100 GPU can theoretically do 312 trillion FLOPs per second (TFLOPS) in bfloat16.
        # This is the “ceiling” we compare against.
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS

        # Ratio of achieved compute throughput vs theoretical max.
        # Example: If the model is doing 150 TFLOPS on A100, MFU = 150 / 312 ≈ 0.48 (48% utilization).
        mfu = flops_achieved / flops_promised

        # Returns a float between 0 and 1 (0–100% GPU compute efficiency).
        return mfu