"""
Build a GPT version decoder from scratch.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_configs import GPTConfig

class SingleHeadAttention(nn.Module):
    """ Masked single head attention layer """
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(GPTConfig.n_embd, GPTConfig.head_size, bias=GPTConfig.includeBias)
        self.query = nn.Linear(GPTConfig.n_embd, GPTConfig.head_size,bias=GPTConfig.includeBias)
        self.value = nn.Linear(GPTConfig.n_embd, GPTConfig.head_size, bias=GPTConfig.includeBias)
        self.dropout = nn.Dropout(GPTConfig.dropout)
        # non-trainable tensors
        self.register_buffer('tril', torch.tril(torch.ones(GPTConfig.context_length, GPTConfig.context_length)))
                    
    
    def forward(self, x):
        """
        Args: 
            x (tuple): shape (batch, seq_len, n_embd). hidden_dim = n_features = n_embd
        """
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

class MultiHeadAttentionWithResidual(nn.Module):
    """ Multi-head attention layer """
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList(
            [SingleHeadAttention(GPTConfig.head_size) for _ in range(GPTConfig.n_head)])
        self.w_o = nn.Linear(GPTConfig.n_embd, GPTConfig.n_embd)
        self.dropout = nn.Dropout(GPTConfig.dropout)

    def forward(self, x):
        """
        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """

        # Concatenates these tensors along the last dimension, i.e., the feature dimension
        # After concatenating: (B,T,num_heads×head_size)
        # Recall: num_heads×head_size=n_embd
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.w_o(out)
        out = self.dropout(out)

        return out

class NormLayer(nn.Module):
    """ Normalization Layer """
    def __init__(self):
        super().__init__()
        # gamma=1, beta=0: pure normalization (zero mean, unit variance).
        self.weight = nn.Parameter(torch.ones(GPTConfig.n_embd))
        self.bias = nn.Parameter(torch.zeros(GPTConfig.n_embd)) if GPTConfig.includeBias else None

    def forward(self, x):
        """
        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """
        return F.layer_norm(x, 
                            normalized_shape=self.weights.shape,
                            weight=self.weights,
                            bias=self.bias,
                            eps=1e-5) # a small constant to avoid division by zero.

class FFN(nn.Module):
    """ Position-wise Feed-Forward Networks """
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            # maps from d_model → d_ff
            nn.Linear(GPTConfig.n_embd, 4*GPTConfig.n_embd, bias=GPTConfig.includeBias),
            
            # GELU = Gaussian Error Linear Unit.
            nn.GELU(),
            
            # maps from d_ff → d_model
            nn.Linear(4*GPTConfig.n_embd, GPTConfig.n_embd, bias=GPTConfig.includeBias),

            # Residual Dropout
            nn.Dropout(GPTConfig.dropout)
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
    def __init__(self):
        super().__init__()
        self.norm1 = NormLayer()
        self.masked_mth = MultiHeadAttentionWithResidual()
        self.norm2 = NormLayer()
        self.ffn = FFN()

    def forward(self, x):
        """
        Add residual connections.

        Args: 
            x (tuple): (batch, seq_len, hidden_dim). hidden_dim = n_features = n_embd
        """
        x = x + self.masked_mth(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
        
class GPT(nn.Module):
    """ Main architecture of a GPT-style decoder """
    def __init__(self):
        super().__init__()
        assert GPTConfig.context_length is not None
        assert GPTConfig.vocab_size is not None

        self.config = GPTConfig()

        self.transformer = nn.ModuleDict(dict(
            token_embd_table = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            pos_embd_table = nn.Embedding(self.config.context_length, self.config.n_embd),
            dropout = nn.Dropout(self.config.dropout),
            blocks = nn.ModuleList([Block() for _ in range(self.config.n_layer)]),

            # an additional layer normalization was added after the final self-attention block
            # source: GPT-2 paper
            norm = NormLayer(),
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
        params_decay = [p for n, p in params.items() if p.dim() >= 2]
        params_non_decay = [p for n, p in params.items() if p.dim() < 2]

        optimize_params = [
            {'params': params_decay, 'weight_decay': weight_decay},
            {'params': params_non_decay, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimize_params, lr=learning_rate, betas=betas)
        
        return optimizer

    def crop_block_size(self, new_context_length):
        """ Reduce the block size if the new_block_size < GPTConfig.context_length
            e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
            but want to use a smaller block size for some smaller, simpler model
        """
        assert new_context_length <= self.config.context_length, f"Error: Model maximal context length is only {GPTConfig.context_length}"
        self.config.context_length = new_context_length

        # Recall: pos_embd_table.shape = (context_length, n_embd)
        self.transformer.pos_embd_table.weight = nn.Parameter(self.transformer.pos_embd_table.weight[:new_context_length])

        for block in self.transformer.blocks:
            for head in block.masked_mth.heads:
                if hasattr(head, "tril"):
                    head.tril = head.tril[:new_context_length, :new_context_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained model from Hugging Face, either GPT-2 or LLaMA.

        Args:
            model_type (str): Name of the model, e.g., 'gpt2', 'gpt2-medium', 'llama-7b'.
            override_args (dict, optional): Overrides for model configuration, e.g., {'dropout': 0.1}.

        Returns:
            model: Hugging Face pretrained model ready for inference or fine-tuning.
        """
        override_args = override_args or {}

        # Validate supported models
        supported_models = {
            'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
            'llama-7b', 'llama-13b'
        }
        assert model_type in supported_models, f"Unsupported model: {model_type}"

        # Load Hugging Face model
        if model_type.startswith('gpt2'):
            from transformers import GPT2LMHeadModel, GPT2Config
            config = GPT2Config.from_pretrained(model_type)
            
            for k, v in override_args.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            model = GPT2LMHeadModel.from_pretrained(model_type, config=config)

        elif model_type.startswith('llama'):
            from transformers import LlamaForCausalLM, LlamaConfig
            config = LlamaConfig.from_pretrained(model_type)
            
            for k, v in override_args.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            model = LlamaForCausalLM.from_pretrained(model_type, config=config)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        print(f"Loaded pretrained {model_type} successfully.")
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

        return context