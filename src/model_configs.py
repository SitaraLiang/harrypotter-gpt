from dataclasses import dataclass

@dataclass
class GPTConfig:
    context_length:int = 1024 # maximum sequence length, GPT-2 configuration
    vocab_size:int = 50304 # GPT-2 configuration
    n_embd: int = 768 #d_model
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    includeBias: bool = False
