import torch
from torch import nn
import torch.nn.functional as F
from .my_transformer import TransformerBlock
from .vq_vae import VectorQuantize
import math

class ARGenerator(nn.Module):

    def __init__(self, n_class = 10, max_len = 7*7, n_blocks = 5, d_embedding = 128, d_model = 64, vocab_size = 128, temperature=1, device='cpu'):
        super(ARGenerator, self).__init__()
        self.n_class = n_class
        self.max_len = max_len
        self.n_blocks = n_blocks
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.device = device

        self.sos_tokens = nn.Embedding(n_class, d_embedding)
        init_std = math.sqrt(1 / d_embedding / 3)
        nn.init.trunc_normal_(self.sos_tokens.weight.data, mean=0, std=init_std)
        self.pos_encoding = nn.Embedding(self.max_len + 1, d_embedding)
        nn.init.trunc_normal_(self.pos_encoding.weight.data, mean=0, std=init_std)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = TransformerBlock(d_embedding=d_embedding, d_model=d_model, n_head=4).to(self.device)
            self.blocks.append(block)
        self.mlp = nn.Linear(d_embedding, vocab_size)
    
    def forward(self, x: torch.Tensor):
        B, L, C = x.shape

        x = x + self.pos_encoding.weight[:L]
        for block in self.blocks:
            x = block(x)
        logits = self.mlp(x)
        return logits