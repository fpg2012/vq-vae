import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    
    def __init__(self, d_embedding = 256, d_model = 128, n_head = 8):
        super(TransformerBlock, self).__init__()
        self.d_embedding = d_embedding
        self.d_model = d_model
        self.n_head = n_head
        self.linear = nn.Linear(self.d_embedding, self.d_model*2)
        self.linear_v = nn.Linear(self.d_embedding, self.d_embedding)
        self.linear2 = nn.Linear(self.d_embedding, self.d_embedding)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_embedding, self.d_embedding * 4),
            nn.SiLU(),
            nn.Linear(self.d_embedding * 4, self.d_embedding)
        )
        self.layer_norm1 = nn.LayerNorm(d_embedding)
        self.layer_norm2 = nn.LayerNorm(d_embedding)
    
    def forward(self, x: torch.Tensor, is_train = True):
        B, L, C = x.shape
        # input transform
        q, k = self.linear(x).reshape(B, L, self.d_model, 2).permute(3, 0, 1, 2).unbind(0)
        v = self.linear_v(x)

        # multi-head partition
        q = q.view(B, L, self.n_head, self.d_model//self.n_head).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_head, self.d_model//self.n_head).permute(0, 2, 1, 3)
        v = v.view(B, L, self.n_head, self.d_embedding//self.n_head).permute(0, 2, 1, 3)

        attn_mask = None
        if is_train:
            attn_mask = torch.full((1, 1, L, L), float('-inf')).to(x.device)
            attn_mask = torch.triu(attn_mask, diagonal=1).expand(B, self.n_head, -1, -1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn = attn.permute(0, 2, 1, 3).reshape(B, L, self.d_embedding)
        attn = self.linear(attn)
        x = x + attn
        x = self.layer_norm1(x)

        x = x + self.ffn(x)
        x = self.layer_norm2(x)
        return x