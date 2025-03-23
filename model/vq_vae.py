import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .my_transformer import TransformerBlock

class ResBlock(nn.Module):

    def __init__(self, in_chan = 256):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_chan),
            nn.SiLU(),
            nn.Conv2d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_chan)
        )
    
    def forward(self, x):
        x = x + self.block(x)
        return x

class VectorQuantize(nn.Module):

    def __init__(self, h_input, w_input, in_chan = 3, d_embedding = 256, patch_size = 16, vocab_size = 1024, beta = 0.15):
        super(VectorQuantize, self).__init__()
        self.vocab_size = vocab_size
        self.in_chan = in_chan
        self.d_embedding = d_embedding
        self.h_input = h_input
        self.w_input = w_input
        self.patch_size = patch_size

        self.h_seq = self.h_input // patch_size
        self.w_seq = self.w_input // patch_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=d_embedding//4, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//4),
            nn.Conv2d(in_channels=d_embedding//4, out_channels=d_embedding//2, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//2),
            nn.Conv2d(in_channels=d_embedding//2, out_channels=d_embedding, kernel_size=3, padding=1)
        )
        self.deconv = nn.Sequential(
            nn.Conv2d(in_channels=d_embedding, out_channels=d_embedding//2, kernel_size=3, padding=1),
            ResBlock(d_embedding//2),
            nn.ConvTranspose2d(in_channels=d_embedding//2, out_channels=d_embedding//4, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//4),
            nn.ConvTranspose2d(in_channels=d_embedding//4, out_channels=in_chan, kernel_size=patch_size//2, stride=patch_size//2),
        )

        self.codebook = nn.Embedding(vocab_size, d_embedding)
        self.beta = beta
        self.gamma = 0.25 * self.beta
    
    def forward(self, x):
        B, _, H, W = x.shape # B 3 H W
        img = x

        # patchify
        x = self.conv(x) # B d_embedding h w
        B, _, h, w = x.shape
        L = h * w
        x = x.permute(0, 2, 3, 1).reshape(B*h*w, self.d_embedding)

        # distances = torch.sum(x**2, dim=1, keepdim=True) + torch.sum(self.codebook.weight**2, dim=1) - 2 * torch.matmul(x, self.codebook.weight.t())
        distances = torch.cdist(x, self.codebook.weight, p=2)
        encoding_indices = torch.argmin(distances, dim=1).view(B, L)

        quantized = self.codebook(encoding_indices.view(B * L)).view(B, L, -1) # B, L, C

        x = x.view(B, L, self.d_embedding)
        temp = x + (quantized - x).detach()

        recons = self.deconv(temp.view(B, self.h_seq, self.w_seq, self.d_embedding).permute(0, 3, 1, 2))

        loss = F.mse_loss(recons, img) + self.beta * F.mse_loss(quantized, x.detach()) + self.gamma * F.mse_loss(x, quantized.detach())
        
        return encoding_indices, quantized, recons, loss
    
    def encode(self, x):
        B, _, H, W = x.shape # B 3 H W
        img = x

        # patchify
        x = self.conv(x) # B d_embedding h w
        B, _, h, w = x.shape
        L = h * w
        x = x.permute(0, 2, 3, 1).reshape(B*h*w, self.d_embedding)

        # distances = torch.sum(x**2, dim=1, keepdim=True) + torch.sum(self.codebook.weight**2, dim=1) - 2 * torch.matmul(x, self.codebook.weight.t())
        distances = torch.cdist(x, self.codebook.weight, p=2)
        encoding_indices = torch.argmin(distances, dim=1).view(B, L)

        quantized = self.codebook(encoding_indices.view(B * L)).view(B, L, -1) # B, L, C
        return encoding_indices, quantized
    
    def decode(self, x):
        B, L, C = x.shape
        recons = self.deconv(x.view(B, self.h_seq, self.w_seq, self.d_embedding).permute(0, 3, 1, 2))
        return recons

class BSQVectorQuantize(nn.Module):

    def __init__(self, h_input, w_input, in_chan = 3, d_embedding = 256, patch_size = 16, bits=8, beta = 0.15):
        super(BSQVectorQuantize, self).__init__()
        self.bits = bits
        self.in_chan = in_chan
        self.d_embedding = d_embedding
        self.h_input = h_input
        self.w_input = w_input
        self.patch_size = patch_size

        self.h_seq = self.h_input // patch_size
        self.w_seq = self.w_input // patch_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=d_embedding//4, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//4),
            nn.Conv2d(in_channels=d_embedding//4, out_channels=d_embedding//2, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//2),
            nn.Conv2d(in_channels=d_embedding//2, out_channels=d_embedding, kernel_size=3, padding=1)
        )
        self.proj = nn.Linear(self.d_embedding, self.bits)
        self.deconv = nn.Sequential(
            nn.Conv2d(in_channels=d_embedding, out_channels=d_embedding//2, kernel_size=3, padding=1),
            ResBlock(d_embedding//2),
            nn.ConvTranspose2d(in_channels=d_embedding//2, out_channels=d_embedding//4, kernel_size=patch_size//2, stride=patch_size//2),
            ResBlock(d_embedding//4),
            nn.ConvTranspose2d(in_channels=d_embedding//4, out_channels=in_chan, kernel_size=patch_size//2, stride=patch_size//2),
        )

        self.codebook = nn.Embedding(2 * self.bits, d_embedding)
        self.register_buffer('pos_ones', torch.ones(1, self.bits, dtype=torch.float32), persistent=False)
        self.register_buffer('neg_ones', -torch.ones(1, self.bits, dtype=torch.float32), persistent=False)
        self.register_buffer('code_indices', torch.arange(2**self.bits).long(), persistent=False)
        self.register_buffer('code_base', (2**torch.arange(self.bits - 1, -1, -1)).long(), persistent=False)

        self.beta = beta
        self.gamma = 0.25 * self.beta
    
    def __debug(t: torch.Tensor, name: str):
        print(f'{name}: {t}')
        print(f'{name} shape: {t.shape}')

    def forward(self, x):
        B, _, H, W = x.shape # B 3 H W
        img = x

        # patchify
        x = self.conv(x) # B d_embedding h w
        B, _, h, w = x.shape
        L = h * w
        x = x.permute(0, 2, 3, 1).reshape(B*h*w, self.d_embedding)
        x = self.proj(x)
        
        # BSQ
        x = F.normalize(x, p=1, dim=-1)
        sign = torch.where(
            x > 0,
            self.pos_ones.expand_as(x),
            self.neg_ones.expand_as(x),
        )
        sign = sign / self.bits**0.5
        
        loss = F.mse_loss(sign, x) * self.beta
        code = (sign - x).detach() + x

        code = code * self.bits**0.5 / 2 + 0.5

        encoding_indices = (code.long() * self.code_base).sum(-1, keepdim=True).view(B, L)

        code = torch.cat([code, 1 - code], dim=-1)
        quantized = code @ self.codebook.weight
        quantized = quantized.view(B, L, self.d_embedding)

        recons = self.deconv(quantized.view(B, self.h_seq, self.w_seq, self.d_embedding).permute(0, 3, 1, 2))

        loss += F.mse_loss(recons, img)
        
        # 熵损失
        tau = 1.0  # 温度参数
        soft_prob = torch.sigmoid(2 * tau * x)  # 软量化概率
        entropy = -(soft_prob * torch.log(soft_prob + 1e-10) + (1 - soft_prob) * torch.log(1 - soft_prob + 1e-10))
        entropy_loss = -entropy.mean()  # 熵损失
        loss += entropy_loss * self.gamma

        return encoding_indices, quantized, recons, loss
    
    def encode(self, x):
        raise NotImplementedError()
    
    def decode(self, x):
        B, L, C = x.shape
        recons = self.deconv(x.view(B, self.h_seq, self.w_seq, self.d_embedding).permute(0, 3, 1, 2))
        return recons

if __name__ == '__main__':
    pass