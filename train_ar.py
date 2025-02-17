import argparse
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from model.vq_vae import VectorQuantize
from model.ar import ARGenerator

class ARTrainer:

    def __init__(self, train_set, test_set, batch_size = 16, shuffle = True, device = 'cpu'):
        super(ARTrainer, self).__init__()
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.device = device
        self.seq_len = 7*7
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, model: ARGenerator, vqvae: VectorQuantize, epoch = 10, lr = 1e-3):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        vqvae.eval()
        if self.device == 'xpu':
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print(f'device: {self.device}')
        for i in range(epoch):
            print(f'epoch {i+1}')
            size = len(self.train_set)

            model.train()
            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    indx, quantized = vqvae.encode(X)
                
                seq = torch.cat([model.sos_tokens(y)[:, None, :], quantized.detach()], dim=1).to(self.device)

                logits = model.forward(seq)
                logits = logits[:, :-1, :].reshape(X.shape[0] * self.seq_len, -1)
                target = indx.reshape(X.shape[0] * self.seq_len).detach().to(self.device)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            torch.save(model, f'ar-epoch{i+1:>02d}.pth')

            total_loss = 0
            model.eval()
            with torch.no_grad():
                for _, (X, y) in enumerate(self.test_loader):
                    X = X.to(self.device)
                    indx, quantized = vqvae.encode(X)
                
                    seq = torch.cat([model.sos_tokens(y)[:, None, :], quantized], dim=1)

                    logits = model.forward(seq)
                    loss = F.cross_entropy(logits[:, :-1, :].reshape(X.shape[0] * self.seq_len, -1), indx.view(X.shape[0] * self.seq_len))
                    total_loss += loss.item()
            print(f'test loss: {total_loss/len(self.test_loader):>7f}')

            torch.xpu.empty_cache()

if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="接受device和checkpoint参数")
    # 添加--device参数，默认值为"cpu"
    parser.add_argument('--device', type=str, default='cpu',
                        help='指定设备，默认为cpu')
    # 添加--checkpoint参数，默认值为None
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定检查点路径，默认为None')
    parser.add_argument('--vqvae', type=str, default='vq-best.safetensors',
                        help='指定检查点路径，默认为None')

    args = parser.parse_args()

    checkpoint = args.checkpoint
    device = args.device
    vqvae = args.vqvae

    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    
    vqvae = torch.load('vq-best.safetensors', weights_only=False).to(device)

    if checkpoint is None:
        model = ARGenerator(n_class=10, max_len=49, n_blocks=2, d_embedding=128, d_model=64, vocab_size=128, device=device).to(device)
    else:
        model = torch.load(checkpoint).to(device)

    trainer = ARTrainer(MNIST(root='data', train=True, transform=ToTensor()), MNIST(root='data', train=False, transform=ToTensor()), batch_size=512, device=device)
    trainer.train(model, vqvae=vqvae, epoch=20, lr=1e-3)