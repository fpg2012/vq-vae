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

from model.vq_vae import VectorQuantize, BSQVectorQuantize

class VQTrainer:

    def __init__(self, train_set, test_set, batch_size = 16, shuffle = True, device = 'cpu'):
        super(VQTrainer, self).__init__()
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.device = device
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, model: VectorQuantize, epoch = 10, lr = 1e-3):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train()
        if self.device == 'xpu':
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print(f'device: {self.device}')
        for i in range(epoch):
            print(f'epoch {i+1}')
            size = len(self.train_set)
            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                encoding_indices, quantized, recons, loss = model.forward(X)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            torch.save(model, f'vq-epoch{i+1:>02d}.pth')

            total_loss = 0
            with torch.no_grad():
                for _, (X, y) in enumerate(self.test_loader):
                    X = X.to(self.device)
                    encoding_indices, quantized, recons, loss = model.forward(X)
                    total_loss += loss.detach().cpu().item()
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

    args = parser.parse_args()

    checkpoint = args.checkpoint
    device = args.device

    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    
    if checkpoint is None:
        # model = VectorQuantize(28, 28, in_chan=1, d_embedding=128, patch_size=4, vocab_size=128, beta=0.15).to(device)
        model = BSQVectorQuantize(28, 28, in_chan=1, d_embedding=128, patch_size=4, bits=8, beta=0.15).to(device)
    else:
        model = torch.load(checkpoint).to(device)

    trainer = VQTrainer(MNIST(root='data', train=True, transform=ToTensor()), MNIST(root='data', train=False, transform=ToTensor()), batch_size=512, device=device)
    trainer.train(model, epoch=5, lr=1e-4)