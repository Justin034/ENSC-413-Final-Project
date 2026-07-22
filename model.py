import torch
import torch.nn as nn

class UNetArch(nn.Module):
    def __init__(self):
        super(UNetArch, self).__init__()
        
        # encoder layers of autoenc.
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # decoder layers of autoenc.
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Final layer accepts 64 channels (32 from dec2 + 32 from enc1 skip connection)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # normal forward pass from ensc1 to dec 2
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        d2 = self.dec2(e2)
        
        # skip connection on last decoder with first encoder
        combined = torch.cat([d2, e1], dim=1)
        
        # return mask from model 
        mask = self.dec1(combined)
        return mask