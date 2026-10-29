import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_feat: int = 5, hid: int = 32):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hid, batch_first=True)
        self.dec = nn.LSTM(hid, n_feat, batch_first=True)

    def forward(self, x):
        # x : (batch, seq_len, n_feat)
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.dec(z)
        return out