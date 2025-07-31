import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20):
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() not in [2, 3, 4, 5]:
            raise ValueError('Input tensor must be 2D, 3D, 4D, or 5D')
            
        original_dim = z_hat.dim()
        
        if original_dim == 2:
            z_hat = z_hat.unsqueeze(0)
        
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=tuple(range(1, z_hat.dim())), keepdim=True) / k    
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)

        if self.channel_type == 'Rayleigh':
            hc = torch.randn(2, device = z_hat.device) 
            # clone for in-place operation  
            z_hat = z_hat.clone()
            
            half_idx = z_hat.size(1) // 2
            shape_suffix = (1,) * (z_hat.dim() - 2) 
            
            z_hat[:, :half_idx] = hc[0].view(1, 1, *shape_suffix) * z_hat[:, :half_idx]
            z_hat[:, half_idx:] = hc[1].view(1, 1, *shape_suffix) * z_hat[:, half_idx:]

        zz = z_hat + noise
        if original_dim == 2:  
            zz = zz.squeeze(0)
        return zz

    def get_channel(self):
        return self.channel_type, self.snr
    

class PowerNormalization(nn.Module):
    def __init__(self, P=1):
        super().__init__()
        self.P = P

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.reshape(x.size(0), -1)  # [batch, features]
        l2_norm = torch.norm(x_flat, p=2, dim=1, keepdim=True)  # [batch, 1]
        l2_norm = torch.clamp(l2_norm, min=1e-8)
        k = x_flat.size(1)  
        scale = torch.sqrt(torch.tensor(self.P * k, dtype=x.dtype, device=x.device))
        x_normalized = scale * x_flat / l2_norm
        
        return x_normalized.view(original_shape)