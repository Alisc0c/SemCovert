import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import numpy as np
import scipy

def calculate_psnr(original, reconstructed, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original video tensor
        reconstructed: Reconstructed video tensor  
        max_value: Maximum possible pixel value (default 1.0 for normalized videos)
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(torch.tensor(mse))).item()
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(v1, v2, window_size=11, size_average=True):
    """
    Calculate SSIM for video (B, C, T, H, W) or (C, T, H, W) by averaging frame-wise SSIM.
    
    Args:
        v1: First video tensor (B, C, T, H, W) or (C, T, H, W)
        v2: Second video tensor 
        window_size: Gaussian window size (default 11)
        size_average: If True, return mean SSIM (default True)
        
    Returns:
        SSIM value (scalar)
    """
    # Ensure input is 5D (B, C, T, H, W)
    if v1.dim() == 4:
        v1 = v1.unsqueeze(0)  # (C, T, H, W) -> (1, C, T, H, W)
        v2 = v2.unsqueeze(0)
    elif v1.dim() != 5:
        raise ValueError("Input must be (B, C, T, H, W) or (C, T, H, W)")

    B, C, T, H, W = v1.shape
    window = create_window(window_size, C)
    
    if v1.is_cuda:
        window = window.cuda(v1.get_device())
    window = window.type_as(v1)

    total_ssim = 0.0
    for b in range(B):
        for t in range(T):
            frame1 = v1[b, :, t, :, :]  # (C, H, W)
            frame2 = v2[b, :, t, :, :]
            total_ssim += _ssim(frame1, frame2, window, window_size, C, size_average)

    return (total_ssim / (B * T)).item()  # Average over all frames and batches

def calculate_fvd(
    model_3d: torch.nn.Module,  # Pre-trained 3D model
    real_videos: torch.Tensor,   # Real video (B, C, T, H, W)
    fake_videos: torch.Tensor,   # Fake video (B, C, T, H, W)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
) -> float:
    """
    Calculate Frechet Video Distance (FVD) between real and generated videos.
    Args:
        model_3d: Pre-trained 3D model for feature extraction
        real_videos: Real video tensor (B, C, T, H, W)
        fake_videos: Fake video tensor (B, C, T, H, W)
        device: Device to run the model on
        batch_size: Batch size for feature extraction
    Returns:
        FVD value (float)
    """
    model_3d.eval().to(device)
    
    assert real_videos.dim() == 5 and fake_videos.dim() == 5, "Input must be 5D tensors (B, C, T, H, W)"
    assert real_videos.size(1) == 3 and fake_videos.size(1) == 3, "Input videos must have 3 channels (RGB)"
    
    # Preprocess videos
    def preprocess(videos: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
        videos = (videos - mean) / std
        
        # Resize videos to (224, 224) if not already
        if videos.shape[-2:] != (224, 224):
            b, c, t, h, w = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, 224, 224).permute(0, 2, 1, 3, 4)
        return videos
    
    # Extract features from videos
    def extract_features(videos: torch.Tensor, batch_size) -> torch.Tensor:
        features = []
        with torch.no_grad():
            for i in range(0, videos.size(0), batch_size):
                batch = videos[i:i + batch_size].to(device)
                if batch.dim() == 4:
                    # If input is (C, T, H, W), add batch dimension
                    batch = batch.unsqueeze(0)
                inputs = preprocess(batch.to(device))
                feature = model_3d(inputs)  # Extract features [B, feature_dim]
                if feature.dim() > 2:
                    feature = feature.flatten(1)  # Flatten to [B, feature_dim]
                features.append(feature)  

            features = torch.cat(features, dim=0)  # (B, feature_dim)
            return features
        
    real_features = extract_features(real_videos, batch_size)  # shape: (B, feature_dim)
    fake_features = extract_features(fake_videos, batch_size)  # shape: (B, feature_dim)
    
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()

    mu_real = np.mean(real_features, axis=0)  # (feature_dim,)
    mu_fake = np.mean(fake_features, axis=0)  # (feature_dim,)

    sigma_real = np.cov(real_features, rowvar=False)  # (feature_dim, feature_dim)
    sigma_fake = np.cov(fake_features, rowvar=False)  # (feature_dim, feature_dim)

    eps = 1e-6
    dim = sigma_real.shape[0]
    sigma_real += eps * np.eye(dim)
    sigma_fake += eps * np.eye(dim)

    try:
        sigma_real_sqrt = scipy.linalg.sqrtm(sigma_real)
        middle = sigma_real_sqrt @ sigma_fake @ sigma_real_sqrt
        middle = (middle + middle.T) / 2  
    except Exception as e:
        print("Error in sqrt computation:", e)
        return float('nan')

    eigenvals, eigenvecs = np.linalg.eigh(middle)
    eigenvals = np.maximum(eigenvals, 0)  
    cov_mean = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T

    m_diff_sq = np.sum((mu_real - mu_fake) ** 2)
    fvd = m_diff_sq + np.trace(sigma_real + sigma_fake - 2 * cov_mean)

    return float(fvd.real)

