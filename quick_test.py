import torch
import os
import argparse
from models.semcovert import create_network
from utils.video_utils import load_video_as_tensor, split_video_tensor, create_comparison_video, create_four_panel_video
from utils.metrics import calculate_psnr, calculate_ssim

pre_model_config = {
        'depth': 4,
        'dim': 96,
        'use_channel': True,
        'vae_config': {
            'dim': 96,
            'z_dim': 16,
            'dim_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_scales': [],
            'temperal_downsample': [False, True, True],
            'dropout': 0.0
        },
        'channel_config': {
            'channel_type': 'AWGN',  # Channel type
            'snr': 25,               # Signal-to-noise ratio
        },    
    }

def visualize_results(model, cover_video_path, secret_video_path, output_dir, target_size, batch_size, device):
    print("\n--- Visualizing Test Results ---")
    model.eval()
    
    if secret_video_path is None or not os.path.exists(secret_video_path):
        secret_mode = False
        print("Secret video not found, running in cover-only mode.")
    else:
        secret_mode = True
        print("Secret video found, running in cover-secret mode.")

    # Prepare test videos
    cover_video = load_video_as_tensor(cover_video_path, target_size=target_size)
    cover_video_chunk = split_video_tensor(cover_video, 5)
    video_cover = cover_video_chunk.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
    video_cover = video_cover.contiguous().view(-1, video_cover.shape[2], video_cover.shape[3], video_cover.shape[4])  # (B*T, C, H, W)
    video_cover = video_cover.permute(1,0,2,3) # (C, B*T, H, W)

    if secret_mode:
        secret_video = load_video_as_tensor(secret_video_path, target_size=target_size)
        secret_video_chunk = split_video_tensor(secret_video, 5)
        video_secret = secret_video_chunk.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        video_secret = video_secret.contiguous().view(-1, video_secret.shape[2], video_secret.shape[3], video_secret.shape[4])  # (B*T, C, H, W)
        video_secret = video_secret.permute(1,0,2,3) # (C, B*T, H, W)

    # Pad the batch dimension (0th dim) to make both chunks the same length
    if secret_mode:
        cover_len = cover_video_chunk.shape[0]
        secret_len = secret_video_chunk.shape[0]
        if cover_len > secret_len:
            pad_shape = list(secret_video_chunk.shape)
            pad_shape[0] = cover_len - secret_len
            pad_chunk = torch.zeros(pad_shape, dtype=secret_video_chunk.dtype, device=secret_video_chunk.device)
            secret_video_chunk = torch.cat([secret_video_chunk, pad_chunk], dim=0)
        elif cover_len < secret_len:
            pad_shape = list(cover_video_chunk.shape)
            pad_shape[0] = secret_len - cover_len
            pad_chunk = torch.zeros(pad_shape, dtype=cover_video_chunk.dtype, device=cover_video_chunk.device)
            cover_video_chunk = torch.cat([cover_video_chunk, pad_chunk], dim=0)

        if secret_video_chunk.shape != cover_video_chunk.shape:
            raise ValueError("Cover and secret video chunks must have the same shape for processing.")
    
    recon_cover = torch.empty_like(cover_video_chunk)
    if secret_mode:
        recon_secret = torch.empty_like(secret_video_chunk) 
    
    with torch.no_grad():
        for i in range(0, cover_video_chunk.shape[0], batch_size):
            batch_cover = cover_video_chunk[i:i + batch_size].to(device)
            if secret_mode:
                batch_secret = secret_video_chunk[i:i + batch_size].to(device) 
            else:
                batch_secret = None

            output = model(batch_cover, batch_secret)
            recon_cover_chunk = output['cover_reconstructed'].cpu()
            recon_cover[i:i + batch_size] = recon_cover_chunk
            if secret_mode:
                recon_secret_chunk = output['secret_reconstructed'].cpu()
                recon_secret[i:i + batch_size] = recon_secret_chunk

    recon_cover = recon_cover.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
    recon_cover = recon_cover.contiguous().view(-1, recon_cover.shape[2], recon_cover.shape[3], recon_cover.shape[4])  # (B*T, C, H, W)
    recon_cover = recon_cover.permute(1,0,2,3) # (C, B*T, H, W)
    if recon_cover.shape[1] > video_cover.shape[1]:
        recon_cover = recon_cover[:,:video_cover.shape[1]]
    
    if secret_mode:
        recon_secret = recon_secret.permute(0, 2, 1, 3, 4)
        recon_secret = recon_secret.contiguous().view(-1, recon_secret.shape[2], recon_secret.shape[3], recon_secret.shape[4])
        recon_secret = recon_secret.permute(1,0,2,3) # (C, B*T, H, W)
        if recon_secret.shape[1] > video_secret.shape[1]:
            recon_secret = recon_secret[:,:video_secret.shape[1]]

    cover_psnr = calculate_psnr(video_cover, recon_cover)
    cover_ssim = calculate_ssim(video_cover, recon_cover)
    create_comparison_video(video_cover, recon_cover, os.path.join(output_dir, 'comparison_cover_pretrained.mp4'))
    if secret_mode:
        secret_psnr = calculate_psnr(video_secret, recon_secret)
        secret_ssim = calculate_ssim(video_secret, recon_secret)
        create_comparison_video(video_secret, recon_secret, os.path.join(output_dir, 'comparison_secret_pretrained.mp4'))
        
        # Create four-panel comparison video for better visualization in secret mode
        print("Creating four-panel comparison video...")
        create_four_panel_video(
            video_cover, recon_cover, 
            video_secret, recon_secret,
            os.path.join(output_dir, 'four_panel_comparison_pretrained.mp4')
        )

    print(f"Cover PSNR: {cover_psnr:.2f} dB, SSIM: {cover_ssim:.4f}")
    if secret_mode:
        print(f"Secret PSNR: {secret_psnr:.2f} dB, SSIM: {secret_ssim:.4f}")

    print("✅ Visualization videos saved.")

def test(cfg, pre_model_path, cover_video_path, secret_video_path, output_dir, target_size, batch_size, device):
    # Example test function. Implement your test logic here.
    model = create_network(cfg)
    model.load_state_dict(torch.load(pre_model_path, map_location=device))
    model.to(device)

    visualize_results(model, cover_video_path, secret_video_path, output_dir, target_size=target_size, batch_size=batch_size, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Test for SemCovert Model")
    parser.add_argument('--config', type=str, default='pre_model_config', help='Configuration for the model')
    parser.add_argument('--pre_model_path', type=str, default='./weights/best_model_weights.pth', help='Path to the pretrained model weights')
    parser.add_argument('--cover_video_path', type=str, required=True, help='Path to the cover video file')
    parser.add_argument('--secret_video_path', type=str, default=None, help='Path to the secret video file')
    parser.add_argument('--output_dir', type=str, default='./visual_result', help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--target_size', type=int, nargs=2, default=None, help='Target size for video frames (height, width)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')

    args = parser.parse_args()
    model_cfg = eval(args.config)  # Evaluate the string to get the config dictionary
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running quick test with config: {model_cfg}")
    test(model_cfg, args.pre_model_path, args.cover_video_path, args.secret_video_path, args.output_dir, 
         target_size=args.target_size, batch_size=args.batch_size, device=args.device)
    print("Test completed successfully.")