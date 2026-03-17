import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random
from torch.utils.data import DataLoader

from models.semcovert import create_network
from models.losses import create_loss_functions
from models.wan_vae import WanVAE
from utils.video_utils import load_video_as_tensor,split_video_tensor, get_video_files, VideoDataset
from utils.metrics import calculate_psnr
from train_config import TRAIN_CONFIG

def load_videos(video_cfg):
    """Load real video data from data folder to memory"""
    print("\n--- Loading Real Videos to Memory ---")
    data_dir = video_cfg['data_dir']
    target_size = video_cfg['target_size']
    max_frames = video_cfg['max_frames']
    train_ratio = video_cfg['train_ratio']
    
    video_files = get_video_files(data_dir)
    if not video_files:
        raise ValueError(f"No video files found in {data_dir}")
    
    print(f"Found {len(video_files)} video files: {[f.name for f in video_files]}")
    
    all_video_batches = []
    for video_file in video_files:
        video_tensor = load_video_as_tensor(
            str(video_file), 
            target_size=target_size
        )
        video_tensor = split_video_tensor(video_tensor, max_frames)
        if video_tensor.numel() > 0:
            for i in range(video_tensor.shape[0]):
                all_video_batches.append(video_tensor[i])
            del video_tensor  # Release individual video tensor immediately
        else:
            print(f"  Failed to load video: {video_file}")
    
    if not all_video_batches:
        raise ValueError("No valid video data loaded")
    
    all_video_tensor = torch.stack(all_video_batches, dim=0)
    del all_video_batches

    indices = torch.randperm(all_video_tensor.shape[0])
    train_count = int(all_video_tensor.shape[0] * train_ratio)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:]
    train_videos = all_video_tensor[train_indices]
    val_videos = all_video_tensor[val_indices]
    del all_video_tensor  

    if val_videos.shape[0] == 0:
        val_videos = train_videos[-2:] if train_videos.shape[0] >= 2 else train_videos[-1:]
        train_videos = train_videos[:-val_videos.shape[0]]

    print(f"Training videos: {train_videos.shape[0]}, Validation videos: {val_videos.shape[0]}")
    
    train_size_gb = train_videos.numel() * train_videos.element_size() / (1024**3)
    val_size_gb = val_videos.numel() * val_videos.element_size() / (1024**3)
    total_size_gb = train_size_gb + val_size_gb
    
    print(f"💾 Video data memory usage:")
    print(f"  Training data: {train_size_gb:.2f} GB")
    print(f"  Validation data: {val_size_gb:.2f} GB")
    print(f"  Total: {total_size_gb:.2f} GB")
    
    return train_videos, val_videos


def load_pretrained_vae(pretrained_path, config):
    """Load pretrained WAN VAE model"""
    print(f"\n--- Loading Pretrained VAE from {pretrained_path} ---")
    
    # Create VAE model
    vae = WanVAE(**config)
    
    # Load pretrained weights
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights
            missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            print(f"Successfully loaded pretrained VAE from {pretrained_path}")
            
            # If checkpoint contains training info, display it
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f"  Pretrained epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  Pretrained loss: {checkpoint['loss']}")
                    
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Using randomly initialized VAE instead")
            
    else:
        print(f"Pretrained model not found at {pretrained_path}")
        print("Using randomly initialized VAE")
    
    return vae

def setup_optimizer(model, config):
    """Setup optimizer with different learning rates for different modules"""
    params = []
    
    # Decide whether to use differential learning rate based on config
    use_differential_lr = config['general'].get('use_differential_lr', True)
    base_lr = config['training']['learning_rate']
    
    if use_differential_lr:
        # Use differential learning rate strategy
        diff_lr_config = config.get('differential_lr', {})
        vae_lr_factor = diff_lr_config.get('vae_factor', 0.2)
        fusion_lr_factor = diff_lr_config.get('fusion_factor', 4.0)
        extraction_lr_factor = diff_lr_config.get('extraction_factor', 4.0)
        other_lr_factor = diff_lr_config.get('other_factor', 1.0)
        
        strategy_name = config['general'].get('lr_strategy', 'aggressive')
        print(f"🎯 Using differential learning rate strategy: {strategy_name}")
    else:
        # Use uniform learning rate multipliers
        vae_lr_factor = 0.5
        fusion_lr_factor = 1.0
        extraction_lr_factor = 1.0
        other_lr_factor = 1.0
        print(f"📊 Using uniform learning rate strategy")
    
    print(f"  Base learning rate: {base_lr:.2e}")
    print(f"  VAE learning rate: {base_lr * vae_lr_factor:.2e} ({vae_lr_factor:.1f}x base)")
    print(f"  Fusion module learning rate: {base_lr * fusion_lr_factor:.2e} ({fusion_lr_factor:.1f}x base)")
    print(f"  Extraction module learning rate: {base_lr * extraction_lr_factor:.2e} ({extraction_lr_factor:.1f}x base)")
    
    # VAE encoder parameters - use very small learning rate to protect pretrained weights
    if config['general']['freeze_encoder']:
        print("❄️ Freezing VAE encoder")
        for param in model.vae.encoder.parameters():
            param.requires_grad = False
    else:
        encoder_params = list(model.vae.encoder.parameters())
        if encoder_params:
            params.append({
                'params': encoder_params, 
                'lr': base_lr * vae_lr_factor,
                'name': 'vae_encoder'
            })
            print(f"  ✅ VAE encoder: {len(encoder_params):,} parameters")
    
    # VAE decoder parameters - use very small learning rate to protect pretrained weights
    if config['general']['freeze_decoder']:
        print("❄️ Freezing VAE decoder")
        for param in model.vae.decoder.parameters():
            param.requires_grad = False
    else:
        decoder_params = list(model.vae.decoder.parameters())
        if decoder_params:
            params.append({
                'params': decoder_params, 
                'lr': base_lr * vae_lr_factor,
                'name': 'vae_decoder'
            })
            print(f"  ✅ VAE decoder: {len(decoder_params):,} parameters")
    
    # Feature fusion module - use high learning rate for focused training
    fusion_params = list(model.fusion_module.parameters())
    if fusion_params:
        params.append({
            'params': fusion_params, 
            'lr': base_lr * fusion_lr_factor,
            'name': 'fusion_module'
        })
        print(f"  🚀 Fusion module: {len(fusion_params):,} parameters")
    
    # Secret extraction module - use high learning rate for focused training
    extraction_params = list(model.extraction_module.parameters())
    if extraction_params:
        params.append({
            'params': extraction_params, 
            'lr': base_lr * extraction_lr_factor,
            'name': 'extraction_module'
        })
        print(f"  🚀 Extraction module: {len(extraction_params):,} parameters")
    
    # If using channel module
    if hasattr(model, 'channel') and model.channel is not None:
        channel_params = list(model.channel.parameters())
        if channel_params:
            params.append({
                'params': channel_params, 
                'lr': base_lr * other_lr_factor,
                'name': 'channel'
            })
            print(f"  📡 Channel module: {len(channel_params):,} parameters")
    
    # Other parameters (if any)
    all_named_params = set()
    for group in params:
        all_named_params.update(id(p) for p in group['params'])
    
    other_params = [p for p in model.parameters() if id(p) not in all_named_params and p.requires_grad]
    if other_params:
        params.append({
            'params': other_params, 
            'lr': base_lr * other_lr_factor,
            'name': 'other'
        })
        print(f"  🔧 Other parameters: {len(other_params):,} parameters")
    
    if not params:
        # If all parameters are frozen, use all parameters
        print("⚠️  All parameters are frozen, using standard optimizer")
        params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params, 
        weight_decay=config['training']['weight_decay']
    )
    
    # Display final parameter group information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Optimizer setup complete:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Parameter groups: {len(params)}")

    # Learning rate scheduler
    scheduler_type = config['scheduler'].get('type', 'ReduceLROnPlateau')
    if scheduler_type == 'ReduceLROnPlateau':
        print("📉 Using ReduceLROnPlateau scheduler")
        mode = config['scheduler'].get('mode', 'min')
        factor = config['scheduler'].get('factor', 0.7)
        patience = config['scheduler'].get('patience', 2)
        min_lr = config['scheduler'].get('min_lr', 1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            min_lr=min_lr
        )
        print(f"  Scheduler params: mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr}")
    else:
        NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")
    
    return optimizer, scheduler

def clear_cuda_cache():
    """Clear CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def plot_training_results(train_losses, val_losses,  output_dir):
    """Plot training results"""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')

    plt.tight_layout()
    plot_name = ('training_results_pretrained.png')
    plt.savefig(os.path.join(output_dir, plot_name), dpi=150)
    plt.close()

def train(cfg):
    """Joint training using pretrained VAE"""
    device = cfg['general']['device']
    output_dir = cfg['general']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load video data
    train_videos, val_videos = load_videos(cfg['video'])
    print(f"Loaded {train_videos.shape[0]} training videos, {val_videos.shape[0]} validation videos")
    
    # Build Dataset and DataLoader
    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(VideoDataset(train_videos), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(VideoDataset(val_videos), batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Load pretrained VAE
    vae_path = cfg['general']['pretrained_model_path']
    if os.path.exists(vae_path):
        pretrained_vae = load_pretrained_vae(vae_path, cfg['model']['vae_config'])
        pretrained_vae = pretrained_vae.to(device)
    else:
        pretrained_vae = None

    model = create_network(cfg['model'], pretrained_vae)
    model = model.to(device)

    # Gradient checkpointing support (requires model to implement this method)
    optimization_cfg = cfg.get('optimization', {})
    use_amp = optimization_cfg.get('enable_mixed_precision', False)
    use_checkpoint = optimization_cfg.get('gradient_checkpointing', False)
    if use_checkpoint and hasattr(model, 'enable_gradient_checkpointing'):
        print("⚡ Enabling gradient checkpointing")
        model.enable_gradient_checkpointing()
    
    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, cfg)
    
    # Loss functions (using global config)
    steganography_loss = create_loss_functions(cfg['loss_weights'])
    steganography_loss = steganography_loss.to(device)
    
    # Display model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # AMP scaler
    scaler = torch.amp.GradScaler(device='cuda') if use_amp and device == 'cuda' else None

    general_cfg = cfg['general']
    check_model_path = os.path.join(output_dir, general_cfg.get('checkpoint_pth_name', 'best_model_pretrained.pth'))
    weights_pth_name = general_cfg.get('weights_pth_name', 'best_model_weights_pretrained.pth')
    best_weights_path = os.path.join(output_dir, weights_pth_name)
    summary_json_name = general_cfg.get('summary_json_name', 'training_summary.json')
    
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    # ========== Resume from checkpoint ==========
    if os.path.exists(check_model_path):
        print(f"Detected existing model {check_model_path}, attempting to resume training...")
        checkpoint = torch.load(check_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if start_epoch >= cfg['training']['epochs']:
            start_epoch = cfg['training']['epochs'] - 1  # Prevent exceeding max epochs
        best_val_loss = checkpoint.get('loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    # ========== Resume from checkpoint END ==========

    # Training loop
    train_losses, val_losses = [], []
    val_psnrs = []
    val_secret_psnrs = []
    accumulate_steps = cfg['training']['accumulate_grad_steps']

    print(f"\n{'='*60}")
    print(f"Starting joint training (with pretrained VAE)")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        print_every_batches = cfg['training'].get('print_every_batches', 50)
        running_loss = 0.0
        running_batches = 0
        
        optimizer.zero_grad()
        
        # Use DataLoader for batch loading
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for batch_idx, cover_batch in progress_bar:
            # cover_batch: [batch_size, ...]
            cover_batch = cover_batch.to(device)
            
            # Mixed training: randomly decide whether to use hidden video
            use_secret = True
            if cfg['training'].get('enable_mixed_training', False):
                cover_only_prob = cfg['training'].get('cover_only_probability', 0.3)
                use_secret = np.random.random() > cover_only_prob
            
            if use_secret:
                # Randomly sample secret_batch
                secret_indices = torch.randint(0, len(train_videos), (cover_batch.size(0),))
                secret_batch = train_videos[secret_indices].to(device)
            else:
                secret_batch = None

            total_loss = None
            try:
                if use_amp and scaler is not None:
                    with torch.amp.autocast(device):
                        outputs = model(cover_batch, secret_batch)
                        losses = steganography_loss(cover_batch, secret_batch, outputs)
                        total_loss = losses['total_loss'] / accumulate_steps
                    scaler.scale(total_loss).backward()
                else:
                    outputs = model(cover_batch, secret_batch)
                    losses = steganography_loss(cover_batch, secret_batch, outputs)
                    total_loss = losses['total_loss'] / accumulate_steps
                    total_loss.backward()
                del outputs, losses
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {batch_idx}, skipping...")
                    clear_cuda_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
            
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                    optimizer.step()
                optimizer.zero_grad()
            
            if total_loss is not None:
                epoch_train_loss += total_loss.item() * accumulate_steps
                running_loss += total_loss.item() * accumulate_steps
                num_batches += 1
                running_batches += 1
            
            if running_batches > 0 and running_batches % print_every_batches == 0:
                avg_running_loss = running_loss / running_batches
                batch_progress = (num_batches * batch_size) / len(train_videos) * 100
                print(f"Epoch {epoch+1} | Batch {num_batches}/{len(train_loader)} ({batch_progress:.1f}%) | "
                      f"Running Loss: {avg_running_loss:.6f}")
                running_loss = 0.0
                running_batches = 0
            
            del cover_batch
            if secret_batch is not None:
                del secret_batch
            if total_loss is not None:
                del total_loss
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        clear_cuda_cache()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_psnr_sum = 0.0
            val_secret_psnr_sum = 0.0
            val_batches = 0

            val_iterator = iter(val_loader)  

            for val_batch in val_loader:
                cover_video = val_batch.to(device)

                if len(val_loader) == 1:
                    break

                try:
                    secret_video = next(val_iterator).to(device)  
                except StopIteration:
                    
                    secret_video = cover_video  

                if use_amp and scaler is not None:
                    with torch.amp.autocast(device):
                        outputs = model(cover_video, secret_video)
                        losses = steganography_loss(cover_video, secret_video, outputs)
                        recon_cover = outputs['cover_reconstructed']
                        recon_secret = outputs['secret_reconstructed']
                        val_loss_sum += losses['total_loss'].item()
                        val_psnr_sum += calculate_psnr(recon_cover, cover_video)
                        val_secret_psnr_sum += calculate_psnr(recon_secret, secret_video)

                        del outputs, losses, recon_cover
                else:
                    outputs = model(cover_video, secret_video)
                    losses = steganography_loss(cover_video, secret_video, outputs)
                    recon_cover = outputs['cover_reconstructed']
                    recon_secret = outputs['secret_reconstructed']
                    val_loss_sum += losses['total_loss'].item()
                    val_psnr_sum += calculate_psnr(recon_cover, cover_video)
                    val_secret_psnr_sum += calculate_psnr(recon_secret, secret_video)

                    del outputs, losses, recon_cover

                val_batches += 1
                del cover_video, secret_video

            if val_batches > 0:
                epoch_val_loss = val_loss_sum / val_batches
                val_psnr = val_psnr_sum / val_batches
                val_secret_psnr = val_secret_psnr_sum / val_batches
            else:
                epoch_val_loss = float('inf')
                val_psnr = 0.0
                val_secret_psnr = 0.0


        val_losses.append(epoch_val_loss)
        val_psnrs.append(val_psnr)
        val_secret_psnrs.append(val_secret_psnr)
        
        scheduler.step(epoch_val_loss)
        
        lr_info = []
        for i, group in enumerate(optimizer.param_groups):
            if 'name' in group:
                lr_info.append(f"{group['name']}: {group['lr']:.2e}")
            else:
                lr_info.append(f"group{i}: {group['lr']:.2e}")
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Val PSNR: {val_psnr:.2f}dB | Val Secret PSNR: {val_secret_psnr:.2f}dB")
        print(f"📊 Module learning rates: {' | '.join(lr_info)}")
        
        # Save best model weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)
            print(f"Saving best model, validation loss: {best_val_loss:.6f}, weights saved to: {best_weights_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save current model state
        save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': epoch_val_loss,
        'config': cfg,
        'patience_counter': patience_counter
        }
        if scaler is not None:
            save_dict['scaler'] = scaler.state_dict()
        torch.save(save_dict, check_model_path)
        # Early stopping check
        if patience_counter >= cfg['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Joint training completed")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation PSNR: {val_psnrs[-1]:.2f}dB")
    
    # Plot training results
    plot_training_results(train_losses, val_losses, output_dir)
    
    # Save training summary
    summary = {
        'pretrained_model': general_cfg['pretrained_model_path'],
        'total_epochs': len(train_losses),
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_psnr': val_psnrs[-1],
    }
    
    import json
    with open(os.path.join(output_dir, summary_json_name), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {output_dir}/{summary_json_name}")
    
    return summary

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    print("="*60)
    print(f"Starting Pretrained VAE Training Script at {datetime.now()}")
    print("="*60)
    cfg = TRAIN_CONFIG
    set_seed(cfg['general'].get('seed', 42))  # Set random seed for reproducibility
    
    try:
        summary = train(cfg)
        print(f"\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
