
import torch

# Main training configuration. All parameters are managed here.
TRAIN_CONFIG = {
    # ========== General Settings ==========
    'general': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'output_dir': 'training_results',
        'pretrained_model_path': 'preweight/Wan2.1_VAE.pth',
        'freeze_encoder': False,  # Freeze encoder weights if True
        'freeze_decoder': False,  # Freeze decoder weights if True
        # Differential learning rate strategy
        'use_differential_lr': True,
        'lr_strategy': 'aggressive',
        # Output file names
        'weights_pth_name': 'best_model_weights.pth',
        'checkpoint_pth_name': 'check_model.pth',
        'summary_json_name': 'training_summary.json',
        'train_plot_name': 'training_results.png',
        'test_metrics_name': 'test_metrics.txt',
    },

    # ========== Differential Learning Rate Settings ==========
    'differential_lr': {
        # Aggressive strategy: smaller lr for VAE, larger for fusion/extraction modules
        'vae_factor': 0.2,
        'fusion_factor': 4.0,
        'extraction_factor': 4.0,
        'other_factor': 1.0,
    },

    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'mode': 'min',
        'factor': 0.7,
        'patience': 3,
        'min_lr': 1e-7,
    },

    # ========== Video Data Settings ==========
    'video': {
        'data_dir': 'train_data',
        'target_size': (240, 320),  # (height, width)
        'max_frames': 5,
        'train_ratio': 0.95,  # Ratio of training data, the rest is validation when training
    },

    # ========== Training Hyperparameters ==========
    'training': {
        'epochs': 50,  # Fewer epochs due to faster convergence
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 2,
        'accumulate_grad_steps': 16,
        'grad_clip': 1.0,
        'patience': 5,  # Early stopping patience

        # Logging and monitoring
        'print_every_batches': 100,

        # Mixed training settings
        'cover_only_probability': 0.3,  # Probability for cover-only training
        'enable_mixed_training': True,

        # Convergence targets
        'target_loss': 0.1,
        'target_psnr': 35,
    },

    # ========== Loss Function Weights ==========
    'loss_weights': {
        'lambda_recon_cover': 1.0,      # Cover reconstruction loss
        'lambda_recon_secret': 1.0,     # Secret reconstruction loss
        'lambda_perceptual': 0.05,     # Perceptual loss (reduced weight)
        'lambda_embedding': 0.1,       # Embedding constraint loss (moderate weight)
        'lambda_cover_only': 1.0,       # Cover-only mode weight
        'lambda_kl_cover': 0.05,      # Cover VAE KL divergence loss (small weight)
        'lambda_kl_secret': 0.05,     # Secret VAE KL divergence loss (small weight)
        'lambda_null_secret': 1.0,      # Null secret loss (reduced weight)
    },

    # ========== Model Architecture Settings ==========
    'model': {
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
            'snr': 20,               # Signal-to-noise ratio
        },
    },

    # ========== Performance Optimization ==========
    'optimization': {
        'enable_mixed_precision': True,  # Enable mixed precision training
        'gradient_checkpointing': True,  # Enable gradient checkpointing for memory saving
    },
}
