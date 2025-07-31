import torch
import torch.nn as nn
import torch.nn.functional as F


class Charbonnier_Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(Charbonnier_Loss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))
        return loss

class TotalLoss(nn.Module):
    """
    Comprehensive loss function
    Supports mixed training: with and without secret videos
    """
    
    def __init__(self, 
                 lambda_recon_cover=1.0,
                 lambda_recon_secret=1.0,
                 lambda_perceptual=0.01,  # Reduced perceptual loss weight
                 lambda_embedding=0.001,  # Reduced embedding loss weight
                 lambda_cover_only=1.0,
                 lambda_kl_cover=0.001,   # Cover VAE KL divergence loss weight
                 lambda_kl_secret=0.001,  # Secret VAE KL divergence loss weight
                 lambda_null_secret=1.0,  # Null secret loss weight for cover-only mode
                 ):  
        super().__init__()
        
        self.lambda_recon_cover = lambda_recon_cover
        self.lambda_recon_secret = lambda_recon_secret
        self.lambda_perceptual = lambda_perceptual  
        self.lambda_embedding = lambda_embedding
        self.lambda_cover_only = lambda_cover_only  # Cover-only mode weight
        self.lambda_kl_cover = lambda_kl_cover      # Cover KL divergence loss weight
        self.lambda_kl_secret = lambda_kl_secret    # Secret KL divergence loss weight
        self.lambda_null_secret = lambda_null_secret  # Null secret loss weight
        
        # Reconstruction loss
        self.recon_loss = Charbonnier_Loss()  
        self.perceptual_net = None
        if lambda_perceptual :
            from torchvision import models
            from torchvision.models import VGG16_Weights
            self.perceptual_net = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.perceptual_net = self.perceptual_net.features[:16].eval()
        
        print(f"TotalLoss initialized with weights:")
        print(f"  lambda_recon_cover: {lambda_recon_cover}")
        print(f"  lambda_recon_secret: {lambda_recon_secret}")
        print(f"  lambda_perceptual: {lambda_perceptual} (perceptual loss weight)")
        print(f"  lambda_embedding: {lambda_embedding} (embedding constraint loss weight)")
        print(f"  lambda_cover_only: {lambda_cover_only} (cover-only mode)")
        print(f"  lambda_kl_cover: {lambda_kl_cover} (cover KL divergence loss)")
        print(f"  lambda_kl_secret: {lambda_kl_secret} (secret KL divergence loss)")
        print(f"  lambda_null_secret: {lambda_null_secret} (null secret video reconstruction loss)")
    

    def perceptual_loss(self, x, y):
        """
        Perceptual loss using VGG network feature extraction
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1, 1)

        # Reshape from 5D to 4D by merging batch and time dimensions
        x = x.permute(0, 2, 1, 3, 4)  # [batch, time, channels, height, width] <- [batch, channels, time, height, width]
        y = y.permute(0, 2, 1, 3, 4)  # [batch, time, channels, height, width] <- [batch, channels, time, height, width]
        original_shape = x.shape
        x = x.reshape(-1, *original_shape[2:])  # [batch*time, channels, height, width]
        y = y.reshape(-1, *original_shape[2:])
        
        from torchvision import transforms
        normalize = transforms.Normalize(mean=mean.squeeze(-1), std=std.squeeze(-1))  # Remove depth dim for normalization
        resize = transforms.Resize((224, 224))  # VGG16 input size requirement
        
        x = resize(x)
        y = resize(y)
        x = normalize(x)
        y = normalize(y)

        # Process through VGG
        x_features = self.perceptual_net(x)
        y_features = self.perceptual_net(y)
        
        loss = nn.MSELoss()(x_features, y_features)  # Fixed MSELoss usage
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Perceptual loss is NaN or Inf, setting to 0")
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        return loss
    
    def kl_divergence_loss(self, mu, log_var):

        log_var = torch.clamp(log_var, min=-5, max=5)  # Limit log_var range
        
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.numel()
        
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: KL divergence loss is NaN or Inf, setting to 0")
            return torch.tensor(0.0, device=mu.device, requires_grad=True)
        
        return kl_loss
    
    def embedding_constraint_loss(self, mu_cover, log_var_cover, mu_fused, log_var_fused):
        """
        Embedding constraint loss - ensures fused features maintain distribution similarity with cover features
        Args:
            mu_cover: Cover mean [B, latent_dim, T, H, W]
            log_var_cover: Cover log variance [B, latent_dim, T, H, W]
            mu_fused: Fused mean [B, latent_dim, T, H, W]
            log_var_fused: Fused log variance [B, latent_dim, T, H, W]
        Returns:
            Embedding constraint loss
        """
        # Numerical stability: limit log_var and mu ranges
        log_var_cover = torch.clamp(log_var_cover, min=-5, max=5)
        log_var_fused = torch.clamp(log_var_fused, min=-5, max=5)
        
        # Calculate KL divergence
        embedding_loss = -0.5 * torch.sum(
            1 + log_var_fused - log_var_cover - 
            ((mu_fused - mu_cover).pow(2) + log_var_fused.exp()) / log_var_cover.exp()
        )
        embedding_loss = embedding_loss / mu_cover.numel()  # Normalize to total element count
        
        if torch.isnan(embedding_loss) or torch.isinf(embedding_loss):
            print("Warning: embedding constraint loss is NaN or Inf, setting to 0")
            return torch.tensor(0.0, device=mu_cover.device, requires_grad=True)
        
        return embedding_loss
    
    def forward(self, 
                cover_video: torch.Tensor,
                secret_video: torch.Tensor | None,
                results: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Calculate total loss - Fixed version supporting cover-only mode
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=cover_video.device, requires_grad=True)
        
        # Detect if in cover-only mode (no secret video)
        cover_only_mode = secret_video is None
        
        # Cover video reconstruction loss
        if 'cover_reconstructed' in results:
            cover_recon = results['cover_reconstructed']
            
            # Ensure shape matching
            if cover_recon.shape != cover_video.shape:
                print(f"Shape mismatch: cover_recon {cover_recon.shape} vs cover_video {cover_video.shape}")
                # Simple shape adjustment
                if cover_recon.shape[2] != cover_video.shape[2]:  # Time dimension mismatch
                    min_t = min(cover_recon.shape[2], cover_video.shape[2])
                    cover_recon = cover_recon[:, :, :min_t]
                    cover_video_adjusted = cover_video[:, :, :min_t]
                else:
                    cover_video_adjusted = cover_video
            else:
                cover_video_adjusted = cover_video
            
            # Use safe loss calculation
            cover_recon_loss = self.recon_loss(cover_recon, cover_video_adjusted)
            if torch.isnan(cover_recon_loss) or torch.isinf(cover_recon_loss):
                print("Warning: recon loss is NaN or Inf, setting to 0")
                cover_recon_loss = torch.tensor(0.0, device=cover_video.device, requires_grad=True)

            losses['cover_recon_loss'] = cover_recon_loss
            
            # Adjust weight based on mode
            cover_weight = self.lambda_cover_only if cover_only_mode else self.lambda_recon_cover
            total_loss = total_loss + cover_weight * cover_recon_loss
            if self.lambda_perceptual and self.perceptual_net is not None:
                conver_perceptual_loss = self.perceptual_loss(cover_recon, cover_video_adjusted)
                losses['cover_perceptual_loss'] = conver_perceptual_loss
                total_loss = total_loss + conver_perceptual_loss * self.lambda_perceptual
        
        # Secret video reconstruction loss (only calculated when secret video exists)
        if not cover_only_mode and 'secret_reconstructed' in results:
            secret_recon = results['secret_reconstructed']
            
            # Ensure shape matching
            if secret_recon.shape != secret_video.shape:
                if secret_recon.shape[2] != secret_video.shape[2]:
                    min_t = min(secret_recon.shape[2], secret_video.shape[2])
                    secret_recon = secret_recon[:, :, :min_t]
                    secret_video_adjusted = secret_video[:, :, :min_t]
                else:
                    secret_video_adjusted = secret_video
            else:
                secret_video_adjusted = secret_video
            
            secret_recon_loss = self.recon_loss(secret_recon, secret_video_adjusted)
            if torch.isnan(secret_recon_loss) or torch.isinf(secret_recon_loss):
                print("Warning: Secret recon loss is NaN or Inf, setting to 0")
                secret_recon_loss = torch.tensor(0.0, device=secret_video.device, requires_grad=True)
                
            losses['secret_recon_loss'] = secret_recon_loss
            total_loss = total_loss + self.lambda_recon_secret * secret_recon_loss
            if self.lambda_perceptual and self.perceptual_net is not None:
                secret_perceptual_loss = self.perceptual_loss(secret_recon, secret_video_adjusted)
                losses['secret_perceptual_loss'] = secret_perceptual_loss
                total_loss = total_loss + secret_perceptual_loss * self.lambda_perceptual
        
        # Embedding constraint loss (only calculated when secret video exists)
        if not cover_only_mode and 'z_cover' in results and 'z_fused' in results:
            z_cover = results['z_cover']
            z_fused = results['z_fused']
            
            # Get cover and fused distribution parameters
            mu_cover = results.get('mu_cover')
            log_var_cover = results.get('log_var_cover')
            mu_fused = results.get('mu_fused')
            log_var_fused = results.get('log_var_fused')
            
            if all(x is not None for x in [mu_cover, log_var_cover, mu_fused, log_var_fused]):
                embedding_loss = self.embedding_constraint_loss(
                    mu_cover, log_var_cover, mu_fused, log_var_fused
                )
                losses['embedding_loss'] = embedding_loss
                total_loss = total_loss + self.lambda_embedding * embedding_loss
            else:
                # If no complete distribution parameters, fallback to simple MSE loss
                embedding_loss = F.mse_loss(z_fused, z_cover)
                losses['embedding_loss'] = embedding_loss
                total_loss = total_loss + self.lambda_embedding * embedding_loss
        
        # VAE KL divergence loss - cover video
        if 'mu_cover' in results and 'log_var_cover' in results:
            mu_cover = results['mu_cover']
            log_var_cover = results['log_var_cover']
            kl_loss_cover = self.kl_divergence_loss(mu_cover, log_var_cover)
            losses['kl_loss_cover'] = kl_loss_cover
            total_loss = total_loss + self.lambda_kl_cover * kl_loss_cover
        
        # VAE KL divergence loss - secret video (only calculated when secret video exists)
        if not cover_only_mode and 'mu_secret' in results and 'log_var_secret' in results:
            mu_secret = results['mu_secret']
            log_var_secret = results['log_var_secret']
            kl_loss_secret = self.kl_divergence_loss(mu_secret, log_var_secret)
            losses['kl_loss_secret'] = kl_loss_secret
            total_loss = total_loss + self.lambda_kl_secret * kl_loss_secret
        
        # Null secret extraction loss in cover-only mode
        if cover_only_mode and 'secret_reconstructed' in results and 'secret_target' in results:
            secret_reconstructed = results['secret_reconstructed']
            secret_target = results['secret_target']  # Should be zero video
            
            # Ensure shape matching
            if secret_reconstructed.shape != secret_target.shape:
                if secret_reconstructed.shape[2] != secret_target.shape[2]:
                    min_t = min(secret_reconstructed.shape[2], secret_target.shape[2])
                    secret_reconstructed = secret_reconstructed[:, :, :min_t]
                    secret_target = secret_target[:, :, :min_t]
            
            # Calculate loss between reconstructed "secret" video and zero video
            null_secret_loss = self.recon_loss(secret_reconstructed, secret_target)
            
            if torch.isnan(null_secret_loss) or torch.isinf(null_secret_loss):
                print("Warning: Null secret loss is NaN or Inf, setting to 0")
                null_secret_loss = torch.tensor(0.0, device=cover_video.device, requires_grad=True)

            losses['null_secret_loss'] = null_secret_loss
            total_loss = total_loss + self.lambda_null_secret * null_secret_loss
        
        # Check total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Total loss is NaN or Inf, using reconstruction loss only")
            total_loss = losses.get('cover_recon_loss', torch.tensor(1.0, device=cover_video.device, requires_grad=True))
        
        losses['total_loss'] = total_loss
        losses['cover_only_mode'] = cover_only_mode  # Record mode

        return losses


def create_loss_functions(config=None):
    """
    Factory function to create loss functions - Supporting mixed training and VAE losses
    """
    if config is None:
        config = {
            'lambda_recon_cover': 1.0,
            'lambda_recon_secret': 1.0,
            'lambda_perceptual': 0.05,  
            'lambda_embedding': 0.1,
            'lambda_cover_only': 1.2,
            'lambda_kl_cover': 0.05,      # Cover VAE KL divergence loss weight
            'lambda_kl_secret': 0.05,     # Secret VAE KL divergence loss weight
            'lambda_null_secret': 1.0,     # Null secret loss weight for cover-only mode
        }
    
    # Extract parameters needed for Loss from config
    loss_params = {
        'lambda_recon_cover': config.get('lambda_recon_cover', 1.0),
        'lambda_recon_secret': config.get('lambda_recon_secret', 1.0),
        'lambda_perceptual': config.get('lambda_perceptual', 0.05),
        'lambda_embedding': config.get('lambda_embedding', 0.1),
        'lambda_cover_only': config.get('lambda_cover_only', 1.2),
        'lambda_kl_cover': config.get('lambda_kl_cover', 0.05),
        'lambda_kl_secret': config.get('lambda_kl_secret', 0.05),
        'lambda_null_secret': config.get('lambda_null_secret', 1.0),
    }
    
    main_loss = TotalLoss(**loss_params)
    
    return main_loss
