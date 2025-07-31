
import torch
import torch.nn as nn
from .wan_vae import WanVAE, RMS_norm
from .channel import Channel, PowerNormalization


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
        x = x.reshape(B * H * W, C, T).permute(0, 2, 1)  # [BHW, T, C]

        attn_output, _ = self.attn(x, x, x)  # [BHW, T, C]
        out = attn_output.permute(0, 2, 1).reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]

        return out

class Conv2dBlock(nn.Module):
    """Apply convolution only on H,W dimensions, input/output are both [B, C, T, H, W]"""
    def __init__(self, dim, out_dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  
        x = x.reshape(B * T, C, H, W)  
        assert self.dim == C, f"Expected input channel {self.dim}, but got {C}"
        x = self.conv(x)  
        x = x.reshape(B, T, self.out_dim, H, W).permute(0, 2, 1, 3, 4)  
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.conv1 = Conv2dBlock(dim, hidden_dim, 3)
        self.temp_attn = TemporalSelfAttention(hidden_dim)
        self.attention = nn.Sequential(
            Conv2dBlock(hidden_dim, hidden_dim, 7),
            RMS_norm(hidden_dim, images=False),
            nn.SiLU(),
            Conv2dBlock(hidden_dim, hidden_dim, 5),
            RMS_norm(hidden_dim, images=False),
            nn.SiLU(),
            Conv2dBlock(hidden_dim, hidden_dim, 3),
            nn.SiLU(),
        )
        self.feature_transform = nn.Sequential(
            Conv2dBlock(hidden_dim, hidden_dim, 7),
            RMS_norm(hidden_dim, images=False),
            nn.SiLU(),
            Conv2dBlock(hidden_dim, hidden_dim, 5),
            RMS_norm(hidden_dim, images=False),
            nn.SiLU(),
            Conv2dBlock(hidden_dim, hidden_dim, 3),
            nn.SiLU(),
        )
        self.conv2 = Conv2dBlock(hidden_dim, dim, 3)

    def forward(self, input):
        x = self.conv1(input)
        x = self.temp_attn(x)
        attention_weights = self.attention(x)
        transformed_features = self.feature_transform(x)
        out = attention_weights * transformed_features
        out = self.conv2(out)
        return out + input


class FeatureFusionModule(nn.Module):
    """Feature fusion module to fuse secret video features into cover video features"""
    
    def __init__(self, dim, hidden_dim = 96, depth = 4):
        super().__init__()
        self.dim = dim

        self.pos_encoding = nn.Sequential(
            Conv2dBlock(dim, dim, 3),
            RMS_norm(dim, images=False),
            nn.SiLU(),
            Conv2dBlock(dim, dim, 3),
            RMS_norm(dim, images=False),
            nn.SiLU()
        )

        self.block = nn.Sequential(
            *[AttentionBlock(dim*2, hidden_dim) for _ in range(depth)]
        )

        # Main feature fusion output
        self.out_proj = Conv2dBlock(dim*2, dim*2, 3)

        
    def forward(self, z_cover, z_secret):
        """
        Fuse cover and secret features
        Args:
            z_cover: Cover video features [B, dim, T, H, W]
            z_secret: Secret video features [B, dim, T, H, W]
        Returns:
            z_fused: Fused features [B, dim, T, H, W]
            mu_fused: Fused mean [B, dim, T, H, W]
            log_var_fused: Fused log variance [B, dim, T, H, W]
        """

        z_cover = z_cover + self.pos_encoding(z_cover)
        z_secret = z_secret + self.pos_encoding(z_secret)
        z_concat = torch.cat([z_cover, z_secret], dim=1)
        
        features = self.block(z_concat)
        
        mu_fused, log_var_fused = self.out_proj(features).chunk(2, dim=1)
        
        return mu_fused, log_var_fused


class SecretExtractionNetwork(nn.Module):
    """Secret extraction network to extract hidden information from fused features"""
    
    def __init__(self, dim, hidden_dim = 96, depth = 4):
        super().__init__()
        self.dim = dim

        self.pos_encoding = nn.Sequential(
            Conv2dBlock(dim, dim, 3),
            RMS_norm(dim, images=False),
            nn.SiLU(),
            Conv2dBlock(dim, dim, 3),
            RMS_norm(dim, images=False),
            nn.SiLU()
        )
        
        self.block = nn.Sequential(
            *[AttentionBlock(dim, hidden_dim) for _ in range(depth)]
        )
        self.out_proj = nn.Sequential(
            Conv2dBlock(dim, dim, 3),
            RMS_norm(dim, images=False),
            nn.Sigmoid()
        )
    
    def forward(self, z_fused):
        """
        Extract hidden information from fused features
        Args:
            z_fused: Fused features [B, dim, T, H, W]
        Returns:
            z_secret_extracted: Extracted secret features [B, dim, T, H, W]
        """
        z_fused = z_fused + self.pos_encoding(z_fused)
        z_secret_extracted = self.block(z_fused)
        z_secret_extracted = self.out_proj(z_secret_extracted)
        
        return z_secret_extracted


class SemCovert(nn.Module):
    """
    Video semantic covert network
    """
    
    def __init__(self, 
                 vae=None,
                 vae_config=None,
                 depth = 4 ,
                 dim = 96,
                 use_channel=False,
                 channel_config=None):
        super().__init__()
        
        # If pretrained VAE is provided, use it directly
        if vae is not None:
            self.vae = vae
            self.z_dim = vae.z_dim  # Get z_dim from VAE
        else:
            # Otherwise create new VAE from config
            if vae_config is None:
                raise ValueError("vae_config must be provided if vae is not given.")
            self.z_dim = vae_config['z_dim']
            # Shared encoder and decoder
            self.vae = WanVAE(**vae_config)
        
        self.use_channel = use_channel
        
        # Feature fusion module
        self.fusion_module = FeatureFusionModule(
            dim=self.z_dim, 
            hidden_dim = dim,
            depth=depth
        )
        
        # Secret extraction network
        self.extraction_module = SecretExtractionNetwork(
            dim=self.z_dim,
            hidden_dim = dim,
            depth=depth
        )
        
        # Optional channel simulation
        if use_channel:
            if channel_config is None:
                raise ValueError("channel_config must be provided if use_channel is True.")
            self.channel = Channel(**channel_config)
            self.power_norm = PowerNormalization()
    
    def encode_videos(self, cover_video, secret_video=None):
        """
        Encode cover and secret videos
        Args:
            cover_video: Cover video [B, 3, T, H, W]
            secret_video: Secret video [B, 3, T, H, W] (optional)
        Returns:
            Encoding results dict containing z features and mu, log_var for VAE loss calculation
        """
        # Encode cover video
        mu_cover, log_var_cover = self.vae.encode(cover_video)
        
        # Decide whether to resample based on training/inference mode
        if self.training:
            # Training mode: use resampling to maintain randomness and regularization
            z_cover = self.vae.reparameterize(mu_cover, log_var_cover)
        else:
            # Inference mode: directly use mean for deterministic results
            z_cover = mu_cover
        
        results = {
            'z_cover': z_cover,
            'mu_cover': mu_cover,
            'log_var_cover': log_var_cover,
        }
        
        if secret_video is not None:
            # Encode secret video
            mu_secret, log_var_secret = self.vae.encode(secret_video)
            
            # Decide whether to resample based on training/inference mode
            if self.training:
                # Training mode: use resampling
                z_secret = self.vae.reparameterize(mu_secret, log_var_secret)
            else:
                # Inference mode: directly use mean
                z_secret = mu_secret
            
            results.update({
                'z_secret': z_secret,
                'mu_secret': mu_secret,
                'log_var_secret': log_var_secret,
            })
        
        return results
    
    def fuse_features(self, z_cover, z_secret):
        """
        Feature fusion
        Args:
            z_cover: Cover video features [B, dim, T, H, W]
            z_secret: Secret video features [B, dim, T, H, W]
        Returns:
            z_fused: Fused features
            mu_fused: Fused mean
            log_var_fused: Fused log variance
        """

        
        mu_fused, log_var_fused = self.fusion_module(z_cover, z_secret)
   
        std = torch.exp(0.5 * log_var_fused)
        eps = torch.randn_like(std)
        z_fused = eps * std + mu_fused
        
        # Decide which features to return based on training/inference mode
        if self.training:
            # Training mode: use resampled fused features
            return z_fused, mu_fused, log_var_fused
        else:
            # Inference mode: directly use mean as fused features for deterministic results
            return mu_fused, mu_fused, log_var_fused
    
    def transmit_through_channel(self, z_fused):
        if self.use_channel:
            z_normalized = self.power_norm(z_fused)
            z_received = self.channel(z_normalized)
            return z_received
        return z_fused
    
    def set_channel(self, channel_cfg):
        if not self.use_channel:
            raise ValueError("Channel is not enabled in this network.")
        self.channel = Channel(**channel_cfg)
        print(f"Channel set with new config: {channel_cfg}")
    
    def extract_secret(self, z_received):
        return self.extraction_module(z_received)
    
    def decode_videos(self, z_cover_received, z_secret_extracted=None):
        """
        Decode videos
        Args:
            z_cover_received: Received cover features
            z_secret_extracted: Extracted secret features (optional)
        Returns:
            Decoding results dict
        """
        # Decode cover video
        cover_reconstructed = self.vae.decode(z_cover_received)
        
        results = {
            'cover_reconstructed': cover_reconstructed
        }
        
        if z_secret_extracted is not None:
            # Decode secret video
            secret_reconstructed = self.vae.decode(z_secret_extracted)
            results['secret_reconstructed'] = secret_reconstructed
        
        return results
    
    def forward(self, cover_video, secret_video=None):
        """
        Forward pass
        Args:
            cover_video: Cover video [B, 3, T, H, W]
            secret_video: Secret video [B, 3, T, H, W] (optional)
        Returns:
            Complete forward pass results
        """
        # 1. Encoding stage
        encode_results = self.encode_videos(cover_video, secret_video)
        z_cover = encode_results['z_cover']
        
        results = encode_results.copy()
        
        if secret_video is not None:
            z_secret = encode_results['z_secret']
            
            # 2. Feature fusion
            z_fused, mu_fused, log_var_fused = self.fuse_features(z_cover, z_secret)
            results['z_fused'] = z_fused
            results['mu_fused'] = mu_fused
            results['log_var_fused'] = log_var_fused
            
            # 3. Channel transmission
            z_received = self.transmit_through_channel(z_fused)
            results['z_received'] = z_received
            
            # 4. Secret extraction
            z_secret_extracted = self.extract_secret(z_received)
            results['z_secret_extracted'] = z_secret_extracted
            
            # 5. Decoding
            decode_results = self.decode_videos(z_received, z_secret_extracted)
            
        else:
            # Cover-only mode: no secret information
            z_received = self.transmit_through_channel(z_cover)
            results['z_received'] = z_received
            
            # In cover-only mode, also perform secret extraction but expect zeros
            z_secret_extracted = self.extract_secret(z_received)
            results['z_secret_extracted'] = z_secret_extracted
            
            # Decode extracted "secret" features, expecting zero video
            secret_reconstructed = self.vae.decode(z_secret_extracted)
            results['secret_reconstructed'] = secret_reconstructed
            
            # Create zero video as target (same shape as input video)
            secret_target = torch.zeros_like(cover_video)
            results['secret_target'] = secret_target
            
            # Decode cover video
            decode_results = self.decode_videos(z_received)
        
        results.update(decode_results)
        
        return results
    

def create_network(network_config, pretrained_vae=None):

    print("\n--- Creating  Network ---")
    
    network = SemCovert(
        vae=pretrained_vae,
        vae_config=network_config.get('vae_config', None),
        depth=network_config.get('depth', 4), 
        dim = network_config.get('dim', 96),  
        use_channel=network_config.get('use_channel', True),
        channel_config=network_config.get('channel_config', None)
    )
    
    return network