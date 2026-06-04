import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# ==========================================
# 1. THE VISION ENCODER (The "Eyes")
# ==========================================
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        # Load a pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Strip the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Compress the single camera feed (512) into a feature vector
        self.compress = nn.Linear(512, feature_dim)
        
        # Standard ImageNet normalization so the pre-trained weights work correctly
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, img):
        # Image must be scaled to [0, 1] before passing here
        img = self.normalize(img)
        
        feat = self.backbone(img).squeeze(-1).squeeze(-1) # (Batch, 512)
        return self.compress(feat) # (Batch, feature_dim)

# ==========================================
# 2. THE NOISE SCHEDULER (DDPM)
# ==========================================
class DDPMScheduler:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=2e-2):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_actions, noise, timesteps):
        """ Used during TRAINING to corrupt the perfect human actions """
        device = original_actions.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1)
        
        return sqrt_alpha_cumprod * original_actions + sqrt_one_minus_alpha_cumprod * noise

    def step(self, model_output, timestep, sample, inference=False):
        """ Used to clean the noise and generate the actual robot movement """
        t = timestep
        device = sample.device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # Calculate the predicted original clean action
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # Compute variance
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        
        # Step backward
        pred_sample_direction = (1 - alpha_prod_t_prev - variance) ** (0.5) * model_output
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        # 🔴 THE FIX: Never add noise during live robot deployment!
        if t > 0 and not inference:
            noise = torch.randn_like(model_output)
            prev_sample = prev_sample + (variance ** 0.5) * noise
            
        return prev_sample
# ==========================================
# 3. THE 1D U-NET (The "Brain")
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
       
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x): return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x): return self.conv(x)

class Conv1dBlock(nn.Module):
   
    def __init__(self, inp_channels, out_channels, kernel_size, cond_dim=None, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()
        
       
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2) if cond_dim else None

    def forward(self, x, cond=None):
        out = self.conv(x)
        out = self.norm(out)
        
      
        if self.cond_proj is not None and cond is not None:
            scale_shift = self.cond_proj(cond).unsqueeze(-1) 
            scale, shift = scale_shift.chunk(2, dim=1)       
            out = out * (scale + 1.0) + shift
            
        return self.mish(out)

class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim=4, global_cond_dim=260): 
        super().__init__()
        time_dim = 128
        cond_dim = time_dim 
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(global_cond_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.down1 = Conv1dBlock(action_dim, 64, 3, cond_dim=time_dim)
        self.down2 = Downsample1d(64)
        self.down3 = Conv1dBlock(64, 128, 3, cond_dim=time_dim)
        self.down4 = Downsample1d(128)

        self.mid1 = Conv1dBlock(128, 256, 3, cond_dim=time_dim)
        self.mid2 = Conv1dBlock(256, 128, 3, cond_dim=time_dim)

        self.up1 = Upsample1d(128)
        self.up2 = Conv1dBlock(256, 64, 3, cond_dim=time_dim) 
        self.up3 = Upsample1d(64)
        self.up4 = Conv1dBlock(128, 32, 3, cond_dim=time_dim) 
        self.final_conv = nn.Conv1d(32, action_dim, kernel_size=3, padding=1)

    def forward(self, x, time, global_cond):
        t_emb = self.time_mlp(time)
        c_emb = self.cond_mlp(global_cond)
        tc_emb = t_emb + c_emb 

        out1 = self.down1(x, tc_emb)
        out2 = self.down2(out1)
        out3 = self.down3(out2, tc_emb)
        out4 = self.down4(out3)

        mid_out = self.mid1(out4, tc_emb)
        mid_out = self.mid2(mid_out, tc_emb)

        up_out = self.up1(mid_out)
        up_out = torch.cat([up_out, out3], dim=1) 
        up_out = self.up2(up_out, tc_emb)

        up_out = self.up3(up_out)
        up_out = torch.cat([up_out, out1], dim=1) 
        up_out = self.up4(up_out, tc_emb)

        return self.final_conv(up_out)