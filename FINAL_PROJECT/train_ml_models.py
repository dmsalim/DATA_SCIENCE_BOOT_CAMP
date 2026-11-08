import os
import json
import numpy               as np
import torch
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F
import pandas              as pd

from   torch.utils.data    import DataLoader
from   sevir_utils         import SEVIRVILDataset

def denormalize(t):
    # Map [-1, 1] back to the original 0-255 VIL scale.
    return (t + 1) / 2 * 255

def csi(pred, target, threshold):
    # pred, target: tensors of VIL values (denormalized back to 0-255 scale).
    pred_bin     = (pred > threshold)
    target_bin   = (target > threshold)
    hits         = (pred_bin & target_bin).sum().item()  # predicted rain,    actually rained
    misses       = (~pred_bin & target_bin).sum().item() # predicted no rain, actually rained
    false_alarms = (pred_bin & ~target_bin).sum().item() # predicted rain,    actually didn’t rain
    total_rain   = hits + misses + false_alarms          # every pixel that was “rain” according to at least one of the two (prediction or truth) — i.e., the union of predicted-rain pixels and actual-rain pixels.
    csi_value    = hits / total_rain if total_rain > 0 else float("nan")
    return csi_value

def persistence_forecast(x):
    """x: (B, 13, H, W) -> predicts all 12 future frames as the last input frame repeated."""
    last_frame = x[:, -1:, :, :]  # (B, 1, H, W)
    return last_frame.repeat(1, 12, 1, 1)

def evaluate_csi(pred_norm, target_norm, thresholds=[16, 74, 133, 160, 181, 219]):
    pred_raw = denormalize(pred_norm)
    target_raw = denormalize(target_norm)
    return {t: csi(pred_raw, target_raw, t) for t in thresholds}

def evaluate_csi_full(model_fn, loader, thresholds, device="cpu"):
    # model_fn: a function that takes x and returns a prediction (handles any model's calling convention).
    totals = {t: {"hits": 0, "misses": 0, "false_alarms": 0} for t in thresholds}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model_fn(x)
        pred_raw = denormalize(pred)
        target_raw = denormalize(y)

        for t in thresholds:
            pred_bin = pred_raw > t
            target_bin = target_raw > t
            totals[t]["hits"] += (pred_bin & target_bin).sum().item()
            totals[t]["misses"] += (~pred_bin & target_bin).sum().item()
            totals[t]["false_alarms"] += (pred_bin & ~target_bin).sum().item()

    results = {}
    for t, counts in totals.items():
        denom = counts["hits"] + counts["misses"] + counts["false_alarms"]
        results[t] = counts["hits"] / denom if denom > 0 else float("nan")
    return results

#######################################################################
### --- Simple Convolutional Baseline Model for VIL Forecasting --- ###
#######################################################################

class SimpleConvBaseline(nn.Module):
    def __init__(self, input_len=13, output_len=12, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_len, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, output_len, 3, padding=1),
        )

    def forward(self, x):
        # x: (B, input_len, H, W) — treat the 13 time steps as channels
        z = self.encoder(x)
        return self.decoder(z)


#################################################
### --- Simplified Space-Time Transformer --- ###
#################################################

class PatchEmbed(nn.Module):
    """converting input tensor into a sequence: Splits an image into non-overlapping patches and projects each to an embedding."""
    def __init__(self, img_size=384, patch_size=16, in_chans=13, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, in_chans, H, W) -> (B, n_patches, embed_dim)
        #B = batch, n_patches =  sequence length, embed_dim = embedding dimension
        x = self.proj(x)                       # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)       # (B, n_patches, embed_dim); sequence of 576 tokens, each a 256-dim vector
        return x

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block: self-attention + feedforward."""
    def __init__(self, embed_dim=256, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True) # for every patch token, it looks at every other patch token and decides how much attention to pay to each one when updating its own representation.
        self.norm2 = nn.LayerNorm(embed_dim) # normalizes each token’s vector independently - analogous to BatchNorm in CNNs (but normalizing across the feature dimension per-token rather than across the batch).
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm residual attention
        # self-attention: the same tensor (normed) is passed as query, key, and value — every token asks “which other tokens are relevant to me” and pulls information from them.
        normed      = self.norm1(x)
        # residual connection: add the attention output back to the original input rather than replacing it. (makes very deep networks trainable by giving gradients a direct path backward.)
        attn_out, _ = self.attn(normed, normed, normed)
        x           = x + attn_out
        # Pre-norm residual feedforward
        x           = x + self.mlp(self.norm2(x))
        return x

class SpaceTimeTransformer(nn.Module):
    # Simplified ViT-style nowcasting model: patchify the 13 input frames,
    # run self-attention over the resulting patch tokens, 
    # then project back to pixel space to predict all 12 output frames at once.
    def __init__(self, img_size=384, patch_size=16, input_len=13, output_len=12,
                 embed_dim=256, depth=6, n_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size          # e.g. 384/16 = 24
        self.n_patches  = self.grid_size ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, input_len, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stack depth=6 transformer blocks back to back — this is the “depth” of the network, analogous to stacking more conv layers in a CNN. Each block lets tokens exchange information with each other once more.
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project each patch token back to a (patch_size x patch_size x output_len) pixel block
        self.head       = nn.Linear(embed_dim, patch_size * patch_size * output_len)
        self.output_len = output_len

    def forward(self, x):
        # x: (B, input_len, H, W)
        B      = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed     # (B, n_patches, embed_dim)

        for block in self.blocks:
            tokens = block.forward(tokens)
        tokens = self.norm(tokens)

        out = self.head(tokens)  # (B, n_patches, patch_size*patch_size*output_len)

        # Reshape patch tokens back into a full image: (B, output_len, H, W)
        out = out.view(B, self.grid_size, self.grid_size,
                        self.patch_size, self.patch_size, self.output_len)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, self.output_len, self.grid_size * self.patch_size,
                        self.grid_size * self.patch_size)
        return torch.tanh(out)  # match your [-1, 1] normalized targets

#############################################################################
### --- Residual diffusion conditioned on frozen transformer backbone --- ###
#############################################################################

### --------------------------------------------- ###
### --- 1. Noise schedule and forward process --- ###
### --------------------------------------------- ###

def make_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    # Linear noise schedule — how much noise gets added at each diffusion step.
    betas = torch.linspace(beta_start, beta_end, timesteps)
    # --> List of 1000 numbers, starting small (1e-4) and increasing linearly to 0.02. 
    #     Each beta_t controls how much noise gets mixed in at step t. Early steps barely perturb the data; later steps add much more noise
    return betas

class GaussianDiffusion:
    # Implements the DDPM forward (noising) process and the reverse (denoising) sampling loop. 
    # Does not contain the denoiser network itself — that's passed in separately so the same diffusion machinery can wrap any architecture.

    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device    = device

        betas          = make_beta_schedule(timesteps).to(device)
        alphas         = 1.0 - betas                  # fraction of the original signal retained at each individual step.
        alphas_cumprod = torch.cumprod(alphas, dim=0) # cumulative product of all alphas up to step t — i.e., how much of the original signal survives after t steps of noise have been layered on, compounding. 
                                                        # lets you jump directly from the clean data to any noise level in one shot, without simulating all the intermediate steps one by one

        self.betas                         = betas
        self.alphas_cumprod                = alphas_cumprod
        self.sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # q_sample: the forward process — gradually corrupt a clean residual with Gaussian noise.
    # p_sample: one reverse step — ask the network “what noise did you add?” and subtract an estimate of it.
    # sample:   chains many reverse steps together, starting from pure noise and ending at a (hopefully) clean residual.

    def q_sample(self, x0, t, noise=None):
        # Forward process: add noise to a clean data sample x0 at timestep t steps of noise.
        # x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        # returns both x_t (the noised version) and the exact noise that was added (that noise becomes training label).
        
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) # .view(-1, 1, 1, 1) reshapes to match (B, C, H, W) image tensor.
        sqrt_one_minus_ac   = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t                 = sqrt_alphas_cumprod * x0 + sqrt_one_minus_ac * noise # weighted mix of the original clean signal and pure random Gaussian noise
                                                                                    # at t=0,   alphas_cumprod close to 1, so x_t ≈ x0 (barely noised);
                                                                                    # at t=999, alphas_cumprod close to 0, so x_t ≈ noise (almost pure noise).
        return x_t, noise

    def p_sample(self, denoiser, x_t, t, cond):
        # Reverse process: one denoising step, from x_t to x_{t-1}.
        # Uses the model to predict the noise, then removes an estimate of it.
        
        betas_t            = self.betas[t].view(-1, 1, 1, 1)
        sqrt_om_ac_t       = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t            = (1.0 - betas_t)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

        # given the network’s noise prediction, estimate what the slightly-less-noisy version at step t-1 should look like.
        predicted_noise    = denoiser(x_t, t, cond)

        model_mean         = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_om_ac_t * predicted_noise)

        if (t == 0).all():
            return model_mean
        else:
            noise    = torch.randn_like(x_t)
            variance = betas_t
            # At every step except the very last (t=0), add a small amount of fresh random noise back in before returning.
            return model_mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, denoiser, shape, cond, device):
        # Full reverse process: start from pure noise, denoise step by step to t=0.
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(denoiser, x, t, cond)
        return x

### ----------------------------------------------------------- ###
### --- 2. The denoiser network — a small conditioned U-Net --- ###
### ----------------------------------------------------------- ###

class TimestepEmbedding(nn.Module):
# Sinusoidal embedding of the diffusion timestep t, same idea as positional embeddings in transformers — lets the network know 'how noisy' the input is.
# Converts the integer t into a richer vector using sine/cosine waves at different frequencies
# Result: vector that smoothly and uniquely encodes “which timestep this is,” which the network can then use.
# (Required because denoiser network needs to behave differently depending on whether it’s looking at barely-noised data (t small) or almost-pure noise (t large) — the right “how much noise to remove” answer depends on t.)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs    = torch.exp(-np.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args     = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ConvBlock(nn.Module):
    # a conv layer that’s aware of the timestep
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(8, out_ch) # nn.GroupNorm: plays the same role nn.LayerNorm/BatchNorm played elsewhere — a normalization step for training stability (just a variant more commonly used in image-generation U-Nets).
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act       = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]  # inject timestep info; self.time_proj(t_emb) projects the timestep embedding into a vector matching the number of channels, 
                                                            #                       then adds it to every spatial position. ([:, :, None, None] broadcasts it across height and width.)
                                                            # Injects “which noise level are we at” information directly into the convolutional feature maps at every block (standard way timestep conditioning is wired into a U-Net).
        h = self.act(self.norm2(self.conv2(h)))
        return h

class ResidualDenoiser(nn.Module):
    # The actual network being trained
    # Small U-Net that predicts the noise added to the residual, conditioned on:
    #      - the noisy residual itself (x_t)
    #      - the diffusion timestep t
    #      - the deterministic backbone's forecast + the original input frames (cond)
    def __init__(self, output_len=12, input_len=13, time_dim=128, hidden=64):
        super().__init__()
        self.time_dim   = time_dim
        self.time_embed = nn.Sequential(
            TimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input: noisy residual (output_len channels) + backbone forecast (output_len)
        # + original input frames (input_len), all concatenated as channels
        in_ch         = output_len + output_len + input_len

        self.down1    = ConvBlock(in_ch, hidden, time_dim)
        self.pool     = nn.MaxPool2d(2)
        self.down2    = ConvBlock(hidden, hidden * 2, time_dim)

        self.up       = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.up_block = ConvBlock(hidden * 2, hidden, time_dim)  # concat skip connection

        self.out_conv = nn.Conv2d(hidden, output_len, 1)

    def forward(self, x_t, t, cond):
        # conditional denoiser - generating a residual that’s consistent with what the transformer already predicted and what the storm actually looked like going in.
        # given a noisy residual, the timestep, and this conditioning information, output a prediction of what noise was added (not the clean residual directly)
        #   x_t:  (B, output_len, H, W)             — the noisy residual
        #   t:    (B,)                              — diffusion timestep
        #   cond: (B, output_len + input_len, H, W) — backbone forecast + input frames; (as wired up in train_diffusion) contains the backbone’s deterministic forecast plus the original input frames.
        
        t_emb = self.time_embed(t)

        h     = torch.cat([x_t, cond], dim=1)
        h1    = self.down1(h, t_emb)
        h2    = self.pool(h1)
        h2    = self.down2(h2, t_emb)

        h_up  = self.up(h2)
        h_up  = torch.cat([h_up, h1], dim=1)  # skip connection carrying detail from the “down” path directly to the “up” path
        h_up  = self.up_block(h_up, t_emb)

        return self.out_conv(h_up)

### --------------------------------- ###
### --- 3. Residual training loop --- ###
### --------------------------------- ###

def train_diffusion(frozen_backbone, denoiser, diffusion, train_loader,
                    num_epochs=10, lr=1e-4, device="cpu"):
    denoiser.to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # get the frozen transformer’s forecast (no gradients needed — it’s not being trained anymore), & compute the residual (the actual error the transformer made).
            with torch.no_grad():
                backbone_forecast = frozen_backbone(x)    # (B, output_len, H, W), frozen
            residual = y - backbone_forecast              # what the diffusion model must learn to generate

            # For each training example in the batch, pick a random timestep t, then noise the residual to that level using q_sample.
            B                     = x.shape[0]
            t                     = torch.randint(0, diffusion.timesteps, (B,), device=device)
            noisy_residual, noise = diffusion.q_sample(residual, t)

            cond                  = torch.cat([backbone_forecast, x], dim=1)  # conditioning info
            predicted_noise       = denoiser(noisy_residual, t, cond)

            loss                  = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch}: avg_loss={running_loss / len(train_loader):.4f}")

    return denoiser

### ---------------------------------------------------------- ###
### --- 4. Inference: backbone forecast + sampled residual --- ###
### ---------------------------------------------------------- ###
@torch.no_grad()
def diffusion_forecast(frozen_backbone, denoiser, diffusion, x, device="cpu"):
    x                 = x.to(device)
    backbone_forecast = frozen_backbone(x)
    cond              = torch.cat([backbone_forecast, x], dim=1)

    shape             = backbone_forecast.shape
    sampled_residual  = diffusion.sample(denoiser, shape, cond, device)

    return backbone_forecast + sampled_residual


if __name__ == "__main__":

    save_training_filepath = os.path.join(os.getcwd(), "CHECKPOINTS")

    train_catalog = pd.read_csv("vil_train_catalog.csv", parse_dates=["time_utc"])
    train_dataset = SEVIRVILDataset(train_catalog, data_dir="data")
    train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    test_catalog  = pd.read_csv("vil_test_catalog.csv", parse_dates=["time_utc"])
    test_dataset  = SEVIRVILDataset(test_catalog, data_dir="data")
    test_loader   = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

    x_sample_test, y_sample_test = next(iter(test_loader))

    num_epochs    = 10
    thresholds    = [16, 74, 133, 160, 181, 219]
    device        = "cuda" if torch.cuda.is_available() else "cpu"

    #######################################################################
    ### --- Simple Convolutional Baseline Model for VIL Forecasting --- ###
    #######################################################################

    model_CNN     = SimpleConvBaseline()
    optimizer_CNN = optim.Adam(model_CNN.parameters(), lr=1e-3)
    loss_fn_CNN   = nn.MSELoss()

    # Persistence baseline, on one batch, before any training
    x_sample, y_sample = next(iter(train_loader))
    persistence_pred   = persistence_forecast(x_sample)
    print("Persistence CSI:", evaluate_csi(persistence_pred, y_sample, thresholds=thresholds))

    cnn_losses = []
    for epoch in range(num_epochs):
        running_loss_CNN = 0.0
        for x, y in train_loader:
            optimizer_CNN.zero_grad()
            pred_CNN          = model_CNN(x)
            loss_CNN          = loss_fn_CNN(pred_CNN, y)
            loss_CNN.backward()
            optimizer_CNN.step()
            running_loss_CNN += loss_CNN.item()

        avg_loss_CNN = running_loss_CNN / len(train_loader)
        cnn_losses.append(avg_loss_CNN)
        print(f"Epoch {epoch}: avg_loss={avg_loss_CNN:.4f}")
    np.save(os.path.join(save_training_filepath, "cnn_losses.npy"), np.array(cnn_losses))

    # Model CSI after training, same batch for a quick before/after comparison
    model_CNN.eval()
    with torch.no_grad():
        model_CNN_pred = model_CNN(x_sample)
    print("Model CSI:", evaluate_csi(model_CNN_pred, y_sample, thresholds=thresholds))

    #################################################
    ### --- Simplified Space-Time Transformer --- ###
    #################################################

    model_TRANS       = SpaceTimeTransformer()
    optimizer_TRANS   = torch.optim.Adam(model_TRANS.parameters(), lr=1e-4)
    loss_fn_TRANS     = nn.MSELoss()

    print("Persistence CSI:", evaluate_csi(persistence_forecast(x_sample), y_sample, thresholds=thresholds))

    trans_losses = []
    for epoch in range(num_epochs):
        running_loss_TRANS = 0.0
        for x, y in train_loader:
            optimizer_TRANS.zero_grad()
            pred_TRANS          = model_TRANS(x)
            loss_TRANS          = loss_fn_TRANS(pred_TRANS, y)
            loss_TRANS.backward()
            optimizer_TRANS.step()
            running_loss_TRANS += loss_TRANS.item()
        avg_loss_TRANS = running_loss_TRANS / len(train_loader)
        trans_losses.append(avg_loss_TRANS)
        print(f"Epoch {epoch}: avg_loss={avg_loss_TRANS:.4f}")
    np.save(os.path.join(save_training_filepath, "trans_losses.npy"), np.array(trans_losses))
    
    model_TRANS.eval()
    with torch.no_grad():
        transformer_pred = model_TRANS(x_sample)
    print("Transformer CSI:", evaluate_csi(transformer_pred, y_sample, thresholds=thresholds))

    #############################################################################
    ### --- Residual diffusion conditioned on frozen transformer backbone --- ###
    #############################################################################

    # 1. Freeze already-trained transformer
    for p in model_TRANS.parameters():
        p.requires_grad = False
    model_TRANS.eval()
    model_TRANS.to(device)

    # 2. Build the diffusion machinery and denoiser
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    denoiser  = ResidualDenoiser(output_len=12, input_len=13)

    # 3. Train
    denoiser = train_diffusion(model_TRANS, denoiser, diffusion, train_loader,
                                num_epochs=10, lr=1e-4, device=device)

    # 4. Evaluate with CSI (like other models)
    diff_pred          = diffusion_forecast(model_TRANS, denoiser, diffusion, x_sample, device=device)
    print("Diffusion CSI:", evaluate_csi(diff_pred.cpu(), y_sample, thresholds=thresholds))

    model_CNN.to(device)
    diffusion_forward_function = lambda x: diffusion_forecast(model_TRANS, denoiser, diffusion, x, device=device)
    
    cnn_csi = evaluate_csi_full(model_CNN, test_loader, thresholds, device=device)
    trans_csi = evaluate_csi_full(model_TRANS, test_loader, thresholds, device=device)
    diff_csi = evaluate_csi_full(diffusion_forward_function, test_loader, thresholds, device=device)

    print("CNN CSI (full val):", cnn_csi)
    print("Transformer CSI (full val):", trans_csi)
    print("Diffusion CSI (full val):", diff_csi)

    results = {"cnn": cnn_csi, "transformer": trans_csi, "diffusion": diff_csi}

    torch.save(model_CNN.state_dict(),   os.path.join(save_training_filepath, "model_cnn.pt"))
    torch.save(model_TRANS.state_dict(), os.path.join(save_training_filepath, "model_transformer.pt"))
    torch.save(denoiser.state_dict(),    os.path.join(save_training_filepath, "denoiser.pt"))

    with open(os.path.join(save_training_filepath, "csi_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    config = {
        "num_epochs": num_epochs,
        "thresholds": thresholds,
        "diffusion_timesteps": diffusion.timesteps,
    }
    with open(os.path.join(save_training_filepath, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)