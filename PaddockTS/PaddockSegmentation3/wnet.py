import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_c, out_c, base=32 * 2):
        super().__init__()
        self.enc1 = ConvBlock(in_c, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.final = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class WNet(nn.Module):
    def __init__(self, in_c, n_classes, base=32):
        super().__init__()
        self.enc = UNet(in_c, n_classes, base)
        self.dec = UNet(n_classes, in_c, base)

    def forward(self, x):
        seg = F.softmax(self.enc(x), dim=1)
        rec = self.dec(seg)
        return seg, rec


def soft_ncut_loss(seg, features, radius=5, sigma_x=4.0, sigma_i=0.1):
    B, K, H, W = seg.shape
    C = features.shape[1]
    device = seg.device

    r = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(r, r, indexing='ij')
    spatial_kernel = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma_x ** 2))
    spatial_kernel[radius, radius] = 0
    d = 2 * radius + 1

    # feature similarity: ||f_i - f_j||^2 via unfolded patches
    patches = F.unfold(features, d, padding=radius)  # (B, C*d*d, H*W)
    patches = patches.view(B, C, d * d, H, W)
    center = features[:, :, :, :].unsqueeze(2)  # (B, C, 1, H, W)
    diff_sq = ((patches - center) ** 2).sum(dim=1)  # (B, d*d, H, W)
    feat_kernel = torch.exp(-diff_sq / (2 * sigma_i ** 2))  # (B, d*d, H, W)
    feat_kernel = feat_kernel * spatial_kernel.view(1, d * d, 1, 1)

    ncut = torch.tensor(0.0, device=device)
    for k in range(K):
        s = seg[:, k:k + 1]  # (B, 1, H, W)
        s_flat = s.view(B, 1, H * W)
        s_patches = F.unfold(s, d, padding=radius)  # (B, d*d, H*W)
        ws = (feat_kernel.view(B, d * d, H * W) * s_patches).sum(dim=1, keepdim=True)  # (B, 1, H*W)
        assoc = (s_flat * ws).sum()
        w_total = feat_kernel.view(B, d * d, H * W).sum(dim=1, keepdim=True)
        total = (s_flat * w_total).sum()
        if total > 1e-8:
            ncut = ncut + (1.0 - assoc / total)
    return ncut / K



def _build_features(*indices):
    channels = []
    for idx in indices:
        # use all timesteps as channels
        for t in range(idx.shape[2]):
            channels.append(idx[:, :, t])
    features = np.stack(channels, axis=0).astype(np.float32)
    for c in range(features.shape[0]):
        f = features[c]
        finite = np.isfinite(f)
        if finite.any():
            mn, mx = f[finite].min(), f[finite].max()
            if mx > mn:
                f = (f - mn) / (mx - mn)
        features[c] = np.nan_to_num(f, nan=0.0)
    return features


def segment_wnet(
    *indices: NDArray[np.float32],
    n_classes: int = 5,
    max_epochs: int = 500,
    lr: float = 1e-3,
    ncut_weight: float = 1.0,
    patience: int = 20,
    tol: float = 1e-4,
    verbose: bool = True,
) -> NDArray[np.int32]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w, _ = indices[0].shape
    features = _build_features(*indices)

    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    x = torch.from_numpy(features[None]).to(device)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))

    model = WNet(features.shape[0], n_classes).to(device)
    enc_optimizer = torch.optim.Adam(model.enc.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(model.dec.parameters(), lr=lr)
    enc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(enc_optimizer, patience=10, factor=0.5)
    dec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dec_optimizer, patience=10, factor=0.5)

    best_loss = float('inf')
    wait = 0

    for epoch in range(max_epochs):
        # phase 1: train encoder with NCut
        seg = F.softmax(model.enc(x), dim=1)
        ncut = soft_ncut_loss(seg, x)
        enc_optimizer.zero_grad()
        ncut.backward()
        enc_optimizer.step()

        # phase 2: train decoder with reconstruction
        seg = F.softmax(model.enc(x), dim=1).detach()
        rec = model.dec(seg)
        rec_loss = F.mse_loss(rec, x)
        dec_optimizer.zero_grad()
        rec_loss.backward()
        dec_optimizer.step()

        enc_scheduler.step(ncut.item())
        dec_scheduler.step(rec_loss.item())

        total_loss = ncut.item() + rec_loss.item()
        if total_loss < best_loss - tol:
            best_loss = total_loss
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f'  epoch {epoch+1}: rec={rec_loss.item():.4f}, ncut={ncut.item():.4f}, lr={enc_optimizer.param_groups[0]["lr"]:.6f}')

        if wait >= patience:
            if verbose:
                print(f'  converged at epoch {epoch+1} (patience={patience})')
            break

    with torch.no_grad():
        seg = F.softmax(model.enc(x), dim=1)
        labels = seg[0, :, :h, :w].argmax(dim=0).cpu().numpy().astype(np.int32)

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'  Segment {u}: {c:,} pixels ({100 * c / (h * w):.1f}%)')

    return labels
