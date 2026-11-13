"""Residual correction: a metal-conditioned wavelet U-Net predicts the residual of the NMAR image
patch by patch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_S = torch.tensor([2 ** -0.5, 2 ** -0.5])
_D = torch.tensor([2 ** -0.5, -(2 ** -0.5)])
_HAAR = (_S[None] * _S[:, None], _S[None] * _D[:, None], _D[None] * _S[:, None], _D[None] * _D[:, None])


def haar_split(x):
    """Stride-2 Haar analysis -> (LL, LH, HL, HH)."""
    c = x.shape[1]
    return [F.conv2d(x, k.to(x).expand(c, 1, 2, 2), stride=2, groups=c) for k in _HAAR]


def haar_merge(ll, lh, hl, hh):
    """Stride-2 Haar synthesis."""
    c = ll.shape[1]
    up = [F.conv_transpose2d(v, k.to(v).expand(c, 1, 2, 2), stride=2, groups=c)
          for v, k in zip((ll, lh, hl, hh), _HAAR)]
    return (up[0] + up[1]) + (up[2] + up[3])


def conv_norm(c_in, c_out, stride=1, norm=True):
    layers = [nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)]
    if norm:
        layers.append(nn.InstanceNorm2d(c_out, affine=True))
    return nn.Sequential(*layers)


class MaskEncoder(nn.Module):
    """128 x 128 metal-mask patch -> conditioning code."""

    def __init__(self, n_code):
        super().__init__()
        act = nn.LeakyReLU(0.2)
        self.features = nn.Sequential(conv_norm(1, 1, 2, norm=False), act, conv_norm(1, 2, 2, norm=False), act,
                                      conv_norm(2, 4, 2, norm=False), act, conv_norm(4, 1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, n_code))

    def forward(self, mask):
        return self.fc(self.features(mask))


class ModulatedConv(nn.Module):
    """Convolution + instance norm, scaled and shifted channel-wise by the mask code."""

    def __init__(self, c_in, c_out, n_code, act=True):
        super().__init__()
        self.conv = conv_norm(c_in, c_out)
        self.scale = nn.Linear(n_code, c_out)
        self.shift = nn.Linear(n_code, c_out)
        self.act = nn.LeakyReLU(0.2) if act else nn.Identity()

    def forward(self, x, code):
        return self.act(self.conv(x) * self.scale(code)[:, :, None, None] + self.shift(code)[:, :, None, None])


class CorrectionNet(nn.Module):
    def __init__(self, n=32, gain=3.0):
        super().__init__()
        self.gain = gain
        self.mask_encoder = MaskEncoder(n)
        enc = [(1, n), (n, n), (n, 2 * n), (2 * n, 2 * n), (2 * n, 4 * n), (4 * n, 4 * n), (4 * n, 8 * n),
               (8 * n, 8 * n)]
        dec = [(16 * n, 8 * n), (8 * n, 4 * n), (8 * n, 4 * n), (4 * n, 2 * n), (4 * n, 2 * n), (2 * n, n),
               (2 * n, n)]
        self.enc = nn.ModuleList([ModulatedConv(i, o, n, act=k % 2 == 1) for k, (i, o) in enumerate(enc)])
        self.bottom = nn.ModuleList([conv_norm(8 * n, 8 * n) for _ in range(4)])
        self.norm = nn.ModuleList([nn.InstanceNorm2d(c, affine=True) for c in (8 * n, 4 * n, 2 * n, n)])
        self.dec = nn.ModuleList([ModulatedConv(i, o, n, act=k % 2 == 1) for k, (i, o) in enumerate(dec)])
        self.out = nn.Conv2d(n, 1, 3, padding=1, bias=False)

    def forward(self, x, mask):
        code = self.mask_encoder(mask)
        skips, bands = [], []
        for level in range(4):
            x = self.enc[2 * level + 1](self.enc[2 * level](x, code), code)
            skips.append(x)
            x, *high = haar_split(x)
            bands.append(high)
        x = haar_merge(*[conv(b) for conv, b in zip(self.bottom, [x] + bands.pop())])
        for level in range(4):
            x = self.dec[2 * level](torch.cat((self.norm[level](x), skips.pop()), 1), code)
            if level < 3:
                x = haar_merge(self.dec[2 * level + 1](x, code), *bands.pop())
        return torch.tanh(self.gain * self.out(F.relu(x)))


@torch.no_grad()
def correct_residual(net, image, mask, patch=128, stride=16, scale=0.1, batch_size=128):
    """Add the predicted residual to a 512 x 512 NMAR image; overlapping patches are averaged."""
    device = next(net.parameters()).device
    n = image.shape[0]
    x = torch.from_numpy(np.clip(image / scale, -1.0, 1.0))
    m = torch.from_numpy(mask)
    corners = [(j, i) for j in range(0, n - patch + 1, stride) for i in range(0, n - patch + 1, stride)]
    out = []
    for k in range(0, len(corners), batch_size):
        xs = torch.stack([x[j:j + patch, i:i + patch] for j, i in corners[k:k + batch_size]])[:, None]
        ms = torch.stack([m[j:j + patch, i:i + patch] for j, i in corners[k:k + batch_size]])[:, None]
        out.append(net(xs.to(device), ms.to(device))[:, 0].cpu().numpy())
    residual = np.zeros((n, n), np.float32)
    count = np.zeros((n, n), np.float32)
    for (j, i), r in zip(corners, np.concatenate(out)):
        residual[j:j + patch, i:i + patch] += r
        count[j:j + patch, i:i + patch] += 1.0
    return image + residual / count * scale
