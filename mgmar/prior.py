"""Prior estimation: an encoder-decoder predicts the prior image from the uncorrected FBP image and
the metal-artifact image mu_MA; the decoder is then refined per case on the measured sinogram."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import iradon
from torch_radon import Radon

IMAGE_SIZE = 543


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))

    def forward(self, x):
        return self.body(x) + x


class Encoder(nn.Module):
    """EDSR-baseline feature extractor (no upsampling)."""

    def __init__(self, in_channels=2, n_feats=64, n_blocks=16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))

    def forward(self, x):
        x = self.head(x)
        return self.body(x) + x


class Decoder(nn.Module):
    """MLP mapping (3x3 local features, pixel coordinate, cell size) to the attenuation value."""

    def __init__(self, in_dim=64 * 9 + 4, hidden=256, n_hidden=4):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden
        layers = [m for i in range(n_hidden) for m in (nn.Linear(dims[i], dims[i + 1]), nn.ReLU())]
        self.layers = nn.Sequential(*layers, nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x.view(-1, x.shape[-1]))


def decoder_input(features):
    """[1, C, H, W] encoder features -> [H*W, 9C + 4] decoder inputs, one per pixel."""
    h, w = features.shape[-2:]
    local = F.unfold(features, 3, padding=1)[0].t()
    ys, xs = [-1 + 1 / n + 2 / n * torch.arange(n).float() for n in (h, w)]
    coord = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1).view(-1, 2)
    cell = torch.tensor([2 / h, 2 / w]) * torch.tensor([h, w])
    return torch.cat([local, coord.to(local), cell.to(local).expand_as(coord)], dim=-1)


def fbp_parallel(sino):
    """FBP of a 360-degree parallel sinogram [view, bin] on the 543 x 543 grid of the prior network."""
    theta = np.linspace(0., 360., sino.shape[0], endpoint=False)
    return iradon(sino.T, theta=theta, filter_name='ramp', output_size=IMAGE_SIZE, circle=False).astype(np.float32)


def estimate_prior(encoder, decoder, image, mu_ma, sino, trace, steps, lr=5e-6, image_weight=10.0):
    """Prior sinogram [view, bin] from the refined prior image.

    image, mu_ma : 543 x 543 network inputs (uncorrected FBP, metal-artifact image)
    sino, trace  : measured parallel sinogram and its metal trace, [view, bin]
    The decoder is refined for `steps` Adam iterations to match the sinogram outside the trace
    while staying close to its initial prediction; the encoder is fixed.
    """
    device = next(encoder.parameters()).device
    n_views, n_bins = sino.shape
    radon = Radon(IMAGE_SIZE, np.linspace(0., 2 * np.pi, n_views, endpoint=False), clip_to_circle=False,
                  det_count=n_bins)
    decoder = copy.deepcopy(decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    valid = torch.from_numpy(~trace).to(device).view(-1)
    measured = torch.from_numpy(sino).to(device).view(-1, 1)[valid]

    def predict():
        return decoder(x).clamp_(0, 1).view(1, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        x = decoder_input(encoder(torch.from_numpy(np.stack([image, mu_ma]))[None].to(device)))
        initial = predict()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = predict()
        data_loss = (radon.forward(pred).view(-1, 1)[valid] - measured).abs().mean()
        loss = data_loss + image_weight * (pred - initial).abs().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return radon.forward(predict())[0].cpu().numpy()
