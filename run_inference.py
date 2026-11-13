"""MGMAR inference on the AAPM CT-MAR final test set.

fan-beam sinogram -> prior (encoder-decoder + refinement) -> NMAR
-> residual correction -> corrected image in HU with the metal inserted.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from mgmar.cases import FOV, METAL_HU, REFINE_STEPS
from mgmar.correction import CorrectionNet, correct_residual
from mgmar.fanbeam import FanBeam
from mgmar.nmar import nmar_fan, nmar_parallel
from mgmar.prior import Decoder, Encoder, estimate_prior, fbp_parallel
from mgmar.rebinning import fan_to_parallel, parallel_to_fan, parallel_trace

ROOT = Path(__file__).resolve().parent


def read(path, shape):
    return np.fromfile(path, np.float32).reshape(shape)


def load_network(net, name, device):
    net.load_state_dict(torch.load(ROOT / 'weights' / name, map_location='cpu'))
    return net.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='folder with XX.metalart.sino_900x1000.raw')
    parser.add_argument('--output', default='submission', help='folder for the corrected images XX.raw')
    parser.add_argument('--cases', type=int, nargs='*', default=sorted(FOV))
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    encoder = load_network(Encoder(), 'prior_encoder.pth', device)
    decoder = load_network(Decoder(), 'prior_decoder.pth', device)
    correction_net = load_network(CorrectionNet(), 'correction_net.pth', device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in args.cases:
        name, fov = f'{k:02d}.raw', FOV[k]
        sino = read(Path(args.input) / f'{k:02d}.metalart.sino_900x1000.raw', (1000, 900))
        mu_ma = read(ROOT / 'data' / 'mu_MA' / name, (543, 543))
        metal = read(ROOT / 'data' / 'M' / name, (512, 512))
        metal_ext = read(ROOT / 'data' / 'M_extended' / name, (512, 512))

        # prior in the parallel geometry
        parallel = fan_to_parallel(sino, fov).astype(np.float32)
        trace = parallel_trace(metal_ext, fov)
        prior = estimate_prior(encoder, decoder, fbp_parallel(parallel), mu_ma, parallel, trace, REFINE_STEPS[k])

        # NMAR with the prior, back in the measured fan geometry
        prior_fan = parallel_to_fan(nmar_parallel(parallel, prior, trace), fov).astype(np.float32)
        fan = FanBeam(fov)
        image = fan.fbp(nmar_fan(sino, prior_fan, fan.project(metal_ext) > 0.01))
        image = correct_residual(correction_net, image, metal_ext)

        # attenuation (1/mm, water = 0.02) -> HU, metal inserted
        hu = (image - 0.02) / 0.02 * 1000
        (hu * (1 - metal) + metal * METAL_HU[k]).astype(np.float32).tofile(out_dir / name)
        print(f'case {k:02d} -> {out_dir / name}', flush=True)


if __name__ == '__main__':
    main()
