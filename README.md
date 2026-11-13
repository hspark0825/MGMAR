# MGMAR

Code for **MGMAR: Metal-Guided Metal Artifact Reduction for X-ray Computed Tomography**
 

## Pipeline

`run_inference.py` takes the measured fan-beam sinogram `XX.metalart.sino_900x1000.raw` of each case and writes the corrected image.

1. **Prior** — an encoder-decoder (conditioned implicit neural representation) predicts the prior image from the uncorrected FBP image and the metal-artifact image μ_MA. The decoder is then refined on the measured projections outside the metal trace.
2. **NMAR** — the sinogram is normalized by the prior, interpolated across the metal trace and reconstructed.
3. **Residual correction** — a metal-conditioned network removes the remaining artifacts.
4. **Output** — the image in HU with the metal inserted, as submitted to the AAPM CT-MAR challenge scoring.

## Contents

| Path | Description |
|---|---|
| `run_inference.py` | entry point |
| `mgmar/` | prior estimation, NMAR, residual correction, fan-beam projector
| `data/M/` | segmented metal masks inserted into the final image |
| `data/submission/` | final corrected CT images (results of `run_inference.py`) for the 29 test cases |
| `scores.txt` | challenge scoring of `data/submission/`: final score and average score of each case |
| `scores.html` | challenge scoring of `data/submission/`: scores of the eight metrics for each case |

All `.raw` files are float32, row-major.

## Environment

Tested on Linux (x86-64) with an NVIDIA RTX 3090. The fan-beam projector is a prebuilt library that needs the GNU OpenMP runtime (`libgomp`).

| Package | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.12.1 (CUDA 11.3) |
| [torch-radon](https://github.com/matteo-ronchetti/torch-radon) | 1.0.0 |
| NumPy / SciPy / scikit-image | 1.24.4 / 1.10.1 / 0.18.1 |

```bash
conda create -n mgmar python=3.8 && conda activate mgmar
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
# install torch-radon following its repository
```

## Usage

Download the final test data (`phase3_final_test_data`) of the [AAPM CT-MAR Grand Challenge](https://aapm.org/GrandChallenge/CT-MAR/).
The input folder must contain the fan-beam sinograms of the 29 cases under their original names
(900 channels × 1000 views, float32):

```
phase3_final_test_data/
├── 01.metalart.sino_900x1000.raw
├── 02.metalart.sino_900x1000.raw
│   ...
└── 29.metalart.sino_900x1000.raw
```

```bash
python run_inference.py --input /path/to/phase3_final_test_data --output submission --gpu 0
```

All 29 test cases are processed by default; `--cases 1 2 3` processes only test cases 01–03. The results `submission/XX.raw` (512×512, HU) can be scored with the challenge scoring script;
the 29 cases give a final score of 0.89. Our results are provided in `data/submission/`.
