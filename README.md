# Enhanced Super-Resolution Framework (FP-HAT)

This project investigates the trade-off between restoration-oriented and perceptual-oriented objectives in single-image super-resolution.
Within a unified PyTorch pipeline, three models are implemented as baselines, and an optimized model of the state-of-the-art HAT is proposed and implemented:
- **SRCNN** (Baseline 1: classical CNN)
- **ESRGAN** (Baseline 2: GAN-based perceptual SR)
- **HAT** (Baseline 3: Hybrid Attention Transformer for restoration SR)
- **FP-HAT** (Optimized HAT variant with perceptual losses and NTIRE-aware validation)

---

## 🔥 Results (Summary)

### Quantitative Results (Bicubic x4 Validation; ×4 SR)

|  Model |  PSNR(Y)  |  NTIRE Score  | Notes |
|--------|-----------|---------------|-------|
| SRCNN  | 27.12     | —             | Classical CNN baseline |
| ESRGAN | 26.21     | **3.87**      | Strong perceptual metric performance with GAN artifacts |
| HAT    | **27.94** | 3.11          | Best restoration performance |
| FP-HAT | 27.57     | **3.61**      | Strong restoration performance while achieving much higher perceptual quality |

**Key observations**
- **HAT** dominates PSNR/SSIM as a restoration-oriented transformer.
- **ESRGAN** achieves the highest NTIRE score but introduces hallucinated textures.
- **FP-HAT** significantly improves perceptual quality over HAT (+0.5 NTIRE) while preserving high PSNR.

---

### Training Curves

The following plots are automatically generated during training and stored under `logs/plots/`:

- Validation PSNR(Y)
- Validation SSIM(Y)
- Validation NTIRE composite score
- Training loss

These curves are directly used in the final report for convergence and trade-off analysis.

---

## Implemented Models

- **SRCNN** — classical CNN baseline (pixel-wise restoration)
- **ESRGAN** — GAN-based perceptual SR
- **HAT** — Hybrid Attention Transformer (state-of-the-art restoration SR)
- **FP-HAT** — optimized HAT with perceptual and frequency-aware losses

---

## Main Contribution: FP-HAT

FP-HAT (*Frequency–Perceptual HAT*) reuses the HAT architecture but **modifies the training objective** to move a restoration-oriented transformer toward the perceptual regime.

### Optimized Loss Computation in FP-HAT

FP-HAT optimizes a weighted combination of:

- **RGB L1 loss**  
  Standard pixel-wise reconstruction loss (as in original HAT).

- **Y-channel (luminance) L1 loss (Added)**  
  Enforces higher fidelity on luminance, aligned with human visual sensitivity.

- **Sobel-based gradient loss (Added)**  
  Penalizes gradient mismatches on the Y channel to preserve edges and high-frequency structures.

- **VGG perceptual loss (Added)**  
  Feature-space distance using a pretrained VGG-19 network.

- **LPIPS loss (Added)**  
  Penalizes learned perceptual discrepancies between SR and HR images.

- **DISTS loss (Added)**  
  Measures structural–textural perceptual similarity.

These terms jointly encourage sharper edges and perceptual realism **without introducing a discriminator**.  
Checkpoint selection is based on a **combined criterion** balancing PSNR and NTIRE-style perceptual score.

---

## Project Structure

```
.
├── main.py
├── model.py
├── dataprocessor.py
├── utils.py
├── plotting.py
├── requirements.txt
├── logs/
│   ├── srcnn_history.csv
│   ├── esrgan_history.csv
│   ├── hat_history.csv
│   ├── fphat_history.csv
│   └── plots/
│       ├── *_train_loss.png
│       ├── *_val_psnr_y.png
│       ├── *_val_ssim_y.png
│       └── *_val_ntire_score.png
├── checkpoints/
└── outputs/
```

---

## Installation

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate     # Windows (PowerShell)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

- PyTorch ≥ 2.2.0 is required
- CUDA, Apple Silicon (MPS), and CPU are supported

---

## Dataset Preparation

Aligned LR–HR image pairs for ×4 SR are required.

**Default paths (used by `main.py`)**

Training:
- LR: `data/Flickr2K/Flickr2K_LR_bicubic/X4`
- HR: `data/Flickr2K/Flickr2K_HR`

Validation:
- LR: `data/validation/LR/DIV2K_valid_LR_bicubic/X4`
- HR: `data/validation/HR/DIV2K_valid_HR`

---

## Training

```bash
python main.py --model srcnn
python main.py --model esrgan
python main.py --model hat
python main.py --model fphat
```

---

## Logs and Plots

During training, the framework records:

### CSV Logs
Stored in `logs/`:
- `srcnn_history.csv`
- `esrgan_history.csv`
- `hat_history.csv`
- `fphat_history.csv`

Each CSV logs epoch-level training loss, validation PSNR(Y), SSIM(Y), and NTIRE metrics.

### Plots
Generated via `plotting.py` and saved in `logs/plots/`, including:
- Training loss curves
- Validation PSNR(Y) curves
- Validation SSIM(Y) curves
- Validation NTIRE score curves

---

## Checkpoints

Model checkpoints are saved under `checkpoints/`.

Pretrained checkpoints corresponding to the reported results can be downloaded from:

**https://drive.google.com/drive/folders/1OqyTxw4eFIybHCNrYstFTQmOOzXVTuQ9?usp=sharing**

**Please note** that these checkpoints are achieved from training on the Flickr benchmark bicubic x4 dataset. To test the models with the checkpoints, you should load the models with the test images downsampled via the same algorithm. 
