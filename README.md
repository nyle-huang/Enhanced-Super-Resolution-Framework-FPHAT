# Enhanced Super-Resolution Framework (FP-HAT)

This repository implements and compares multiple single-image super-resolution (SISR) models under a **unified PyTorch training and evaluation pipeline**, with a focus on both:

- **Restoration quality** (PSNR / SSIM, NTIRE protocol)
- **Perceptual quality** (NTIRE-style composite perceptual metrics)

The project was developed as a **CMPT742 Final Project** and includes both baseline models and an enhanced transformer-based approach (**FP-HAT**).

---

## Implemented Models

| Model   | Type | Optimization Focus |
|--------|------|--------------------|
| SRCNN  | CNN  | Pixel-wise restoration |
| ESRGAN | GAN  | Perceptual realism |
| HAT    | Transformer | High-fidelity restoration |
| FP-HAT | Transformer (enhanced) | Balanced restoration + perceptual quality |

---

## Main Contribution: FP-HAT

**FP-HAT (Frequency–Perceptual Enhanced HAT)** extends the standard HAT architecture with:

- **Gradient loss** (edge / high-frequency preservation)
- **Y-channel loss** (luminance-aware optimization)
- **VGG perceptual loss**
- **Differentiable perceptual proxies** (LPIPS, DISTS)
- **Two-stage training schedule**
- **Combined checkpoint selection criterion**
  (PSNR + NTIRE perceptual score)

This design improves perceptual quality while retaining strong restoration performance.

---

## Evaluation Metrics

### Restoration Track (NTIRE protocol)
- **PSNR (Y-channel, 4-pixel shave)**
- **SSIM (Y-channel, 4-pixel shave)**

### Perceptual Track (NTIRE-style)
Computed using `pyiqa`:
- LPIPS
- DISTS
- CLIP-IQA
- MANIQA
- MUSIQ
- NIQE

The composite perceptual score is computed as:

