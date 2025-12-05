import math
import pyiqa

import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Basic PSNR
# ------------------------------------------------------------

def calc_psnr(pred: torch.Tensor,
              target: torch.Tensor,
              max_pixel: float = 1.0) -> float:
    """
    Simple PSNR on RGB in [0, 1].
    """
    # pred, target: same shape, [C,H,W] or [B,C,H,W]
    mse = torch.mean((pred - target) ** 2)
    if mse <= 0:
        return 100.0
    return 20.0 * math.log10(max_pixel / math.sqrt(mse.item()))


# ------------------------------------------------------------
# Helpers: RGB -> Y and border cropping
# ------------------------------------------------------------

def rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image(s) in [0,1] to Y (luma) using BT.601-ish weights.

    img: [3,H,W] or [B,3,H,W]
    returns: [1,H,W] or [B,1,H,W]
    """
    if img.dim() == 3:
        # [3,H,W]
        assert img.size(0) == 3, f"Expected [3,H,W], got {img.shape}"
        r = img[0]
        g = img[1]
        b = img[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y.unsqueeze(0)

    if img.dim() == 4:
        # [B,3,H,W]
        assert img.size(1) == 3, f"Expected [B,3,H,W], got {img.shape}"
        r = img[:, 0, :, :]
        g = img[:, 1, :, :]
        b = img[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y.unsqueeze(1)

    raise ValueError(f"rgb_to_y_channel expects 3D or 4D tensor, got {img.dim()}D")


def shave_border(img: torch.Tensor, shave: int = 4) -> torch.Tensor:
    """
    Crop a border from all sides: [..., H, W] -> [..., H-2*shave, W-2*shave].
    """
    if shave <= 0:
        return img
    return img[..., shave:-shave, shave:-shave]


# ------------------------------------------------------------
# NTIRE-style PSNR / SSIM on Y channel
# ------------------------------------------------------------

def calc_psnr_y_ntire(pred: torch.Tensor,
                      target: torch.Tensor,
                      max_pixel: float = 1.0,
                      shave: int = 4) -> float:
    """
    NTIRE-style PSNR on Y channel:
      - convert RGB -> Y
      - shave border
      - compute PSNR on Y
    """
    assert pred.shape == target.shape, \
        f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    y_pred = shave_border(rgb_to_y_channel(pred), shave)
    y_gt = shave_border(rgb_to_y_channel(target), shave)

    assert y_pred.shape == y_gt.shape, \
        f"Shape mismatch after shaving: pred {y_pred.shape}, gt {y_gt.shape}"

    mse = torch.mean((y_pred - y_gt) ** 2)
    if mse <= 0:
        return 100.0
    return 20.0 * math.log10(max_pixel / math.sqrt(mse.item()))


def _gaussian_window(window_size: int,
                     sigma: float,
                     channels: int,
                     device,
                     dtype):
    """
    2D normalized Gaussian window used by SSIM.
    """
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - window_size // 2

    # 1D Gaussian
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    # 2D Gaussian via outer product
    g2d = g[:, None] * g[None, :]
    g2d = g2d / g2d.sum()

    # [C,1,ws,ws]
    window = g2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def calc_ssim_y_ntire(pred: torch.Tensor,
                      target: torch.Tensor,
                      max_val: float = 1.0,
                      window_size: int = 11,
                      sigma: float = 1.5,
                      shave: int = 4) -> float:
    """
    NTIRE-style SSIM on Y channel:
      - convert RGB -> Y
      - shave border
      - compute single-channel SSIM
    """
    assert pred.shape == target.shape, \
        f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # Convert to Y and crop
    y_pred = shave_border(rgb_to_y_channel(pred), shave)
    y_gt = shave_border(rgb_to_y_channel(target), shave)

    # Expect [1,H,W] or [B,1,H,W]
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(0)
        y_gt = y_gt.unsqueeze(0)

    assert y_pred.dim() == 4 and y_pred.size(1) == 1, \
        f"calc_ssim_y_ntire expects [B,1,H,W], got {y_pred.shape}"

    B, C, H, W = y_pred.shape
    device = y_pred.device
    dtype = y_pred.dtype

    window = _gaussian_window(window_size, sigma, C, device, dtype)

    # Means
    mu1 = F.conv2d(y_pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(y_gt, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Variances and covariance
    sigma1_sq = F.conv2d(y_pred * y_pred, window,
                         padding=window_size // 2,
                         groups=C) - mu1_sq
    sigma2_sq = F.conv2d(y_gt * y_gt, window,
                         padding=window_size // 2,
                         groups=C) - mu2_sq
    sigma12 = F.conv2d(y_pred * y_gt, window,
                       padding=window_size // 2,
                       groups=C) - mu1_mu2

    # SSIM constants
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    ssim = ssim_map.mean()

    return float(ssim.item())


# ------------------------------------------------------------
# Perceptual track score (NTIRE-style)
# ------------------------------------------------------------

def compute_ntire_perceptual_score(lpips: float,
                                   dists: float,
                                   clip_iqa: float,
                                   maniqa: float,
                                   musiq: float,
                                   niqe: float) -> float:
    """
    Combine six metrics into a single NTIRE-style perceptual score.
    Higher is better.
    """
    # Lower is better for lpips and dists
    term_lpips = 1.0 - lpips
    term_dists = 1.0 - dists

    # MUSIQ is roughly [0, 100], normalize a bit
    term_musiq = musiq / 100.0

    # NIQE: smaller is better, "good" roughly <= 5
    term_niqe = (10.0 - niqe) / 10.0
    if term_niqe < 0.0:
        term_niqe = 0.0

    score = term_lpips + term_dists + clip_iqa + maniqa + term_musiq + term_niqe
    return float(score)


def eval_restoration_metrics(pred: torch.Tensor,
                             target: torch.Tensor,
                             shave: int = 4) -> (float, float):
    """
    Convenience wrapper:
      returns (PSNR_Y, SSIM_Y) with NTIRE-style border crop.
    """
    psnr_y = calc_psnr_y_ntire(pred, target, max_pixel=1.0, shave=shave)
    ssim_y = calc_ssim_y_ntire(pred, target, max_val=1.0, shave=shave)
    return psnr_y, ssim_y


# ------------------------------------------------------------
# Full NTIRE metrics via pyiqa (LPIPS, DISTS, CLIP-IQA, MANIQA,
# MUSIQ, NIQE)
# ------------------------------------------------------------

_lpips_metric = None
_dists_metric = None
_clip_iqa_metric = None
_maniqa_metric = None
_musiq_metric = None
_niqe_metric = None


def init_ntire_metrics(device: torch.device):
    """
    Create / cache all pyiqa metric objects on the given device.
    Call once before using compute_ntire_metrics_for_pair.
    """
    global _lpips_metric, _dists_metric, _clip_iqa_metric
    global _maniqa_metric, _musiq_metric, _niqe_metric

    if _lpips_metric is None:
        _lpips_metric = pyiqa.create_metric(
            "lpips", device=device, as_loss=False
        ).eval()

    if _dists_metric is None:
        _dists_metric = pyiqa.create_metric(
            "dists", device=device, as_loss=False
        ).eval()

    if _clip_iqa_metric is None:
        _clip_iqa_metric = pyiqa.create_metric(
            "clipiqa", device=device, as_loss=False
        ).eval()

    if _maniqa_metric is None:
        _maniqa_metric = pyiqa.create_metric(
            "maniqa", device=device, as_loss=False
        ).eval()

    if _musiq_metric is None:
        _musiq_metric = pyiqa.create_metric(
            "musiq", device=device, as_loss=False
        ).eval()

    if _niqe_metric is None:
        _niqe_metric = pyiqa.create_metric(
            "niqe", device=device, as_loss=False
        ).eval()


def compute_ntire_metrics_for_pair(sr: torch.Tensor,
                                   hr: torch.Tensor,
                                   device: torch.device = None):
    """
    Compute (lpips, dists, clip_iqa, maniqa, musiq, niqe)
    for a single SR–HR pair in [0,1].
    """
    # Move to device if provided
    if device is not None:
        sr = sr.to(device)
        hr = hr.to(device)

    if sr.dim() == 3:
        sr_b = sr.unsqueeze(0)
        hr_b = hr.unsqueeze(0)
    else:
        sr_b = sr
        hr_b = hr

    with torch.no_grad():
        lpips_val = float(_lpips_metric(sr_b, hr_b).item())
        dists_val = float(_dists_metric(sr_b, hr_b).item())
        clip_val = float(_clip_iqa_metric(sr_b, hr_b).item())
        mani_val = float(_maniqa_metric(sr_b, hr_b).item())
        musiq_val = float(_musiq_metric(sr_b).item())
        niqe_val = float(_niqe_metric(sr_b).item())

    return lpips_val, dists_val, clip_val, mani_val, musiq_val, niqe_val


# ------------------------------------------------------------
# Simple gradient loss (Sobel on Y)
# ------------------------------------------------------------

def gradient_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:

    if sr.shape != hr.shape:
        raise ValueError(f"gradient_loss: shape mismatch {sr.shape} vs {hr.shape}")

    if sr.dim() != 4 or sr.size(1) != 3:
        raise ValueError("gradient_loss expects [B,3,H,W]")

    # Convert to Y, keep batch dimension
    sr_y = rgb_to_y_channel(sr)   # [B,1,H,W]
    hr_y = rgb_to_y_channel(hr)   # [B,1,H,W]

    # Fixed Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=sr.dtype,
        device=sr.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0,   0,  0],
         [1,   2,  1]],
        dtype=sr.dtype,
        device=sr.device
    ).view(1, 1, 3, 3)

    sr_gx = F.conv2d(sr_y, sobel_x, padding=1)
    sr_gy = F.conv2d(sr_y, sobel_y, padding=1)
    hr_gx = F.conv2d(hr_y, sobel_x, padding=1)
    hr_gy = F.conv2d(hr_y, sobel_y, padding=1)

    loss = F.l1_loss(sr_gx, hr_gx) + F.l1_loss(sr_gy, hr_gy)
    return loss
