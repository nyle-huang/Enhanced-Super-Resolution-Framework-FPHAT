# **Enhanced Super-Resolution Framework (FP-HAT, HAT, ESRGAN, SRCNN)**

This repository provides a **complete, research-oriented super-resolution (SR) framework** supporting both distortion-focused and perceptual-focused SR models.  
 It implements classic, GAN-based, transformer-based, and **my optimized model**, **FP-HAT (Frequency–Perceptual Enhanced HAT)**.

FP-HAT is the primary contribution of this project, similar to those evaluated in **NTIRE** and **CVPR** competitions.

---

## **🔥 Key Contribution: FP-HAT (Frequency–Perceptual Enhanced HAT)**

The original HAT (Hybrid Attention Transformer) is a transformer-based SOTA model optimized mainly for **PSNR** using L1 loss.  
 My optimized model, **FP-HAT**, enhances both:

### **1\. Frequency Fidelity**

* Adds **gradient (edge) loss** to preserve high-frequency structure

* Reduces smoothing typical in transformer SR models

* Improves sharpness and luminance transitions

### **2\. Perceptual Quality**

* Adds **VGG perceptual loss**, aligned with NTIRE perceptual metrics

* Adds **Y-channel luminance loss**, reflecting human visual sensitivity

* Uses a **combined score (NTIRE \+ PSNR)** for checkpointing

FP-HAT produces **higher perceptual reconstruction quality** compared with standard HAT while maintaining strong PSNR performance.

---

## **📌 Implemented Models**

### **FP-HAT (Optimized HAT Model — Main Showcase)**

My enhanced version of HAT with perceptual and frequency-domain improvements.  
 Designed to outperform standard HAT in perceptual metrics (LPIPS, DISTS, MUSIQ, MANIQA).

### **HAT (Hybrid Attention Transformer)**

High-performing transformer SR model for restoration-track evaluation (PSNR/SSIM).

### **ESRGAN**

GAN-based SR model emphasizing realistic texture generation and perceptual sharpness.

### **SRCNN**

Classic CNN baseline for understanding fundamental SR strategies.

---

## **📊 Evaluation Metrics**

### **Restoration Track**

* PSNR (Y-channel, NTIRE protocol)

* SSIM (Y-channel)

### **Perceptual Track (NTIRE 2025 style)**

Using **pyiqa**, the framework computes:

* LPIPS

* DISTS

* CLIP-IQA

* MANIQA

* MUSIQ

* NIQE

* Composite perceptual score (normalized)

FP-HAT is trained and checkpointed using this **full perceptual evaluation system**.

---

## **🧩 Project Structure**

`.`  
`├── dataprocessor.py   # Dataset loading, preprocessing, augmentation`  
`├── model.py           # SRCNN, ESRGAN, HAT, FP-HAT models`  
`├── utils.py           # PSNR/SSIM, NTIRE metrics, losses (VGG, gradient)`  
`├── main.py            # Unified training/testing scripts`  
`├── plotting.py        # Training curve visualization tools`  
`├── logs/              # Auto-generated logs (not included)`  
`├── checkpoints/       # Saved checkpoints (not included)`  
`└── outputs/           # Model outputs (optional)`

---

## **🚀 Training Instructions**

### **Train FP-HAT**

`python main.py --model fphat --epochs 800 --batch_size 16 --lr 2e-4`

### **Train HAT**

`python main.py --model hat`

### **Train ESRGAN**

`python main.py --model esrgan`

### **Train SRCNN**

`python main.py --model srcnn`

---

## **🧪 Testing Instructions**

`python main.py --model fphat --test --test_lr path/to/LR/images/`

Outputs will be saved in `outputs/`.

---

## **📂 Datasets**

### **Flickr2K (Primary Training Dataset)**

* 2000+ HR/LR pairs

* Ideal for improving generalization and texture learning

### **DIV2K (Validation Dataset)**

* Standard benchmark for SR

* Used for PSNR(Y)/SSIM(Y)/NTIRE-score validation

Dataset loading and augmentation logic are implemented in `dataprocessor.py`.

---

## **📈 Logging & Visualization**

The framework automatically logs:

* PSNR(Y)

* SSIM(Y)

* NTIRE composite perceptual score

* Loss components

* Best model checkpoints

`plotting.py` generates training curves for comparison across models.

