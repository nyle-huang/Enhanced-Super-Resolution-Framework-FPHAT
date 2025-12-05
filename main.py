import argparse
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import csv

from utils import *
from dataprocessor import *
from model import *


SCALE = 4            # x4 super-resolution
PATCH_SIZE = 96      # HR patch size


def current_time():

    orig_time = time.localtime()
    curr_time = time.strftime("%Y-%m-%d-%H:%M", orig_time)

    return curr_time

def srcnn_train():

    # Model, loss, optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                     patience=10, threshold=1e-4)
    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoints, "srcnn_checkpoint.pth.tar")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_psnr_y": [],
        "val_ssim_y": [],
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.model}_history.csv")

    best_psnr = 0.0

    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"** Resuming from checkpoint {checkpoint_path} **")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if checkpoint.get("scheduler_state") is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_psnr = checkpoint.get("best_psnr", 0.0)
            print(f"** Loaded previous best PSNR = {best_psnr:.3f} **")
        else:
            print("** Tried to resume but no checkpoint found. Starting from scratch. **")

    # Datasets
    train_dataset = SRCNNDataset(
        lr_dir=args.train_lr,
        hr_dir=args.train_hr,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = SRCNNDataset(
        lr_dir=args.val_lr,
        hr_dir=args.val_hr,
        scale=SCALE,
        patch_size=None,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):

        model.train()
        running_loss = 0.0

        for i, (lr, hr) in enumerate(train_loader):

            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type, enabled=True):
                    sr = model(lr)
                    loss = criterion(sr, hr)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = model(lr)
                loss = criterion(sr, hr)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        # Validation (NTIRE restoration track: Y-PSNR & Y-SSIM with 4-pixel shave)
        model.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        count = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)

                sr = model(lr)
                sr = torch.clamp(sr, 0.0, 1.0)

                for b in range(sr.size(0)):
                    psnr_y, ssim_y = eval_restoration_metrics(sr[b], hr[b], shave=4)
                    psnr_total += psnr_y
                    ssim_total += ssim_y
                    count += 1

        val_psnr = psnr_total / max(count, 1)
        val_ssim = ssim_total / max(count, 1)
        print(f"[Epoch {epoch}] Val PSNR(Y): {val_psnr:.3f} dB, SSIM(Y): {val_ssim:.4f}")

        # ---- Logging history ----
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_psnr_y"].append(val_psnr)
        history["val_ssim_y"].append(val_ssim)

        # Save to CSV every epoch
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(history.keys())
            writer.writerow(header)
            for i in range(len(history["epoch"])):
                row = [history[k][i] for k in header]
                writer.writerow(row)

        scheduler.step(val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_psnr": best_psnr,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  >> SRCNN: New best model saved with PSNR={best_psnr:.3f} dB")

    print(f"{args.model}: Training finished.")
    print(f"Best Val PSNR: {best_psnr:.3f} dB")


def srcnn_test():

    os.makedirs(args.outputs, exist_ok=True)

    model = SRCNN().to(device)
    checkpoint_path = os.path.join(args.checkpoints, "srcnn_checkpoint.pth.tar")

    if not os.path.exists(checkpoint_path):
        print(f"** No checkpoint found at {checkpoint_path} **")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loading SRCNN checkpoint from {checkpoint_path}")

    lr_paths = sorted(
        glob(os.path.join(args.test_lr, "*.png")) +
        glob(os.path.join(args.test_lr, "*.jpg")) +
        glob(os.path.join(args.test_lr, "*.jpeg"))
    )

    if len(lr_paths) == 0:
        print(f"** No LR images found at {args.test_lr} **")
        return

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    curr_time = current_time()
    out_dir = os.path.join(args.outputs, curr_time, f"srcnn_{SCALE}")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for lr_path in lr_paths:
            lr = Image.open(lr_path).convert('RGB')
            w, h = lr.size
            lr_up = lr.resize((w * SCALE, h * SCALE), resample=Image.Resampling.BICUBIC)
            lr_up = to_tensor(lr_up).unsqueeze(0).to(device)
            sr = model(lr_up)  # [1, 3, H, W]
            sr = torch.clamp(sr, 0.0, 1.0).squeeze(0).cpu()  # [3, H, W]
            img_sr = to_pil(sr)
            basename = os.path.basename(lr_path)
            name, _ = os.path.splitext(basename)
            out_path = os.path.join(out_dir, f"{name}_x{SCALE}.png")
            img_sr.save(out_path)

    print(f"SRCNN test finished.")


def esrgan_train():

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)

    # -------------------------
    # 1. Build datasets / loaders
    # -------------------------
    train_dataset = ESRGANDataset(
        lr_dir=args.train_lr,
        hr_dir=args.train_hr,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = ESRGANDataset(
        lr_dir=args.val_lr,
        hr_dir=args.val_hr,
        scale=SCALE,
        patch_size=None,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[ESRGAN] Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # -------------------------
    # 2. Models
    # -------------------------
    generator = ESRGANGenerator(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=SCALE).to(device)
    discriminator = ESRGANDiscriminator(in_nc=3, base_nf=64).to(device)
    vgg_feat = VGGFeatureExtractor(layer_index=35).to(device)
    vgg_feat.eval()

    # -------------------------
    # 3. Optimizers, schedulers, losses
    # -------------------------
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='max', factor=0.5,
                                                       patience=10, threshold=1e-4)
    d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='max', factor=0.5,
                                                       patience=10, threshold=1e-4)

    pixel_criterion = nn.L1Loss().to(device)
    perceptual_criterion = nn.L1Loss().to(device)
    gan_criterion = nn.BCEWithLogitsLoss().to(device)

    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    # Loss weights (tune if needed)
    pixel_weight = 1.0
    perceptual_weight = 1e-4
    gan_weight = 5e-3

    history = {
        "epoch": [],
        "train_g_loss": [],
        "train_d_loss": [],
        "val_ntire_score": [],
        "val_psnr_y": [],
    }

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"{args.model}_history.csv")

    # -------------------------
    # 4. Checkpoint path & resume
    # -------------------------
    # Treat args.checkpoint as a directory.
    ckpt_path = os.path.join(args.checkpoints, "esrgan_checkpoint.pth.tar")

    # Initialize NTIRE perceptual metrics
    init_ntire_metrics(device)

    # Track best NTIRE composite score (higher is better) and its PSNR(Y)
    best_ntire_score = -1e9
    best_psnr_y = 0.0

    if args.resume and os.path.exists(ckpt_path):
        print(f"** [ESRGAN] Resuming from {ckpt_path} **")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state"])

        if checkpoint.get("g_scheduler_state") is not None:
            g_scheduler.load_state_dict(checkpoint["g_scheduler_state"])
        if checkpoint.get("d_scheduler_state") is not None:
            d_scheduler.load_state_dict(checkpoint["d_scheduler_state"])

        best_ntire_score = checkpoint.get("best_ntire_score", -1e9)
        best_psnr_y = checkpoint.get("best_psnr_y", 0.0)
        print(f"** [ESRGAN] Loaded previous best NTIRE score = "
              f"{best_ntire_score:.4f}, PSNR(Y) = {best_psnr_y:.3f} dB **")
    elif args.resume:
        print(f"** [ESRGAN] Tried to resume but no checkpoint found at {ckpt_path}. Starting from scratch. **")

    # -------------------------
    # 5. Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            # =====================
            # Train Discriminator
            # =====================
            d_optimizer.zero_grad()
            if use_amp:
                with autocast(device_type="cuda"):
                    with torch.no_grad():
                        sr_detached = generator(lr)

                    real_logits = discriminator(hr)
                    fake_logits = discriminator(sr_detached)

                    real_labels = torch.ones_like(real_logits)
                    fake_labels = torch.zeros_like(fake_logits)

                    d_loss_real = gan_criterion(real_logits, real_labels)
                    d_loss_fake = gan_criterion(fake_logits, fake_labels)
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)

                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    sr_detached = generator(lr)

                real_logits = discriminator(hr)
                fake_logits = discriminator(sr_detached)

                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(fake_logits)

                d_loss_real = gan_criterion(real_logits, real_labels)
                d_loss_fake = gan_criterion(fake_logits, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                d_loss.backward()
                d_optimizer.step()

            # =====================
            # Train Generator
            # =====================
            g_optimizer.zero_grad()
            if use_amp:
                with autocast(device_type="cuda"):
                    sr = generator(lr)

                    # Pixel loss
                    l_pixel = pixel_criterion(sr, hr) * pixel_weight

                    # Perceptual loss
                    with torch.no_grad():
                        hr_feat = vgg_feat(hr)
                    sr_feat = vgg_feat(sr)
                    l_percep = perceptual_criterion(sr_feat, hr_feat) * perceptual_weight

                    # GAN loss
                    fake_logits = discriminator(sr)
                    real_labels_for_g = torch.ones_like(fake_logits)
                    l_gan = gan_criterion(fake_logits, real_labels_for_g) * gan_weight

                    g_loss = l_pixel + l_percep + l_gan

                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                sr = generator(lr)

                l_pixel = pixel_criterion(sr, hr) * pixel_weight

                with torch.no_grad():
                    hr_feat = vgg_feat(hr)
                sr_feat = vgg_feat(sr)
                l_percep = perceptual_criterion(sr_feat, hr_feat) * perceptual_weight

                fake_logits = discriminator(sr)
                real_labels_for_g = torch.ones_like(fake_logits)
                l_gan = gan_criterion(fake_logits, real_labels_for_g) * gan_weight

                g_loss = l_pixel + l_percep + l_gan

                g_loss.backward()
                g_optimizer.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()

        avg_g_loss = running_g_loss / len(train_loader)
        avg_d_loss = running_d_loss / len(train_loader)
        print(f"[Epoch {epoch}] ESRGAN Train G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")

        # -------------------------
        # 6. Validation (NTIRE perceptual track)
        # -------------------------
        generator.eval()
        val_psnr_y_total = 0.0
        val_score_total = 0.0
        val_count = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)

                sr = generator(lr)
                sr = torch.clamp(sr, 0.0, 1.0)

                for b in range(sr.size(0)):
                    # Distortion metric (for reporting only)
                    psnr_y = calc_psnr_y_ntire(sr[b], hr[b], max_pixel=1.0, shave=4)
                    val_psnr_y_total += psnr_y

                    # Full NTIRE perceptual metrics
                    lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v = compute_ntire_metrics_for_pair(
                        sr[b], hr[b], device=device
                    )
                    score = compute_ntire_perceptual_score(
                        lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v
                    )

                    val_score_total += score
                    val_count += 1

        if val_count > 0:
            avg_psnr_y = val_psnr_y_total / val_count
            avg_score = val_score_total / val_count
        else:
            avg_psnr_y = 0.0
            avg_score = 0.0

        print(f"[Epoch {epoch}] ESRGAN Val: NTIRE score={avg_score:.4f}, "
              f"PSNR(Y)={avg_psnr_y:.3f} dB")

        # ---- Logging history ----
        history["epoch"].append(epoch)
        history["train_g_loss"].append(avg_g_loss)
        history["train_d_loss"].append(avg_d_loss)
        history["val_ntire_score"].append(avg_score)
        history["val_psnr_y"].append(avg_psnr_y)

        # Save to CSV every epoch
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(history.keys())
            writer.writerow(header)
            for i in range(len(history["epoch"])):
                row = [history[k][i] for k in header]
                writer.writerow(row)

        # Step schedulers based on NTIRE perceptual score (mode='max')
        g_scheduler.step(avg_score)
        d_scheduler.step(avg_score)

        # -------------------------
        # 7. Save best model (by NTIRE composite score)
        # -------------------------
        if avg_score > best_ntire_score:
            best_ntire_score = avg_score
            best_psnr_y = avg_psnr_y

            checkpoint = {
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "g_optimizer_state": g_optimizer.state_dict(),
                "d_optimizer_state": d_optimizer.state_dict(),
                "g_scheduler_state": g_scheduler.state_dict(),
                "d_scheduler_state": d_scheduler.state_dict(),
                "best_ntire_score": best_ntire_score,
                "best_psnr_y": best_psnr_y,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"  >> [ESRGAN] New best model saved "
                  f"(NTIRE score={best_ntire_score:.4f}, PSNR(Y)={best_psnr_y:.3f} dB)")


    print(f"[ESRGAN] Training finished. Best NTIRE score: {best_ntire_score:.4f}, "
          f"corresponding PSNR(Y): {best_psnr_y:.3f} dB")


def esrgan_test():

    os.makedirs(args.outputs, exist_ok=True)

    # -------------------------
    # 1. Build generator and load checkpoint
    # -------------------------
    generator = ESRGANGenerator(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=SCALE).to(device)

    ckpt_path = os.path.join(args.checkpoints, "esrgan_checkpoint.pth.tar")
    if not os.path.exists(ckpt_path):
        print(f"** [ESRGAN Test] No checkpoint found at {ckpt_path} **")
        print("Run esrgan_train() (training) first or check your --checkpoint path.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    print(f"[ESRGAN Test] Loaded checkpoint from {ckpt_path}")

    # -------------------------
    # 2. Collect LR test images
    # -------------------------
    lr_paths = sorted(
        glob(os.path.join(args.test_lr, "*.png")) +
        glob(os.path.join(args.test_lr, "*.jpg")) +
        glob(os.path.join(args.test_lr, "*.jpeg"))
    )

    if len(lr_paths) == 0:
        print(f"[ESRGAN Test] No images found in {args.test_lr}")
        return

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    scale = SCALE  # or use args.scale if you prefer
    subdir = f"esrgan_x{scale}"
    curr_time = current_time()
    out_dir = os.path.join(args.outputs, curr_time, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 3. Run inference
    # -------------------------
    with torch.no_grad():
        for lr_path in lr_paths:
            img_lr = Image.open(lr_path).convert("RGB")
            # w, h = img_lr.size  # LR size

            # Original ESRGAN: feed LR directly; generator upsamples internally
            inp = to_tensor(img_lr).unsqueeze(0).to(device)  # [1, 3, H, W]

            sr = generator(inp)          # [1, 3, H*scale, W*scale]
            sr = torch.clamp(sr, 0.0, 1.0).squeeze(0).cpu()  # [3, H*scale, W*scale]

            img_sr = to_pil(sr)

            base = os.path.basename(lr_path)
            name, _ = os.path.splitext(base)
            out_path = os.path.join(out_dir, f"{name}_esrgan_x{scale}.png")
            img_sr.save(out_path)

            print(f"[ESRGAN Test] Saved {out_path}")


def hat_train():

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)

    train_dataset = SRDataset(
        lr_dir=args.train_lr,
        hr_dir=args.train_hr,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        is_train=True
    )

    val_dataset = SRDataset(
        lr_dir=args.val_lr,
        hr_dir=args.val_hr,
        scale=SCALE,
        patch_size=None,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[HAT] Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # Model (you can increase num_groups/depth_per_group if your GPU allows)
    model = HATSRNet(
        scale=SCALE,
        num_in_ch=3,
        num_out_ch=3,
        dim=96,
        num_groups=4,
        depth_per_group=6,
        window_size=8,
        num_heads=6,
        mlp_ratio=2.0,
        drop_path=0.0
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, threshold=1e-4
    )

    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_psnr_y": [],
        "val_ssim_y": [],
        "val_ntire": [],
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.model}_history.csv")

    checkpoint_path = os.path.join(args.checkpoints, "hat_checkpoint.pth.tar")
    best_psnr = 0.0

    # Resume if requested
    if args.resume and os.path.exists(checkpoint_path):
        print(f"** [HAT] Resuming from checkpoint {checkpoint_path} **")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"** [HAT] Loaded previous best PSNR(Y) = {best_psnr:.3f} dB **")

    init_ntire_metrics(device)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device.type, enabled=True):
                    sr = model(lr)
                    loss = criterion(sr, hr)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = model(lr)
                loss = criterion(sr, hr)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[HAT][Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        # Validation (NTIRE-style Y PSNR)
        # Validation (NTIRE restoration: Y-PSNR & Y-SSIM)
        model.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        ntire_total = 0.0  # NEW
        count = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model(lr)
                sr = torch.clamp(sr, 0.0, 1.0)

                for b in range(sr.size(0)):
                    # Distortion metrics (Y-PSNR & Y-SSIM with 4-pixel shave)
                    psnr_y, ssim_y = eval_restoration_metrics(sr[b], hr[b], shave=4)
                    psnr_total += psnr_y
                    ssim_total += ssim_y

                    # NTIRE perceptual metrics for this pair
                    lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v = compute_ntire_metrics_for_pair(
                        sr[b], hr[b], device=device
                    )
                    score = compute_ntire_perceptual_score(
                        lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v
                    )
                    ntire_total += score

                    count += 1

        val_psnr = psnr_total / max(count, 1)
        val_ssim = ssim_total / max(count, 1)
        val_ntire = ntire_total / max(count, 1)  # NEW

        print(
            f"[HAT][Epoch {epoch}] Val PSNR(Y): {val_psnr:.3f} dB, "
            f"SSIM(Y): {val_ssim:.4f}, NTIRE: {val_ntire:.4f}"
        )

        # ---- Logging history ----
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_psnr_y"].append(val_psnr)
        history["val_ssim_y"].append(val_ssim)
        history["val_ntire"].append(val_ntire)

        # Save to CSV every epoch
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(history.keys())
            writer.writerow(header)
            for i in range(len(history["epoch"])):
                row = [history[k][i] for k in header]
                writer.writerow(row)

        scheduler.step(val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_psnr": best_psnr,
            }
            torch.save(ckpt, checkpoint_path)
            print(f"  >> [HAT] New best model saved with PSNR(Y)={best_psnr:.3f} dB")

    print(f"[HAT] Training finished. Best Val PSNR(Y): {best_psnr:.3f} dB")


def hat_test():

    model = HATSRNet(
        scale=SCALE,
        num_in_ch=3,
        num_out_ch=3,
        dim=96,
        num_groups=4,
        depth_per_group=6,
        window_size=8,
        num_heads=6,
        mlp_ratio=2.0,
        drop_path=0.0
    ).to(device)

    checkpoint_path = os.path.join(args.checkpoints, "hat_checkpoint.pth.tar")
    if not os.path.exists(checkpoint_path):
        print(f"** [HAT Test] No checkpoint found at {checkpoint_path} **")
        print("Run hat_train() first.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[HAT Test] Loaded checkpoint from {checkpoint_path}")

    lr_paths = sorted(
        glob(os.path.join(args.test_lr, "*.png")) +
        glob(os.path.join(args.test_lr, "*.jpg")) +
        glob(os.path.join(args.test_lr, "*.jpeg"))
    )
    if len(lr_paths) == 0:
        print(f"[HAT Test] No LR images found in {args.test_lr}")
        return

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    curr_time = current_time()
    out_dir = os.path.join(args.outputs, curr_time, f"hat_x{SCALE}")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for lr_path in lr_paths:
            img_lr = Image.open(lr_path).convert("RGB")
            lr = to_tensor(img_lr).unsqueeze(0).to(device)  # [1,3,H,W]

            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0).squeeze(0).cpu()
            img_sr = to_pil(sr)

            base = os.path.basename(lr_path)
            name, _ = os.path.splitext(base)
            out_path = os.path.join(out_dir, f"{name}_hat_x{SCALE}.png")
            img_sr.save(out_path)
            print(f"[HAT Test] Saved {out_path}")

    print(f"[HAT Test] Finished. Results in {out_dir}")


def fp_hat_train():

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)

    lambda_rgb_stage2 = 0.3
    lambda_y_stage2 = 0.3
    lambda_grad_stage2 = 0.08
    lambda_perc_stage2 = 0.04

    train_dataset = SRDataset(
        lr_dir=args.train_lr,
        hr_dir=args.train_hr,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        is_train=True
    )

    val_dataset = SRDataset(
        lr_dir=args.val_lr,
        hr_dir=args.val_hr,
        scale=SCALE,
        patch_size=None,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[FPHAT] Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # Model (you can increase num_groups/depth_per_group if your GPU allows)
    model = HATSRNet(
        scale=SCALE,
        num_in_ch=3,
        num_out_ch=3,
        dim=96,
        num_groups=4,
        depth_per_group=6,
        window_size=8,
        num_heads=6,
        mlp_ratio=2.0,
        drop_path=0.0
    ).to(device)

    pixel_criterion = nn.L1Loss()
    vgg_feat = VGGFeatureExtractor(layer_index=35).to(device)
    vgg_feat.eval()
    for p in vgg_feat.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, threshold=1e-4
    )

    init_ntire_metrics(device)

    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_psnr_y": [],
        "val_ssim_y": [],
        "val_ntire": [],
        "val_combined": [],
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.model}_history.csv")

    checkpoint_path = os.path.join(args.checkpoints, "fphat_checkpoint.pth.tar")
    best_psnr = 0.0
    best_ntire = 0.0
    best_combined = -1e9

    # Resume if requested
    if args.resume and os.path.exists(checkpoint_path):
        print(f"** [FPHAT] Resuming from checkpoint {checkpoint_path} **")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_psnr = ckpt.get("best_psnr_y", 0.0)
        best_ntire = ckpt.get("best_ntire", 0.0)
        best_combined = ckpt.get("best_combined", 0.0)
        print(f"** [FPHAT] Loaded previous best Combined = {best_combined:.3f} **")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        lambda_rgb = lambda_rgb_stage2
        lambda_y = lambda_y_stage2
        lambda_grad = lambda_grad_stage2
        lambda_perc = lambda_perc_stage2

        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device.type, enabled=True):
                    sr = model(lr)

                    # 1) RGB L1
                    loss_rgb = pixel_criterion(sr, hr)

                    # 2) Y-channel L1 (with 4-pixel shave)
                    #    sr, hr: [B,3,H,W] in [0,1]
                    sr_y = shave_border(rgb_to_y_channel(sr), shave=4)
                    hr_y = shave_border(rgb_to_y_channel(hr), shave=4)
                    loss_y = F.l1_loss(sr_y, hr_y)

                    # 3) Gradient loss
                    loss_grad = gradient_loss(sr, hr)

                    # 4) Perceptual loss via VGG features
                    with torch.no_grad():
                        feat_hr = vgg_feat(hr)
                    feat_sr = vgg_feat(sr)
                    loss_perc = F.l1_loss(feat_sr, feat_hr)

                    loss = (
                            lambda_rgb * loss_rgb +
                            lambda_y * loss_y +
                            lambda_grad * loss_grad +
                            lambda_perc * loss_perc
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = model(lr)

                # 1) RGB L1
                loss_rgb = pixel_criterion(sr, hr)

                # 2) Y-channel L1 (with 4-pixel shave)
                #    sr, hr: [B,3,H,W] in [0,1]
                sr_y = shave_border(rgb_to_y_channel(sr), shave=4)
                hr_y = shave_border(rgb_to_y_channel(hr), shave=4)
                loss_y = F.l1_loss(sr_y, hr_y)

                # 3) Gradient loss
                loss_grad = gradient_loss(sr, hr)

                # 4) Perceptual loss via VGG features
                with torch.no_grad():
                    feat_hr = vgg_feat(hr)
                feat_sr = vgg_feat(sr)
                loss_perc = F.l1_loss(feat_sr, feat_hr)

                loss = (
                        lambda_rgb * loss_rgb +
                        lambda_y * loss_y +
                        lambda_grad * loss_grad +
                        lambda_perc * loss_perc
                )

                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        print(f"[Debug] rgb={loss_rgb.item():.4f}, "
              f"y={loss_y.item():.4f}, "
              f"grad={loss_grad.item():.4f}, "
              f"vgg={loss_perc.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"[FPHAT][Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        # Validation (NTIRE-style Y PSNR)
        # Validation (NTIRE restoration: Y-PSNR & Y-SSIM)
        model.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        ntire_score_total = 0.0
        count = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model(lr)
                sr = torch.clamp(sr, 0.0, 1.0)

                for b in range(sr.size(0)):
                    psnr_y, ssim_y = eval_restoration_metrics(sr[b], hr[b], shave=4)
                    psnr_total += psnr_y
                    ssim_total += ssim_y

                    lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v = compute_ntire_metrics_for_pair(
                        sr[b], hr[b], device=device
                    )
                    score = compute_ntire_perceptual_score(
                        lpips_v, dists_v, clip_v, mani_v, musiq_v, niqe_v
                    )
                    ntire_score_total += score

                    count += 1

        val_psnr = psnr_total / max(count, 1)
        val_ssim = ssim_total / max(count, 1)
        val_ntire_score = ntire_score_total / max(count, 1)
        print(f"[FPHAT][Epoch {epoch}] Val PSNR(Y): {val_psnr:.3f} dB, NTIRE(Y): {val_ntire_score:.4f}")

        w_psnr = 0.5
        w_ntire = 0.5
        psnr_norm = val_psnr / 40.0  # assume 40 dB is "very good"
        ntire_norm = val_ntire_score / 6.0  # assume 6 is "very good"

        combined_score = w_psnr * psnr_norm + w_ntire * ntire_norm

        # ---- Logging history ----
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_psnr_y"].append(val_psnr)
        history["val_ssim_y"].append(val_ssim)
        history["val_ntire"].append(val_ntire_score)
        history["val_combined"].append(combined_score)

        # Save to CSV every epoch
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(history.keys())
            writer.writerow(header)
            for i in range(len(history["epoch"])):
                row = [history[k][i] for k in header]
                writer.writerow(row)

        scheduler.step(combined_score)

        if combined_score > best_combined:
            best_combined = combined_score
            best_psnr_y_at_best = val_psnr
            best_ntire_at_best = val_ntire_score

            checkpoint_path = os.path.join(args.checkpoints, "fphat_checkpoint.pth.tar")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "best_combined": best_combined,
                    "best_psnr_y": best_psnr_y_at_best,
                    "best_ntire": best_ntire_at_best,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"** Saved new best FP-HAT checkpoint with Combined={best_combined:.3f} **")

    print(f"[FPHAT] Training finished. Best Val Combined: {best_combined:.3f}")


def fp_hat_test():

    model = HATSRNet(
        scale=SCALE,
        num_in_ch=3,
        num_out_ch=3,
        dim=96,
        num_groups=4,
        depth_per_group=6,
        window_size=8,
        num_heads=6,
        mlp_ratio=2.0,
        drop_path=0.0
    ).to(device)

    checkpoint_path = os.path.join(args.checkpoints, "fphat_checkpoint.pth.tar")
    if not os.path.exists(checkpoint_path):
        print(f"** [FPHAT Test] No checkpoint found at {checkpoint_path} **")
        print("Run fp_hat_train() first.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[FPHAT Test] Loaded checkpoint from {checkpoint_path}")

    lr_paths = sorted(
        glob(os.path.join(args.test_lr, "*.png")) +
        glob(os.path.join(args.test_lr, "*.jpg")) +
        glob(os.path.join(args.test_lr, "*.jpeg"))
    )
    if len(lr_paths) == 0:
        print(f"[FPHAT Test] No LR images found in {args.test_lr}")
        return

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    curr_time = current_time()
    out_dir = os.path.join(args.outputs, curr_time, f"fphat_x{SCALE}")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for lr_path in lr_paths:
            img_lr = Image.open(lr_path).convert("RGB")
            lr = to_tensor(img_lr).unsqueeze(0).to(device)  # [1,3,H,W]

            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0).squeeze(0).cpu()
            img_sr = to_pil(sr)

            base = os.path.basename(lr_path)
            name, _ = os.path.splitext(base)
            out_path = os.path.join(out_dir, f"{name}_fphat_x{SCALE}.png")
            img_sr.save(out_path)
            print(f"[FPHAT Test] Saved {out_path}")

    print(f"[FPHAT Test] Finished. Results in {out_dir}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing weights.")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (no training).")
    parser.add_argument("--model", type=str, default=None,
                        help="Specify the model to use.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=2e-4,
                        help="Weight decay to reduce overfitting.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Add parallel dataloaders.")
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="Path to save/load a checkpoint.")
    parser.add_argument("--train_lr", type=str, default="data/Flickr2K/Flickr2K_LR_bicubic/X4",
                        help="Path to find LR images for training.")
    parser.add_argument("--train_hr", type=str, default="data/Flickr2K/Flickr2K_HR",
                        help="Path to find HR images for training.")
    parser.add_argument("--val_lr", type=str, default="data/validation/LR/DIV2K_valid_LR_bicubic/X4",
                        help="Path to find LR images for validation.")
    parser.add_argument("--val_hr", type=str, default="data/validation/HR/DIV2K_valid_HR",
                        help="Path to find HR images for validation.")
    parser.add_argument("--test_lr", type=str, default="test",
                        help="Path to find LR images for testing.")
    parser.add_argument("--outputs", type=str, default="outputs",
                        help="Path to save output images.")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        print(">> Using Apple Silicon GPU (MPS)")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(">> Using CUDA GPU")
        device = torch.device("cuda")
    else:
        print(">> Using CPU")
        device = torch.device("cpu")

    if args.model is None:
        print("Please specify a model to start: srcnn | esrgan | hat | fphat")

    elif args.model.lower() == "srcnn":
        if not args.test:
            srcnn_train()
        else:
            srcnn_test()

    elif args.model.lower() == "esrgan":
        if not args.test:
            esrgan_train()
        else:
            esrgan_test()

    elif args.model.lower() == "hat":
        if not args.test:
            hat_train()
        else:
            hat_test()

    elif args.model.lower() == "fphat":
        if not args.test:
            fp_hat_train()
        else:
            fp_hat_test()

    else:
        print("Please enter a valid model option: srcnn | esrgan | hat | fphat")

