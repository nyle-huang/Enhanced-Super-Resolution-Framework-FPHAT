import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return {}

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        return {}

    header = rows[0]
    data = {name: [] for name in header}

    for row in rows[1:]:
        for name, value in zip(header, row):
            # try to cast to float; otherwise keep the raw string
            try:
                num = float(value)
                data[name].append(num)
            except ValueError:
                data[name].append(value)

    return data


def plot_curve(epochs, values, title, y_label, out_path):
    """
    Simple wrapper to draw a curve and save it.
    """
    plt.figure()
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # SRCNN
    srcnn_csv = logs_dir / "srcnn_history.csv"
    if srcnn_csv.exists():
        hist = load_history(srcnn_csv)
        epochs = hist.get("epoch", [])
        if epochs:
            plot_curve(
                epochs, hist["train_loss"],
                "SRCNN Train Loss", "Train Loss",
                logs_dir / "srcnn_train_loss.png",
            )
            plot_curve(
                epochs, hist["val_psnr_y"],
                "SRCNN Val PSNR(Y)", "PSNR(Y) [dB]",
                logs_dir / "srcnn_val_psnr_y.png",
            )
            plot_curve(
                epochs, hist["val_ssim_y"],
                "SRCNN Val SSIM(Y)", "SSIM(Y)",
                logs_dir / "srcnn_val_ssim_y.png",
            )

    # ESRGAN
    esrgan_csv = logs_dir / "esrgan_history.csv"
    if esrgan_csv.exists():
        hist = load_history(esrgan_csv)
        epochs = hist.get("epoch", [])
        if epochs:
            plot_curve(
                epochs, hist["train_g_loss"],
                "ESRGAN Train G Loss", "G Loss",
                logs_dir / "esrgan_train_g_loss.png",
            )
            plot_curve(
                epochs, hist["train_d_loss"],
                "ESRGAN Train D Loss", "D Loss",
                logs_dir / "esrgan_train_d_loss.png",
            )
            # Uses 'val_ntire_score' from esrgan_history.csv
            plot_curve(
                epochs, hist["val_ntire_score"],
                "ESRGAN Val NTIRE Score", "NTIRE Score",
                logs_dir / "esrgan_val_ntire_score.png",
            )
            plot_curve(
                epochs, hist["val_psnr_y"],
                "ESRGAN Val PSNR(Y)", "PSNR(Y) [dB]",
                logs_dir / "esrgan_val_psnr_y.png",
            )

    # HAT
    hat_csv = logs_dir / "hat_history.csv"
    if hat_csv.exists():
        hist = load_history(hat_csv)
        epochs = hist.get("epoch", [])
        if epochs:
            plot_curve(
                epochs, hist["train_loss"],
                "HAT Train Loss", "Train Loss",
                logs_dir / "hat_train_loss.png",
            )
            plot_curve(
                epochs, hist["val_psnr_y"],
                "HAT Val PSNR(Y)", "PSNR(Y) [dB]",
                logs_dir / "hat_val_psnr_y.png",
            )
            plot_curve(
                epochs, hist["val_ssim_y"],
                "HAT Val SSIM(Y)", "SSIM(Y)",
                logs_dir / "hat_val_ssim_y.png",
            )
            # NEW: HAT NTIRE score curve (requires val_ntire in hat_history.csv)
            if "val_ntire" in hist:
                plot_curve(
                    epochs, hist["val_ntire"],
                    "HAT Val NTIRE Score", "NTIRE Score",
                    logs_dir / "hat_val_ntire_score.png",
                )

    # FPHAT
    fphat_csv = logs_dir / "fphat_history.csv"
    if fphat_csv.exists():
        hist = load_history(fphat_csv)
        epochs = hist.get("epoch", [])
        if epochs:
            plot_curve(
                epochs, hist["train_loss"],
                "FPHAT Train Loss", "Train Loss",
                logs_dir / "fphat_train_loss.png",
            )
            plot_curve(
                epochs, hist["val_psnr_y"],
                "FPHAT Val PSNR(Y)", "PSNR(Y) [dB]",
                logs_dir / "fphat_val_psnr_y.png",
            )
            plot_curve(
                epochs, hist["val_ntire"],
                "FPHAT Val NTIRE Score", "NTIRE Score",
                logs_dir / "fphat_val_ntire_score.png",
            )

    print("Saved plots under:", logs_dir)


if __name__ == "__main__":
    main()
