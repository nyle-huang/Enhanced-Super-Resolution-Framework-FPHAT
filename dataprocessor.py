import os
import random
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def _list_images_in_dir(root_dir):

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        pattern = os.path.join(root_dir, ext)
        image_paths.extend(glob(pattern))

    image_paths.sort()
    return image_paths


class PFTSRDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, scale=4, patch_size=128, is_train=True):
        super().__init__()

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train

        self.lr_paths = _list_images_in_dir(lr_dir)
        self.hr_paths = _list_images_in_dir(hr_dir)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"Number of LR and HR images does not match: "
                f"{len(self.lr_paths)} vs {len(self.hr_paths)}"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size

        # training: random aligned crop
        if self.is_train and self.patch_size is not None:
            hr_patch = self.patch_size
            lr_patch = hr_patch // self.scale

            # pick HR crop region
            if hr_w >= hr_patch and hr_h >= hr_patch:
                x_hr = random.randint(0, hr_w - hr_patch)
                y_hr = random.randint(0, hr_h - hr_patch)
            else:
                # if HR is too small, just center crop
                x_hr = max(0, (hr_w - hr_patch) // 2)
                y_hr = max(0, (hr_h - hr_patch) // 2)

            # corresponding LR coords
            x_lr = x_hr // self.scale
            y_lr = y_hr // self.scale

            # make sure LR patch stays inside boundaries
            x_lr = min(x_lr, max(lr_w - lr_patch, 0))
            y_lr = min(y_lr, max(lr_h - lr_patch, 0))

            hr_box = (x_hr, y_hr, x_hr + hr_patch, y_hr + hr_patch)
            lr_box = (x_lr, y_lr, x_lr + lr_patch, y_lr + lr_patch)

            hr_img = hr_img.crop(hr_box)
            lr_img = lr_img.crop(lr_box)

        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        # ----- geometric augmentation (train only) -----
        if self.is_train:
            # random horizontal flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[2])  # width
                hr_tensor = torch.flip(hr_tensor, dims=[2])
            # random vertical flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[1])  # height
                hr_tensor = torch.flip(hr_tensor, dims=[1])
            # random rotation: 0, 90, 180, 270 degrees
            k = random.randint(0, 3)
            if k:
                lr_tensor = torch.rot90(lr_tensor, k, dims=[1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, dims=[1, 2])

        return lr_tensor, hr_tensor


class SRDataset(Dataset):

    def __init__(self,
                 lr_dir,
                 hr_dir,
                 scale=4,
                 patch_size=None,
                 is_train=True):
        super().__init__()

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.is_train = is_train
        self.patch_size = patch_size  # size in LR pixels

        self.lr_files = sorted(glob(os.path.join(lr_dir, "*")))
        self.hr_files = sorted(glob(os.path.join(hr_dir, "*")))

        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(
                f"SRDataset: mismatch between LR and HR files "
                f"({len(self.lr_files)} vs {len(self.hr_files)})"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        lr_img = Image.open(self.lr_files[index]).convert("RGB")
        hr_img = Image.open(self.hr_files[index]).convert("RGB")

        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        # optional random crop in LR space
        if self.is_train and self.patch_size is not None:
            _, h_lr, w_lr = lr_tensor.shape
            patch = self.patch_size

            if h_lr < patch or w_lr < patch:
                # image is too small, skip cropping
                # (still can apply augmentation on full image)
                pass
            else:
                top = random.randint(0, h_lr - patch)
                left = random.randint(0, w_lr - patch)

                lr_tensor = lr_tensor[:, top:top + patch, left:left + patch]

                top_hr = top * self.scale
                left_hr = left * self.scale
                bottom_hr = (top + patch) * self.scale
                right_hr = (left + patch) * self.scale

                hr_tensor = hr_tensor[:, top_hr:bottom_hr, left_hr:right_hr]

        # ----- geometric augmentation (train only) -----
        if self.is_train:
            # random horizontal flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[2])
                hr_tensor = torch.flip(hr_tensor, dims=[2])
            # random vertical flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[1])
                hr_tensor = torch.flip(hr_tensor, dims=[1])
            # random rotation
            k = random.randint(0, 3)
            if k:
                lr_tensor = torch.rot90(lr_tensor, k, dims=[1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, dims=[1, 2])

        return lr_tensor, hr_tensor


class ESRGANDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, scale=4, patch_size=128, is_train=True):
        super().__init__()

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train

        self.lr_paths = _list_images_in_dir(lr_dir)
        self.hr_paths = _list_images_in_dir(hr_dir)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"ESRGANDataset: mismatch between LR and HR images "
                f"({len(self.lr_paths)} vs {len(self.hr_paths)})"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_paths)

    def _get_pair_paths(self, index):
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]
        return lr_path, hr_path

    def __getitem__(self, index):
        lr_path, hr_path = self._get_pair_paths(index)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size

        if self.is_train and self.patch_size is not None:
            hr_patch = self.patch_size
            lr_patch = hr_patch // self.scale

            if hr_w >= hr_patch and hr_h >= hr_patch:
                x_hr = random.randint(0, hr_w - hr_patch)
                y_hr = random.randint(0, hr_h - hr_patch)
            else:
                x_hr = max(0, (hr_w - hr_patch) // 2)
                y_hr = max(0, (hr_h - hr_patch) // 2)

            x_lr = x_hr // self.scale
            y_lr = y_hr // self.scale

            x_lr = min(x_lr, max(lr_w - lr_patch, 0))
            y_lr = min(y_lr, max(lr_h - lr_patch, 0))

            hr_box = (x_hr, y_hr, x_hr + hr_patch, y_hr + hr_patch)
            lr_box = (x_lr, y_lr, x_lr + lr_patch, y_lr + lr_patch)

            hr_img = hr_img.crop(hr_box)
            lr_img = lr_img.crop(lr_box)

        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        # ----- geometric augmentation (train only) -----
        if self.is_train:
            # random horizontal flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[2])
                hr_tensor = torch.flip(hr_tensor, dims=[2])
            # random vertical flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[1])
                hr_tensor = torch.flip(hr_tensor, dims=[1])
            # random rotation
            k = random.randint(0, 3)
            if k:
                lr_tensor = torch.rot90(lr_tensor, k, dims=[1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, dims=[1, 2])

        return lr_tensor, hr_tensor


class SRCNNDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, scale=4, patch_size=None, is_train=True):
        super().__init__()

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train

        self.lr_paths = _list_images_in_dir(lr_dir)
        self.hr_paths = _list_images_in_dir(hr_dir)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"SRCNNDataset: LR/HR length mismatch "
                f"({len(self.lr_paths)} vs {len(self.hr_paths)})"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_paths)

    def _get_pair_paths(self, index):
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]
        return lr_path, hr_path

    def __getitem__(self, index):
        lr_path, hr_path = self._get_pair_paths(index)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        hr_w, hr_h = hr_img.size

        # upsample LR to HR size
        lr_up = lr_img.resize((hr_w, hr_h), resample=Image.Resampling.BICUBIC)

        if self.is_train and self.patch_size is not None:
            if hr_w >= self.patch_size and hr_h >= self.patch_size:
                x = random.randint(0, hr_w - self.patch_size)
                y = random.randint(0, hr_h - self.patch_size)

                crop_box = (x, y, x + self.patch_size, y + self.patch_size)
                hr_img = hr_img.crop(crop_box)
                lr_up = lr_up.crop(crop_box)

        lr_tensor = self.to_tensor(lr_up)
        hr_tensor = self.to_tensor(hr_img)

        # ----- geometric augmentation (train only) -----
        if self.is_train:
            # random horizontal flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[2])
                hr_tensor = torch.flip(hr_tensor, dims=[2])
            # random vertical flip
            if random.random() < 0.5:
                lr_tensor = torch.flip(lr_tensor, dims=[1])
                hr_tensor = torch.flip(hr_tensor, dims=[1])
            # random rotation
            k = random.randint(0, 3)
            if k:
                lr_tensor = torch.rot90(lr_tensor, k, dims=[1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, dims=[1, 2])

        return lr_tensor, hr_tensor
