"""
CelebA Pipeline — ResNet-9
  Step 1 : Face Detection   (img_celeba    → bbox regression,  4 outputs)
  Step 2 : Landmark Detection (img_align_celeba → 5 pt regression, 10 outputs)

Usage:
  python train_celeba.py --task detect   --overfit_only   # single-batch overfit test
  python train_celeba.py --task landmark --overfit_only
  python train_celeba.py --task detect                    # full training
  python train_celeba.py --task landmark
"""

import os
import random
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, ColorJitter
import torchvision.transforms.functional as TF
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'data')
IMG_CELEBA_DIR        = os.path.join(BASE, 'img_celeba')
IMG_ALIGN_CELEBA_DIR  = os.path.join(BASE, 'img_align_celeba')
BBOX_FILE             = os.path.join(BASE, 'landmark', 'list_bbox_celeba.txt')
LANDMARK_FILE         = os.path.join(BASE, 'landmark', 'list_landmarks_align_celeba.txt')

IMG_W, IMG_H = 178, 218   # original CelebA image size
INPUT_SIZE   = 128        # resize target

# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(ch, ch), ConvBlock(ch, ch))

    def forward(self, x):
        return x + self.net(x)


class ResNet9(nn.Module):
    """
    ResNet-9 regression head.
      out_size=4  → face bbox  [x1, y1, w, h]  normalized [0,1]
      out_size=10 → 5 landmarks [x0,y0, x1,y1, ...] normalized [0,1]
    """
    def __init__(self, out_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),               # 128 → 128
            ConvBlock(64, 128, pool=True),  # 128 →  64
            ResBlock(128),
            ConvBlock(128, 256, pool=True), #  64 →  32
            ResBlock(256),
            ConvBlock(256, 512, pool=True), #  32 →  16
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_size),
            # No Sigmoid — regression output is unbounded; model learns [0,1] range naturally
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ─────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────

_base_transform = Compose([Resize((INPUT_SIZE, INPUT_SIZE)), ToTensor()])
_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)


class FaceDetectDataset(Dataset):
    """
    img_celeba + list_bbox_celeba.txt
    Target: [x1, y1, w, h] normalized to [0,1]
    """
    def __init__(self, bbox_file: str, img_dir: str, samples=None, augment: bool = False):
        self.img_dir = img_dir
        self.augment = augment
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            with open(bbox_file) as f:
                lines = f.readlines()
            for line in lines[2:]:          # skip count + header
                p = line.split()
                name = p[0]
                x1, y1, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                self.samples.append((name, x1, y1, w, h))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, x1, y1, w, h = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')

        x1n = x1 / IMG_W
        y1n = y1 / IMG_H
        wn  = w  / IMG_W
        hn  = h  / IMG_H

        if self.augment:
            img = _jitter(img)
            if random.random() < 0.5:
                img  = TF.hflip(img)
                x1n  = 1.0 - x1n - wn  # mirror bbox x

        img_t = _base_transform(img)
        bbox = torch.tensor([x1n, y1n, wn, hn], dtype=torch.float32).clamp(0.0, 1.0)
        return img_t, bbox


class LandmarkDataset(Dataset):
    """
    img_align_celeba + list_landmarks_align_celeba.txt
    Target: [lx,ly, rx,ry, nx,ny, lmx,lmy, rmx,rmy] normalized to [0,1]

    CelebA landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
    Horizontal flip swaps: left_eye↔right_eye, left_mouth↔right_mouth, and mirrors all x.
    """
    def __init__(self, lm_file: str, img_dir: str, samples=None, augment: bool = False):
        self.img_dir = img_dir
        self.augment = augment
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            with open(lm_file) as f:
                lines = f.readlines()
            for line in lines[2:]:
                p = line.split()
                name   = p[0]
                coords = list(map(float, p[1:]))    # 10 values
                self.samples.append((name, coords))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, coords = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')

        # even index = x → /IMG_W,  odd = y → /IMG_H
        norm = np.array([v / (IMG_W if i % 2 == 0 else IMG_H) for i, v in enumerate(coords)],
                        dtype=np.float32)

        if self.augment:
            img = _jitter(img)
            if random.random() < 0.5:
                img = TF.hflip(img)
                # Mirror all x coordinates
                norm[0::2] = 1.0 - norm[0::2]
                # Swap left_eye (0,1) ↔ right_eye (2,3)
                norm[[0, 1, 2, 3]] = norm[[2, 3, 0, 1]]
                # Swap left_mouth (6,7) ↔ right_mouth (8,9)
                norm[[6, 7, 8, 9]] = norm[[8, 9, 6, 7]]

        img_t = _base_transform(img)
        lm = torch.tensor(norm, dtype=torch.float32).clamp(0.0, 1.0)
        return img_t, lm


# ─────────────────────────────────────────────────────────────────
# SINGLE-BATCH OVERFIT TEST
# ─────────────────────────────────────────────────────────────────

def single_batch_overfit(model, loader, device, n_iters=2000, task_name='model'):
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/overfit_{task_name}_{time.strftime("%Y%m%d_%H%M%S")}.txt'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'\n{"="*55}')
    log(f'  Single-batch overfit test — {task_name}')
    log(f'{"="*55}')

    imgs, targets = next(iter(loader))
    imgs, targets = imgs.to(device), targets.to(device)
    log(f'  batch shape : {imgs.shape}  target shape : {targets.shape}')
    log(f'  target min={targets.min():.4f}  max={targets.max():.4f}  mean={targets.mean():.4f}')
    log(f'  device      : {device}')
    log(f'  n_iters     : {n_iters}')
    log(f'  log file    : {log_path}')

    model.to(device).train()
    for m in model.modules():           # disable Dropout, keep BN in train mode
        if isinstance(m, nn.Dropout):
            m.eval()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(1, n_iters + 1):
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i == 1 or i % (n_iters // 10) == 0:
            log(f'  iter {i:4d}  loss = {loss.item():.7f}')

    final = loss.item()
    status = '[PASS]' if final < 1e-4 else '[NOTE] loss not near 0 — check arch/data'
    log(f'\n  Final loss : {final:.7f}  {status}\n')
    return final


# ─────────────────────────────────────────────────────────────────
# TIME ESTIMATE
# ─────────────────────────────────────────────────────────────────

def estimate_training_time(model, loader, device, epochs, n_batches=30):
    model.to(device).train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # warm-up
    imgs, targets = next(iter(loader))
    imgs, targets = imgs.to(device), targets.to(device)
    optimizer.zero_grad()
    criterion(model(imgs), targets).backward()
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for i, (imgs, targets) in enumerate(loader):
        if i >= n_batches:
            break
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        criterion(model(imgs), targets).backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    sec_per_batch = (time.time() - t0) / n_batches
    total_batches  = len(loader) * epochs
    total_sec      = sec_per_batch * total_batches
    total_min      = total_sec / 60

    print(f'\n  [{n_batches} batch avg]  {sec_per_batch*1000:.1f} ms/batch')
    print(f'  {len(loader):,} batches/epoch × {epochs} epochs = {total_batches:,} batches')
    print(f'  Estimated time : {total_min:.1f} min  ({total_min/60:.2f} h)\n')


# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        total += criterion(model(imgs), targets).item()
    return total / len(loader)


def train(model, train_loader, val_loader, device,
          epochs=10, lr=1e-3, task_name='model'):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_path = f'logs/{task_name}_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    best_val  = float('inf')
    patience, patience_ctr = 5, 0

    model.to(device)
    print(f'\nTraining {task_name}  |  device={device}  |  epochs={epochs}')
    print(f'Log → {log_path}\n')

    with open(log_path, 'w') as f:
        f.write(f'task={task_name}  device={device}  epochs={epochs}\n\n')

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss  = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        line = f'Epoch {epoch:3d}/{epochs}  train={tr_loss:.6f}  val={val_loss:.6f}'
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            ckpt = f'checkpoints/{task_name}_best.pt'
            torch.save(model.state_dict(), ckpt)
            print(f'  → saved {ckpt}')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print('  Early stopping.')
                break

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f} min  |  best val loss = {best_val:.6f}')


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',         choices=['detect', 'landmark'], default='detect')
    parser.add_argument('--overfit_only', action='store_true',
                        help='Run single-batch overfit test only')
    parser.add_argument('--time_only',   action='store_true',
                        help='Estimate training time then exit')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--workers',      type=int,   default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Dataset (no augment — used for overfit/time tests) ──
    if args.task == 'detect':
        dataset    = FaceDetectDataset(BBOX_FILE, IMG_CELEBA_DIR)
        model      = ResNet9(out_size=4)
        task_name  = 'face_detect'
    else:
        dataset    = LandmarkDataset(LANDMARK_FILE, IMG_ALIGN_CELEBA_DIR)
        model      = ResNet9(out_size=10)
        task_name  = 'landmark'

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model  : ResNet9  |  out_size={4 if args.task=="detect" else 10}'
          f'  |  params={n_params:,}')
    print(f'Dataset: {len(dataset):,} samples')

    # ── Time estimate only ──
    if args.time_only:
        est_loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        estimate_training_time(model, est_loader, device, epochs=args.epochs)
        return

    # ── Overfit test ──
    overfit_loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
    single_batch_overfit(model, overfit_loader, device, task_name=task_name)

    if args.overfit_only:
        return

    # ── Time estimate ──
    est_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    estimate_training_time(model, est_loader, device, epochs=args.epochs)

    # ── Split samples, create separate datasets with augment only for train ──
    all_samples = dataset.samples
    n = len(all_samples)
    n_test  = int(0.10 * n)
    n_val   = int(0.10 * n)
    n_train = n - n_val - n_test

    indices = list(range(n))
    random.shuffle(indices)
    train_samples = [all_samples[i] for i in indices[:n_train]]
    val_samples   = [all_samples[i] for i in indices[n_train:n_train + n_val]]
    test_samples  = [all_samples[i] for i in indices[n_train + n_val:]]

    if args.task == 'detect':
        train_set = FaceDetectDataset(BBOX_FILE, IMG_CELEBA_DIR, samples=train_samples, augment=True)
        val_set   = FaceDetectDataset(BBOX_FILE, IMG_CELEBA_DIR, samples=val_samples,   augment=False)
        test_set  = FaceDetectDataset(BBOX_FILE, IMG_CELEBA_DIR, samples=test_samples,  augment=False)
    else:
        train_set = LandmarkDataset(LANDMARK_FILE, IMG_ALIGN_CELEBA_DIR, samples=train_samples, augment=True)
        val_set   = LandmarkDataset(LANDMARK_FILE, IMG_ALIGN_CELEBA_DIR, samples=val_samples,   augment=False)
        test_set  = LandmarkDataset(LANDMARK_FILE, IMG_ALIGN_CELEBA_DIR, samples=test_samples,  augment=False)

    print(f'Split  : train={n_train:,}  val={n_val:,}  test={n_test:,}  (augment=True for train)')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # re-init model (overfit test updated weights)
    if args.task == 'detect':
        model = ResNet9(out_size=4)
    else:
        model = ResNet9(out_size=10)

    train(model, train_loader, val_loader, device,
          epochs=args.epochs, lr=args.lr, task_name=task_name)

    # ── Test evaluation ──
    ckpt = f'checkpoints/{task_name}_best.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loss = evaluate(model, test_loader, nn.MSELoss(), device)
    print(f'Test MSE : {test_loss:.6f}')


if __name__ == '__main__':
    main()
