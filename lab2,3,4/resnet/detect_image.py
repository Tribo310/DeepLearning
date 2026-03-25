"""
Static-path evaluator for one sample from img_align_celeba.

Pipeline:
1) Pick one fixed sample index from list_landmarks_align_celeba.txt
2) Load aligned face image
3) Resize to 128x128 and run landmark model
4) Draw GT vs Pred landmarks and save image
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_celeba import ResNet9


SCRIPT_DIR = os.path.dirname(__file__)
CKPT_LANDMARK = os.path.join(SCRIPT_DIR, 'checkpoints', 'landmark_best.pt')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'results', 'align_celeba_one_sample_eval.jpg')

# User-provided static dataset root.
STATIC_DATA_ROOT = '/home/tr1bo/Documents/8. 3B/F.CSM332 Гүн сургалт/DeepLearning/lab2,3,4/data'

# Static path candidates (first existing one will be used)
LANDMARK_TXT_CANDIDATES = [
    os.path.join(STATIC_DATA_ROOT, 'list_landmarks_align_celeba.txt'),
    os.path.join(STATIC_DATA_ROOT, 'landmark', 'list_landmarks_align_celeba.txt'),
    os.path.join(SCRIPT_DIR, '..', 'data', 'list_landmarks_align_celeba.txt'),
    os.path.join(SCRIPT_DIR, '..', 'data', 'landmark', 'list_landmarks_align_celeba.txt'),
    '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/landmark/list_landmarks_align_celeba.txt',
]

IMG_ALIGN_DIR_CANDIDATES = [
    os.path.join(STATIC_DATA_ROOT, 'img_align_celeba'),
    os.path.join(SCRIPT_DIR, '..', 'data', 'img_align_celeba'),
    '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba',
]

# Static sample index from landmark list (0-based after skipping 2 header lines)
SAMPLE_INDEX = 0

INPUT_SIZE = 128
IMG_W, IMG_H = 178, 218
LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']


def pick_first_existing(paths, kind):
    for p in paths:
        if os.path.exists(p):
            return p
    joined = '\n'.join(paths)
    raise FileNotFoundError(f'No valid {kind} path found. Checked:\n{joined}')


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    model = ResNet9(out_size=10)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def load_one_sample(landmark_txt, img_dir, sample_index):
    if not os.path.exists(landmark_txt):
        raise FileNotFoundError(f'Landmark file not found: {landmark_txt}')

    with open(landmark_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]

    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(f'SAMPLE_INDEX out of range: {sample_index} (max {len(lines)-1})')

    parts = lines[sample_index].split()
    img_name = parts[0]
    gt = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(5, 2)

    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    return image, img_name, gt


def to_tensor_128(image_bgr, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().div(255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def draw_landmarks(image_bgr, points_xy, color, text_prefix):
    out = image_bgr
    for i, (x, y) in enumerate(points_xy):
        px, py = int(round(x)), int(round(y))
        cv2.circle(out, (px, py), 4, color, -1)
        cv2.putText(
            out,
            f'{text_prefix}:{LANDMARK_NAMES[i]}',
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def main():
    landmark_txt = pick_first_existing(LANDMARK_TXT_CANDIDATES, 'landmark file')
    img_align_dir = pick_first_existing(IMG_ALIGN_DIR_CANDIDATES, 'img_align_celeba directory')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CKPT_LANDMARK, device)

    image, img_name, gt_points = load_one_sample(landmark_txt, img_align_dir, SAMPLE_INDEX)
    h, w = image.shape[:2]

    with torch.no_grad():
        pred_norm = model(to_tensor_128(image, device)).squeeze(0).cpu().numpy().reshape(5, 2)

    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    pred_points = np.zeros_like(pred_norm)
    pred_points[:, 0] = pred_norm[:, 0] * w
    pred_points[:, 1] = pred_norm[:, 1] * h

    # GT from CelebA list is already pixel coordinates on aligned image.
    vis = image.copy()
    vis = draw_landmarks(vis, gt_points, color=(0, 255, 0), text_prefix='GT')
    vis = draw_landmarks(vis, pred_points, color=(0, 0, 255), text_prefix='PR')

    mae = float(np.mean(np.abs(gt_points - pred_points)))
    cv2.putText(
        vis,
        f'Sample: {img_name} | idx={SAMPLE_INDEX} | MAE(px)={mae:.2f}',
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, vis)

    print(f'Device: {device}')
    print(f'Landmark list: {landmark_txt}')
    print(f'Aligned image dir: {img_align_dir}')
    print(f'Image: {img_name}')
    print(f'SAMPLE_INDEX: {SAMPLE_INDEX}')
    print(f'Checkpoint: {CKPT_LANDMARK}')
    print(f'MAE(px): {mae:.4f}')
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
