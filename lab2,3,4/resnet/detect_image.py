"""
Face detection + 5-point landmark pipeline on a single image.

Usage:
    python detect_image.py --input img/tony.png --show
"""

import argparse
import torch
import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from train_celeba import ResNet9

CKPT_DETECT   = os.path.join(os.path.dirname(__file__), 'checkpoints', 'face_detect_best.pt')
CKPT_LANDMARK = os.path.join(os.path.dirname(__file__), 'checkpoints', 'landmark_best.pt')
INPUT_SIZE    = 128
LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']


def load_model(ckpt_path, out_size, device):
    model = ResNet9(out_size=out_size)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def to_tensor(image_bgr, device):
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor  = torch.from_numpy(resized).float().div(255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, default='img/tony.png')
    parser.add_argument('--output', type=str, default='img/tony_result.jpg')
    parser.add_argument('--show',   action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detect_model   = load_model(CKPT_DETECT,   out_size=4,  device=device)
    landmark_model = load_model(CKPT_LANDMARK, out_size=10, device=device)
    print(f'Models loaded  |  device={device}')

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f'Cannot read: {args.input}')
    h_orig, w_orig = image.shape[:2]

    # ── Step 1: Face bbox ──
    with torch.no_grad():
        pred_bbox = detect_model(to_tensor(image, device)).squeeze(0).cpu().numpy()

    x1 = int(pred_bbox[0] * w_orig)
    y1 = int(pred_bbox[1] * h_orig)
    bw = int(pred_bbox[2] * w_orig)
    bh = int(pred_bbox[3] * h_orig)
    x2, y2 = x1 + bw, y1 + bh
    print(f'BBox  x1={x1} y1={y1} w={bw} h={bh}')

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # ── Step 2: Landmarks on cropped face ──
    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w_orig, x2);  y2c = min(h_orig, y2)
    face_crop = image[y1c:y2c, x1c:x2c]

    if face_crop.size > 0:
        with torch.no_grad():
            pred_lm = landmark_model(to_tensor(face_crop, device)).squeeze(0).cpu().numpy()

        for i, name in enumerate(LANDMARK_NAMES):
            lx = int(x1c + pred_lm[i * 2]     * (x2c - x1c))
            ly = int(y1c + pred_lm[i * 2 + 1] * (y2c - y1c))
            cv2.circle(image, (lx, ly), 5, (0, 255, 0), -1)
            cv2.putText(image, name, (lx + 6, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 50), 1, cv2.LINE_AA)
        print(f'Landmarks drawn: {LANDMARK_NAMES}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, image)
    print(f'Saved → {args.output}')

    if args.show:
        cv2.imshow('Face + Landmarks', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
