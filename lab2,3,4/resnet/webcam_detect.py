import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_celeba import ResNet9


LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
INPUT_SIZE = 128
TARGET_W = 178
TARGET_H = 218


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Webcam face landmark detection")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(script_dir / "checkpoints" / "landmark_best.pt"),
        help="Path to landmark checkpoint",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar minNeighbors")
    parser.add_argument("--min-size", type=int, default=80, help="Haar min face size")
    parser.add_argument("--padding", type=float, default=0.15, help="Padding around detected face")
    parser.add_argument("--roi-scale", type=float, default=1.0, help="Scale factor for detected ROI")
    parser.add_argument("--roi-shift-x", type=float, default=0.0, help="Horizontal ROI shift ratio")
    parser.add_argument("--roi-shift-y", type=float, default=-0.06, help="Vertical ROI shift ratio")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA smoothing factor (0 disables)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = ResNet9(out_size=10)
    state_dict = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def preprocess_face(face_bgr, device: torch.device) -> torch.Tensor:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    face_tensor = torch.from_numpy(face_resized).float().div(255.0)
    return face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def padded_square_box(x, y, w, h, frame_w, frame_h, padding_ratio, roi_scale, shift_x, shift_y):
    side = int(max(w, h) * (1.0 + padding_ratio))
    side = int(max(1, side * roi_scale))
    cx = int(x + w * (0.5 + shift_x))
    cy = int(y + h * (0.5 + shift_y))

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(frame_w, x0 + side)
    y1 = min(frame_h, y0 + side)

    side = min(x1 - x0, y1 - y0)
    x1 = x0 + side
    y1 = y0 + side
    return x0, y0, x1, y1


def draw_landmarks_aligned(frame, points_xy, ema_points, ema_alpha):
    current = []

    for lx, ly in points_xy:
        current.append([lx, ly])

    current = np.array(current, dtype=np.float32)
    if ema_alpha > 0.0:
        if ema_points is None:
            smoothed = current
        else:
            smoothed = ema_alpha * current + (1.0 - ema_alpha) * ema_points
    else:
        smoothed = current

    for idx, (lx, ly) in enumerate(smoothed):
        lx = int(lx)
        ly = int(ly)
        cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)
        cv2.putText(
            frame,
            LANDMARK_NAMES[idx],
            (lx + 5, ly - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (50, 255, 50),
            1,
            cv2.LINE_AA,
        )

    return smoothed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}")

    print(f"Loaded checkpoint: {args.checkpoint}")
    print("Press 'q' to quit")
    print(
        f"ROI tune: scale={args.roi_scale}, shift_x={args.roi_shift_x}, "
        f"shift_y={args.roi_shift_y}, ema_alpha={args.ema_alpha}"
    )

    ema_points = None

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_size, args.min_size),
            )

            for (x, y, w, h) in faces:
                x0, y0, x1, y1 = padded_square_box(
                    x,
                    y,
                    w,
                    h,
                    frame.shape[1],
                    frame.shape[0],
                    args.padding,
                    args.roi_scale,
                    args.roi_shift_x,
                    args.roi_shift_y,
                )

                # CelebA-style affine alignment to 178x218 before resizing to model input.
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                box_h = max(1.0, float(y1 - y0))
                s = (TARGET_H * 0.45) / box_h
                tx = (TARGET_W / 2.0) - s * cx
                ty = (TARGET_H * 0.565) - s * cy
                M = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)

                aligned = cv2.warpAffine(frame, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR)
                face_tensor = preprocess_face(aligned, device)
                pred = model(face_tensor).squeeze(0).cpu().numpy().reshape(-1, 2)

                # Map normalized predictions (on aligned canvas) back to original frame.
                points_xy = []
                for px, py in pred:
                    ax = float(max(0.0, min(1.0, px))) * TARGET_W
                    ay = float(max(0.0, min(1.0, py))) * TARGET_H
                    ox = int((ax - tx) / max(s, 1e-6))
                    oy = int((ay - ty) / max(s, 1e-6))
                    points_xy.append((ox, oy))

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 180, 0), 2)
                ema_points = draw_landmarks_aligned(
                    frame,
                    points_xy,
                    ema_points,
                    args.ema_alpha,
                )

            cv2.imshow("Webcam Landmark Detect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()