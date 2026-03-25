import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, out_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, pool=True),
            ResBlock(128),
            ConvBlock(128, 256, pool=True),
            ResBlock(256),
            ConvBlock(256, 512, pool=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_size),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="OpenCV face detection on 128x128 + landmark detection on 128x128 face crop."
    )
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="result_128x128.jpg", help="Output image path")
    parser.add_argument(
        "--crop-output",
        type=str,
        default="face_crop_original.jpg",
        help="Optional path to save detected face crop before resize",
    )
    parser.add_argument(
        "--crop-resized-output",
        type=str,
        default="face_crop_128x128.jpg",
        help="Optional path to save detected face crop resized for landmark",
    )
    parser.add_argument(
        "--landmark-ckpt",
        type=str,
        default=os.path.join(script_dir, "checkpoints", "landmark_best.pt"),
        help="Path to landmark checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding ratio around OpenCV detected face bbox",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="OpenCV Haar detectMultiScale scaleFactor",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="OpenCV Haar detectMultiScale minNeighbors",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="Minimum face size in pixels on resized detect image",
    )
    parser.add_argument(
        "--detect-size",
        type=int,
        default=128,
        help="Resize input image to this size for OpenCV face detection",
    )
    parser.add_argument(
        "--landmark-size",
        type=int,
        default=128,
        help="Resize detected face crop to this size for landmark model",
    )
    parser.add_argument("--show", action="store_true", help="Show result window")
    return parser.parse_args()


def load_model(checkpoint_path: str, out_size: int, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ResNet9(out_size=out_size)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def to_tensor_bgr(image_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().div(255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def detect_face_opencv(
    image_bgr: np.ndarray,
    padding: float,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    detect_size: int,
):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade.")

    if detect_size <= 0:
        raise ValueError("detect_size must be > 0")

    img_h, img_w = image_bgr.shape[:2]
    detect_img = cv2.resize(image_bgr, (detect_size, detect_size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    if len(faces) == 0:
        return None

    # Use the largest detected face on detect-size image.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Map resized detection bbox back to original image coordinates.
    sx = img_w / float(detect_size)
    sy = img_h / float(detect_size)
    x = int(x * sx)
    y = int(y * sy)
    w = int(w * sx)
    h = int(h * sy)

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    return x1, y1, x2, y2

def predict_landmarks_128x128(
    landmark_model: nn.Module,
    face_crop_bgr: np.ndarray,
    device: torch.device,
    landmark_size: int,
):
    face_tensor = to_tensor_bgr(face_crop_bgr, size=landmark_size, device=device)
    with torch.no_grad():
        pred = landmark_model(face_tensor).squeeze(0).cpu().numpy().astype(np.float32)

    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    pred = np.clip(pred, 0.0, 1.0).reshape(5, 2)
    return pred


def draw_results(image_bgr: np.ndarray, bbox, landmarks_xy: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    x1, y1, x2, y2 = bbox

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
    cv2.putText(
        out,
        "OpenCV Haar Face",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )

    colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, (x, y) in enumerate(landmarks_xy):
        lx = int(x)
        ly = int(y)
        cv2.circle(out, (lx, ly), 4, colors[i], -1)
        cv2.putText(
            out,
            LANDMARK_NAMES[i],
            (lx + 6, ly - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colors[i],
            1,
            cv2.LINE_AA,
        )

    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    landmark_model = load_model(args.landmark_ckpt, out_size=10, device=device)

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")

    bbox = detect_face_opencv(
        image_bgr=image,
        padding=args.padding,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=args.min_size,
        detect_size=args.detect_size,
    )
    if bbox is None:
        raise RuntimeError("No face detected by OpenCV. Try another image or adjust detection arguments.")

    x1, y1, x2, y2 = bbox
    print(f"Face detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Step 2: Crop detected face from original image.
    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise RuntimeError("Detected face crop is empty.")

    if args.crop_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.crop_output)), exist_ok=True)
        cv2.imwrite(args.crop_output, face_crop)

    # Step 3: Resize cropped face to 128x128 and run landmark detection.
    face_crop_resized = cv2.resize(
        face_crop,
        (args.landmark_size, args.landmark_size),
        interpolation=cv2.INTER_LINEAR,
    )
    if args.crop_resized_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.crop_resized_output)), exist_ok=True)
        cv2.imwrite(args.crop_resized_output, face_crop_resized)

    landmarks_norm = predict_landmarks_128x128(
        landmark_model=landmark_model,
        face_crop_bgr=face_crop,
        device=device,
        landmark_size=args.landmark_size,
    )

    # Convert normalized landmarks from crop space to original image space.
    crop_h, crop_w = face_crop.shape[:2]
    landmarks_xy = landmarks_norm.copy()
    landmarks_xy[:, 0] = landmarks_xy[:, 0] * crop_w + x1
    landmarks_xy[:, 1] = landmarks_xy[:, 1] * crop_h + y1

    result = draw_results(image, bbox, landmarks_xy)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, result)

    print(f"Device: {device}")
    print(f"Input image: {args.input}")
    print(f"Face detector: OpenCV Haar Cascade ({args.detect_size}x{args.detect_size})")
    print(f"Landmark checkpoint: {args.landmark_ckpt}")
    print(f"Face crop resize for landmark: {args.landmark_size}x{args.landmark_size}")
    print(f"Saved raw crop: {args.crop_output}")
    print(f"Saved resized crop: {args.crop_resized_output}")
    print(f"Output saved: {args.output}")

    if args.show:
        cv2.imshow("Face + Landmark 128x128", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
