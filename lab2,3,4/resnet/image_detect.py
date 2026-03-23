import argparse
from pathlib import Path

import cv2
import torch

from resnet9 import ResNet9Landmark


LANDMARK_NAMES = [
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth",
    "right_mouth",
]

# If you do not pass --input, this path will be used.
DEFAULT_INPUT_IMAGE = "/home/tr1bo/Documents/8. 3B/F.CSM332 Гүн сургалт/DeepLearning/lab2,3,4/resnet/test.png"


def parse_args() -> argparse.Namespace:
    default_checkpoint = (
        "/home/tr1bo/Documents/8. 3B/F.CSM332 Гүн сургалт/DeepLearning/"
        "lab2,3,4/resnet/checkpoints/resnet9_landmark_epoch2_loss0.0000.pt"
    )
    parser = argparse.ArgumentParser(
        description="Detect face landmarks on a single image with ResNet9."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_IMAGE,
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detect_result.jpg",
        help="Output image path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.15,
        help="Face detector scaleFactor.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Face detector minNeighbors.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Extra padding ratio around detected face.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result window after saving.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> ResNet9Landmark:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
    face_tensor = torch.from_numpy(face_resized).float() / 255.0
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
    return face_tensor


def padded_square_box(x, y, w, h, frame_w, frame_h, padding_ratio):
    side = int(max(w, h) * (1.0 + padding_ratio))
    cx = x + w // 2
    cy = y + h // 2

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(frame_w, x0 + side)
    y1 = min(frame_h, y0 + side)

    side = min(x1 - x0, y1 - y0)
    x1 = x0 + side
    y1 = y0 + side
    return x0, y0, x1, y1


def draw_landmarks(frame, box, preds):
    x, y, w, h = box
    points = preds.reshape(-1, 2)

    for idx, (px, py) in enumerate(points):
        px = float(max(0.0, min(1.0, px)))
        py = float(max(0.0, min(1.0, py)))
        lx = int(x + px * w)
        ly = int(y + py * h)
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        print("No face detected.")
    else:
        print(f"Detected faces: {len(faces)}")

    for (x, y, w, h) in faces:
        x0, y0, x1, y1 = padded_square_box(
            x,
            y,
            w,
            h,
            image.shape[1],
            image.shape[0],
            args.padding,
        )
        face_roi = image[y0:y1, x0:x1]
        if face_roi.size == 0:
            continue

        face_tensor = preprocess_face(face_roi).to(device)
        with torch.no_grad():
            pred = model(face_tensor).squeeze(0).cpu().numpy()

        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 180, 0), 2)
        draw_landmarks(image, (x0, y0, x1 - x0, y1 - y0), pred)

    cv2.imwrite(args.output, image)
    print(f"Saved output image: {args.output}")

    if args.show:
        cv2.imshow("Landmark Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()