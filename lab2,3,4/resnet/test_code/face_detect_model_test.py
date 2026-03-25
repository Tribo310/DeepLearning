import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RESNET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RESNET_DIR))

from train_celeba import ResNet9  # noqa: E402


CKPT_PATH = RESNET_DIR / "checkpoints" / "face_detect_best.pt"
DEFAULT_INPUT = RESNET_DIR / "img" / "image2.png"
DEFAULT_OUTPUT = RESNET_DIR / "results" / "test_codo_model_result" / "face_crop_image2.jpg"
INPUT_SIZE = 128


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Detect face and save cropped face image.")
	parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input image path")
	parser.add_argument("--checkpoint", type=str, default=str(CKPT_PATH), help="Path to face_detect_best.pt")
	parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output cropped face image path")
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		choices=["cpu", "cuda"],
		help="Inference device",
	)
	return parser.parse_args()


def preprocess(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
	tensor = torch.from_numpy(resized).float().div(255.0)
	return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def decode_bbox(pred: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
	# Model target in training: [x1, y1, bw, bh] mostly normalized to [0,1].
	pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

	x1n = float(np.clip(pred[0], 0.0, 1.0))
	y1n = float(np.clip(pred[1], 0.0, 1.0))
	bwn = float(np.clip(pred[2], 0.01, 1.0))
	bhn = float(np.clip(pred[3], 0.01, 1.0))

	x1 = int(x1n * w)
	y1 = int(y1n * h)
	bw = int(bwn * w)
	bh = int(bhn * h)

	x2 = min(w, x1 + max(1, bw))
	y2 = min(h, y1 + max(1, bh))
	x1 = max(0, min(x1, w - 1))
	y1 = max(0, min(y1, h - 1))
	return x1, y1, x2, y2


def main() -> None:
	args = parse_args()
	device = torch.device(args.device)

	ckpt = Path(args.checkpoint)
	if not ckpt.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

	model = ResNet9(out_size=4)
	state = torch.load(str(ckpt), map_location=device)
	model.load_state_dict(state)
	model.to(device).eval()

	image_path = Path(args.input)
	if not image_path.exists():
		raise FileNotFoundError(f"Input image not found: {image_path}")

	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"Cannot read image: {image_path}")

	h, w = image.shape[:2]
	inp = preprocess(image, device)
	with torch.no_grad():
		pred = model(inp).squeeze(0).cpu().numpy()

	x1, y1, x2, y2 = decode_bbox(pred, w=w, h=h)
	face_crop = image[y1:y2, x1:x2]
	if face_crop.size == 0:
		raise RuntimeError("Detected face crop is empty")

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(output_path), face_crop)

	print(f"Checkpoint: {ckpt}")
	print(f"Device: {device}")
	print(f"Input: {image_path}")
	print(f"BBox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
	print(f"Saved crop: {output_path}")


if __name__ == "__main__":
	main()
