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


def parse_args() -> argparse.Namespace:
	default_checkpoint = (
		"/home/tr1bo/Documents/8. 3B/F.CSM332 Гүн сургалт/DeepLearning/"
		"lab2,3,4/resnet/checkpoints/resnet9_landmark_epoch2_loss0.0000.pt"
	)
	parser = argparse.ArgumentParser(
		description="Run webcam face landmark detection with ResNet9 checkpoint."
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=default_checkpoint,
		help="Path to .pt checkpoint file.",
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=0,
		help="Webcam index for OpenCV VideoCapture.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		choices=["cpu", "cuda"],
		help="Device to run inference on.",
	)
	parser.add_argument(
		"--conf-threshold",
		type=float,
		default=1.1,
		help=(
			"Face detector minNeighbors value proxy: higher means stricter detection. "
			"Mapped internally to int minNeighbors."
		),
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


def draw_landmarks(frame, box, preds):
	x, y, w, h = box
	points = preds.reshape(-1, 2)

	for idx, (px, py) in enumerate(points):
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
	print(f"Running on device: {device}")

	face_cascade = cv2.CascadeClassifier(
		cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	)
	if face_cascade.empty():
		raise RuntimeError("Failed to load Haar cascade for face detection.")

	cap = cv2.VideoCapture(args.camera_index)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open webcam index {args.camera_index}.")

	min_neighbors = max(3, int(round(args.conf_threshold * 4)))

	print("Press 'q' to quit")
	while True:
		ok, frame = cap.read()
		if not ok:
			print("Failed to read frame from webcam.")
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.2,
			minNeighbors=min_neighbors,
			minSize=(80, 80),
		)

		for (x, y, w, h) in faces:
			x0, y0 = max(0, x), max(0, y)
			x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
			face_roi = frame[y0:y1, x0:x1]
			if face_roi.size == 0:
				continue

			face_tensor = preprocess_face(face_roi).to(device)
			with torch.no_grad():
				pred = model(face_tensor).squeeze(0).cpu().numpy()

			cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 180, 0), 2)
			draw_landmarks(frame, (x0, y0, x1 - x0, y1 - y0), pred)

		cv2.imshow("ResNet9 Landmark Webcam Detect", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
