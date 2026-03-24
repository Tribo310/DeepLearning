"""
Dual Model Landmark Detection Pipeline
======================================
2 моделийг ашиглаж landmark detection хийх:
1. Face Detection Model (ResNet9) - нүүрний bounding box илрүүлэх
2. Landmark Detection Model (ResNet9) - 5 цэг (нүд, хамар, сая)-ыг илрүүлэх

Usage:
    python dual_model_landmark_detection.py --input img/tony.png --show
    python dual_model_landmark_detection.py --video video.mp4 --output output.mp4
"""

import argparse
import torch
import torch.nn as nn
import cv2
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(__file__))

# =========================================
# MODEL DEFINITIONS
# =========================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# =========================================
# CONFIGURATION
# =========================================

CKPT_DETECT = os.path.join(os.path.dirname(__file__), 'checkpoints', 'face_detect_best.pt')
CKPT_LANDMARK = os.path.join(os.path.dirname(__file__), 'checkpoints', 'landmark_best.pt')

INPUT_SIZE = 128
LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

# BBox output: [x1, y1, x2, y2]
# Landmark output: [lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy]


# =========================================
# MODEL LOADING & INFERENCE
# =========================================

class DualModelDetector:
    """Нүүр болон landmark-ыг илрүүлэх pipeline"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.face_detector = None
        self.landmark_detector = None
        self.load_models()
    
    def load_models(self):
        """Хоёр моделийг ачаалах"""
        # Model 1: Face Detection
        self.face_detector = ResNet9(out_size=4)
        if os.path.exists(CKPT_DETECT):
            self.face_detector.load_state_dict(
                torch.load(CKPT_DETECT, map_location=self.device)
            )
            print(f"✓ Face detector loaded: {CKPT_DETECT}")
        else:
            print(f"⚠ Face detector checkpoint not found: {CKPT_DETECT}")
        
        # Model 2: Landmark Detection
        self.landmark_detector = ResNet9(out_size=10)
        if os.path.exists(CKPT_LANDMARK):
            self.landmark_detector.load_state_dict(
                torch.load(CKPT_LANDMARK, map_location=self.device)
            )
            print(f"✓ Landmark detector loaded: {CKPT_LANDMARK}")
        else:
            print(f"⚠ Landmark detector checkpoint not found: {CKPT_LANDMARK}")
        
        self.face_detector = self.face_detector.to(self.device).eval()
        self.landmark_detector = self.landmark_detector.to(self.device).eval()
    
    def to_tensor(self, image_bgr):
        """BGR image → normalized tensor"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().div(255.0)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
    
    def detect_face(self, image):
        """
        Нүүр илрүүлэх — OpenCV Haar cascade ашиглана
        Output: bbox [x1, y1, x2, y2] буюу None
        """
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None

        # Хамгийн том нүүрийг авах
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        # Padding нэмэх
        pad = int(max(w, h) * 0.15)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(image.shape[1], x + w + pad)
        y2  = min(image.shape[0], y + h + pad)
        return (x1, y1, x2, y2)
    
    def detect_landmarks(self, face_crop):
        """
        Face crop дотроос 5 landmark-ыг илрүүлэх
        Input: face crop image (H, W, 3)
        Output: landmarks [(x1, y1), (x2, y2), ..., (x5, y5)]
        """
        h, w = face_crop.shape[:2]
        tensor = self.to_tensor(face_crop)
        
        with torch.no_grad():
            output = self.landmark_detector(tensor)  # [1, 10]
        
        landmarks_norm = output[0].cpu().numpy()  # [10]
        
        # Reshape to 5 points
        landmarks = landmarks_norm.reshape(5, 2)
        
        # Denormalize to face crop coordinates
        landmarks[:, 0] *= w
        landmarks[:, 1] *= h
        
        return landmarks
    
    def process_image(self, image_path):
        """
        Зургаас нүүр болон landmark-ыг илрүүлэх
        
        Returns:
            image: Зураг
            face_bbox: Нүүрний bounding box буюу None
            landmarks: Normalized 5-point landmarks буюу None
            landmarks_pixels: Original image дахь pixel coordinates буюу None
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        
        # Step 1: Detect face
        bbox = self.detect_face(image)
        if bbox is None:
            print("No face detected")
            return image, None, None, None
        
        x1, y1, x2, y2 = bbox
        face_crop = image[y1:y2, x1:x2]
        
        # Step 2: Detect landmarks
        landmarks = self.detect_landmarks(face_crop)
        
        # Convert to original image coordinates
        landmarks_pixels = landmarks.copy()
        landmarks_pixels[:, 0] += x1
        landmarks_pixels[:, 1] += y1
        
        return image, bbox, landmarks, landmarks_pixels


# =========================================
# VISUALIZATION
# =========================================

def draw_results(image, bbox, landmarks, output_path=None, show=False):
    """Нүүр болон landmark-уудыг зургаар харуулах"""
    img = image.copy()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Face", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if landmarks is not None:
        colors = [
            (255, 0, 0),    # Left eye - Blue
            (0, 0, 255),    # Right eye - Red
            (0, 255, 0),    # Nose - Green
            (255, 0, 255),  # Left mouth - Magenta
            (0, 255, 255)   # Right mouth - Yellow
        ]
        
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(img, (int(x), int(y)), 5, colors[i], -1)
            cv2.putText(img, LANDMARK_NAMES[i], (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"✓ Result saved: {output_path}")
    
    if show:
        cv2.imshow("Dual Model Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


def plot_landmarks(image, bbox, landmarks, output_path=None):
    """Matplotlib ашиглан илүү сайн харуулах"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # BGR → RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_image)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    
    if landmarks is not None:
        colors = ['blue', 'red', 'green', 'magenta', 'yellow']
        
        for i, (x, y) in enumerate(landmarks):
            ax.plot(x, y, 'o', color=colors[i], markersize=10, label=LANDMARK_NAMES[i])
            ax.text(x + 5, y - 5, LANDMARK_NAMES[i], color=colors[i], fontsize=10)
    
    ax.legend()
    ax.set_title("Dual Model Landmark Detection", fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_path}")
    
    plt.show()


# =========================================
# MAIN EXECUTION
# =========================================

def main():
    parser = argparse.ArgumentParser(description="Dual Model Landmark Detection")
    parser.add_argument('--input', type=str, default='img/tony.png',
                       help='Input image path')
    parser.add_argument('--output', type=str, default='results/result.jpg',
                       help='Output image path')
    parser.add_argument('--plot', type=str, default=None,
                       help='Save matplotlib plot')
    parser.add_argument('--show', action='store_true',
                       help='Display result')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    print("=" * 60)
    print("Dual Model Landmark Detection Pipeline")
    print("=" * 60)
    
    # Initialize detector
    detector = DualModelDetector(device=args.device)
    print(f"Device: {detector.device}\n")
    
    # Process image
    print(f"Processing: {args.input}")
    image, bbox, landmarks, landmarks_pixels = detector.process_image(args.input)
    
    if bbox is not None and landmarks_pixels is not None:
        print(f"✓ Face detected: {bbox}")
        print(f"✓ Landmarks detected:")
        for i, name in enumerate(LANDMARK_NAMES):
            x, y = landmarks_pixels[i]
            print(f"  {name}: ({x:.1f}, {y:.1f})")
        
        # Visualize
        result_img = draw_results(image, bbox, landmarks_pixels,
                                  output_path=args.output, show=args.show)
        
        if args.plot:
            plot_landmarks(image, bbox, landmarks_pixels, output_path=args.plot)
    else:
        print("✗ Could not detect face or landmarks")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
