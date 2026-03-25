"""
2 Моделыг тус тусдаа шалгах
============================
Model 1 - Face Detection: Нүүрний bounding box илрүүлэх
Model 2 - Landmark Detection: 5 цэг илрүүлэх

Usage:
    python test_models_separately.py --input img/tony.png
"""

import argparse
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =========================================
# MODEL ARCHITECTURE
# =========================================

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


# =========================================
# MODEL 1: FACE DETECTION
# =========================================

class FaceDetectionTester:
    """Нүүр илрүүлэх модель (Model 1)"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ResNet9(out_size=4)
        
        if os.path.exists(CKPT_DETECT):
            self.model.load_state_dict(
                torch.load(CKPT_DETECT, map_location=self.device)
            )
            print(f"✓ Face Detection Model loaded")
        else:
            print(f"✗ Checkpoint not found: {CKPT_DETECT}")
        
        self.model = self.model.to(self.device).eval()
    
    def to_tensor(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().div(255.0)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
    
    def detect(self, image):
        """
        Face bounding box илрүүлэх
        Гаралт: [x1, y1, x2, y2] normalized [0, 1]
        """
        h, w = image.shape[:2]
        tensor = self.to_tensor(image)
        
        with torch.no_grad():
            output = self.model(tensor)
        
        bbox_raw = output[0].cpu().numpy().astype(np.float32)
        
        # Check if output is normalized or in pixel coordinates
        # If values are > 2, likely in pixel coordinates already
        if np.max(np.abs(bbox_raw)) > 2:
            # Already in pixel coordinates
            bbox_pixels = np.clip(bbox_raw, 0, max(w, h)).astype(int)
            bbox_norm = bbox_raw / np.array([w, h, w, h])
        else:
            # Normalized [0, 1]
            bbox_norm = np.clip(bbox_raw, 0, 1)
            bbox_pixels = (bbox_norm * np.array([w, h, w, h])).astype(int)

        # output is [x1, y1, w, h] not [x1, y1, x2, y2]
        x1 = bbox_pixels[0]
        y1 = bbox_pixels[1]
        x2 = bbox_pixels[0] + bbox_pixels[2]
        y2 = bbox_pixels[1] + bbox_pixels[3]
        
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        
        return (x1, y1, x2, y2), bbox_norm


# =========================================
# MODEL 2: LANDMARK DETECTION
# =========================================

class LandmarkDetectionTester:
    """Landmark илрүүлэх модель (Model 2)"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ResNet9(out_size=10)
        
        if os.path.exists(CKPT_LANDMARK):
            self.model.load_state_dict(
                torch.load(CKPT_LANDMARK, map_location=self.device)
            )
            print(f"✓ Landmark Detection Model loaded")
        else:
            print(f"✗ Checkpoint not found: {CKPT_LANDMARK}")
        
        self.model = self.model.to(self.device).eval()
    
    def to_tensor(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().div(255.0)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
    
    def detect(self, image):
        """
        5 landmark-ыг илрүүлэх (нүд, нүд, хамар, сая, сая)
        Гаралт: landmarks normalized [0, 1]
        """
        h, w = image.shape[:2]
        tensor = self.to_tensor(image)
        
        with torch.no_grad():
            output = self.model(tensor)
        
        landmarks_norm = output[0].cpu().numpy().astype(np.float32)
        landmarks_norm = landmarks_norm.reshape(5, 2)
        landmarks_norm = np.clip(landmarks_norm, 0, 1)
        
        # Denormalize
        landmarks_pixels = landmarks_norm.copy()
        landmarks_pixels[:, 0] *= w
        landmarks_pixels[:, 1] *= h
        
        return landmarks_pixels, landmarks_norm


# =========================================
# VISUALIZATION
# =========================================

def visualize_face_detection(image, bbox, output_path=None):
    """Model 1 үр дүнг харуулах"""
    img = image.copy()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, "FACE DETECTED", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"({x1}, {y1}) -> ({x2}, {y2})", 
                   (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


def visualize_landmarks(image, face_crop, landmarks, output_path=None):
    """Model 2 үр дүнг харуулах"""
    img = image.copy()
    
    colors = [
        (255, 0, 0),    # Left eye - Blue
        (0, 0, 255),    # Right eye - Red
        (0, 255, 0),    # Nose - Green
        (255, 0, 255),  # Left mouth - Magenta
        (0, 255, 255)   # Right mouth - Yellow
    ]
    
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), 8, colors[i], -1)
        cv2.circle(img, (int(x), int(y)), 8, (255, 255, 255), 2)
        cv2.putText(img, LANDMARK_NAMES[i], (int(x) + 10, int(y) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


def plot_model_comparison(image, bbox, landmarks, output_path=None):
    """Хоёр моделийн үр дүнг нэг дээр нь харуулах"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ========== MODEL 1: Face Detection ==========
    ax = axes[0]
    ax.imshow(rgb_image)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, "FACE", color='lime', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title("MODEL 1: Face Detection\n(Bounding Box)", 
                fontsize=14, fontweight='bold', color='lime')
    ax.axis('off')
    
    # ========== MODEL 2: Landmark Detection ==========
    ax = axes[1]
    ax.imshow(rgb_image)
    
    colors_rgb = [
        (1, 0, 0),      # Left eye - Blue
        (0, 0, 1),      # Right eye - Red
        (0, 1, 0),      # Nose - Green
        (1, 0, 1),      # Left mouth - Magenta
        (0, 1, 1)       # Right mouth - Yellow
    ]
    
    if landmarks is not None:
        for i, (x, y) in enumerate(landmarks):
            ax.plot(x, y, 'o', color=colors_rgb[i], markersize=12, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.text(x + 15, y - 10, LANDMARK_NAMES[i], color=colors_rgb[i], 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title("MODEL 2: Landmark Detection\n(5-point landmarks)", 
                fontsize=14, fontweight='bold', color='lime')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Comparison plot saved: {output_path}")
    
    plt.show()


def print_results_table(image_path, bbox, bbox_norm, landmarks, landmarks_norm):
    """Үр дүнг хүснэгтээр харуулах"""
    print("\n" + "=" * 80)
    print(f"Input Image: {image_path}")
    print("=" * 80)
    
    # ========== MODEL 1 RESULTS ==========
    print("\n📦 MODEL 1: FACE DETECTION")
    print("-" * 80)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        print(f"  Status          : ✓ Face Detected")
        print(f"  BBox (pixels)   : X1={x1}, Y1={y1}, X2={x2}, Y2={y2}")
        print(f"  BBox (normalized): X1={bbox_norm[0]:.4f}, Y1={bbox_norm[1]:.4f}, "
              f"X2={bbox_norm[2]:.4f}, Y2={bbox_norm[3]:.4f}")
        print(f"  Width (pixels)  : {x2 - x1}")
        print(f"  Height (pixels) : {y2 - y1}")
    else:
        print(f"  Status          : ✗ No face detected")
    
    # ========== MODEL 2 RESULTS ==========
    print("\n🎯 MODEL 2: LANDMARK DETECTION")
    print("-" * 80)
    if landmarks is not None and len(landmarks) == 5:
        print(f"  Status          : ✓ {len(landmarks)} Landmarks Detected\n")
        print(f"  {'Landmark Name':<20} {'X (pixels)':<15} {'Y (pixels)':<15} "
              f"{'X (norm)':<12} {'Y (norm)':<12}")
        print("-" * 80)
        
        # Reshape landmarks_norm if needed
        if isinstance(landmarks_norm, np.ndarray):
            if len(landmarks_norm.shape) == 1:
                landmarks_norm_2d = landmarks_norm.reshape(5, 2)
            else:
                landmarks_norm_2d = landmarks_norm
        else:
            landmarks_norm_2d = np.zeros((5, 2))
        
        for i, name in enumerate(LANDMARK_NAMES):
            x_pix, y_pix = landmarks[i]
            x_norm, y_norm = landmarks_norm_2d[i]
            print(f"  {name:<20} {x_pix:<15.2f} {y_pix:<15.2f} "
                  f"{x_norm:<12.4f} {y_norm:<12.4f}")
    else:
        print(f"  Status          : ✗ Failed to detect landmarks")
    
    print("\n" + "=" * 80 + "\n")


# =========================================
# MAIN EXECUTION
# =========================================

def main():
    parser = argparse.ArgumentParser(description="2 Моделыг тус тусдаа шалгах")
    parser.add_argument('--input', type=str, default='img/tony.png',
                       help='Input image path')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("2 МОДЕЛЫГ ТУС ТУСДАА ШАЛГАХ")
    print("=" * 80)
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"✗ Cannot read: {args.input}")
        return
    
    print(f"\n📷 Image loaded: {args.input}")
    print(f"   Size: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"   Device: {args.device}")
    
    # ========== TEST MODEL 1: Face Detection ==========
    print("\n" + "─" * 80)
    print("🔍 Testing MODEL 1: Face Detection")
    print("─" * 80)
    
    face_detector = FaceDetectionTester(device=args.device)
    bbox, bbox_norm = face_detector.detect(image)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        face_crop = image[y1:y2, x1:x2]
        print(f"✓ Face BBox: {bbox}")
        print(f"✓ Face crop size: {face_crop.shape[1]} x {face_crop.shape[0]}")
    else:
        print("✗ No face detected")
        face_crop = None
    
    # Save face detection result
    face_det_img = visualize_face_detection(image, bbox, 
                                            os.path.join(args.output_dir, 'model1_face_detection.jpg'))
    print(f"✓ Result saved: {args.output_dir}/model1_face_detection.jpg")
    
    # ========== TEST MODEL 2: Landmark Detection ==========
    print("\n" + "─" * 80)
    print("🔍 Testing MODEL 2: Landmark Detection")
    print("─" * 80)
    
    landmark_detector = LandmarkDetectionTester(device=args.device)
    
    if face_crop is not None:
        landmarks, landmarks_norm = landmark_detector.detect(face_crop)
        
        # Convert to original image coordinates
        landmarks_orig = landmarks.copy()
        landmarks_orig[:, 0] += x1
        landmarks_orig[:, 1] += y1
        
        print(f"✓ {len(landmarks)} landmarks detected")
        for i, name in enumerate(LANDMARK_NAMES):
            x, y = landmarks_orig[i]
            print(f"  {name}: ({x:.1f}, {y:.1f})")
    else:
        print("✗ Face crop not available")
        landmarks_orig = None
        landmarks_norm = None
    
    # Save landmark detection result
    if landmarks_orig is not None:
        landmark_img = visualize_landmarks(image, face_crop, landmarks_orig,
                                          os.path.join(args.output_dir, 'model2_landmarks_detection.jpg'))
        print(f"✓ Result saved: {args.output_dir}/model2_landmarks_detection.jpg")
    
    # ========== RESULTS TABLE ==========
    print_results_table(args.input, bbox, bbox_norm, landmarks_orig, landmarks_norm)
    
    # ========== COMPARISON PLOT ==========
    if bbox is not None and landmarks_orig is not None:
        plot_model_comparison(image, bbox, landmarks_orig,
                            os.path.join(args.output_dir, 'model_comparison.png'))
    
    print("✓ All tests completed!")


if __name__ == '__main__':
    main()
