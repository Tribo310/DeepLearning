"""
Two-Stage Pipeline: Face Detection + Landmark Detection
1. MediaPipe-оор нүүр илрүүлэх
2. ResNet9Landmark-оор landmark детектшн хийх
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import mediapipe as mp
from PIL import Image
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import Resize, ToTensor, Compose
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from resnet9 import ResNet9Landmark


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        self.landmarks_frame = []
        file = open(csv_file, "r")
        while True:
            content = file.readline()
            if not content:
                break
            self.landmarks_frame.append(content.split())
        file.close()
        self.landmarks_frame = self.landmarks_frame[2:]
        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root_dir, self.landmarks_frame[idx][0])
        image = io.imread(img_name)
        landmarks = np.array([self.landmarks_frame[idx][1:]], dtype=float).reshape(-1, 2)
        
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] = scaled_landmarks[:, 0] / 178
        scaled_landmarks[:, 1] = scaled_landmarks[:, 1] / 218
        scaled_landmarks = scaled_landmarks.reshape(-1)
        
        image_pil = Image.fromarray(image)
        transform_img = Compose([
            Resize((128, 128)),
            ToTensor(),
        ])
        image_tr = transform_img(image_pil)
        
        sample = {'image_tr': image_tr, 'scaled_landmarks': scaled_landmarks}
        return sample


def load_trained_model(checkpoint_path, device):
    """Хадгалсан моделийг ачаалах"""
    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def detect_faces_yolo(image, confidence_threshold=0.5):
    """
    YOLO эсвэл санамсаргүй OpenCV ашиглан нүүр илрүүлэх
    (MediaPipe-ын API өөрчлөгдсөн)
    """
    # OpenCV Haar Cascade ашиглана
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Haar Cascade параметрүүд
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Масштаб дээшлүүлэх
        minNeighbors=5,       # Нүүрний эргэн тойрон байх хэмжээ
        minSize=(30, 30),     # Хамгийн бага нүүрний хэмжээ
        maxSize=(500, 500)    # Хамгийн их нүүрний хэмжээ
    )
    
    return [(x, y, w, h) for x, y, w, h in faces]


def detect_faces_opencv(image):
    """
    OpenCV Haar Cascade-оор нүүр илрүүлэх (эргүүлэлт)
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return [(x, y, w, h) for x, y, w, h in faces]


def extract_face_region(image, bbox, expansion=0.2):
    """
    Bounding box-ээс нүүрний бүс авах, дараа нь сунгах
    expansion: 0.2 нь 20% нэмж авна
    """
    x, y, w, h = bbox
    
    # Expansion
    exp_w = int(w * expansion / 2)
    exp_h = int(h * expansion / 2)
    
    x1 = max(0, x - exp_w)
    y1 = max(0, y - exp_h)
    x2 = min(image.shape[1], x + w + exp_w)
    y2 = min(image.shape[0], y + h + exp_h)
    
    face_region = image[y1:y2, x1:x2]
    return face_region, (x1, y1)


def denormalize_landmarks(landmarks_normalized, img_height=218, img_width=178):
    """Нормалчлагдсан координатаас анхны координатаар буцаах"""
    landmarks = landmarks_normalized.copy()
    landmarks[0::2] = landmarks[0::2] * img_width
    landmarks[1::2] = landmarks[1::2] * img_height
    return landmarks


def pipeline_on_image(image_path, device, model):
    """
    Бүрэн pipeline: Face detection → Landmark detection
    """
    print(f"\n{'='*70}")
    print(f"📷 Зураг: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # Зураг ачаалах
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Зураг ачаалахад алдаа: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"📏 Зургийн хэмжээ: {w}x{h}")
    
    # 1️⃣ FACE DETECTION
    print(f"\n🔍 Нүүр илрүүлж байна...")
    face_detector = "OpenCV Haar Cascade"
    print(f"   Ашиглаж байна: {face_detector}")
    
    faces = detect_faces_opencv(image)
    
    if not faces:
        print(f"❌ Нүүр олдсонгүй!")
        return None
    
    print(f"✓ {len(faces)} нүүр илрүүлэгдэв")
    
    # 2️⃣ LANDMARK DETECTION
    print(f"\n🎯 Landmark детектшн хийж байна...")
    
    all_results = []
    
    for face_idx, bbox in enumerate(faces, 1):
        x, y, width, height = bbox
        
        # Нүүрний бүс авах
        face_region, offset = extract_face_region(image, bbox, expansion=0.1)
        
        # Tensor-ээр хөрвүүлэх
        face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        transform = Compose([
            Resize((128, 128)),
            ToTensor(),
        ])
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Landmark таамаглал
        with torch.no_grad():
            landmarks_norm = model(face_tensor).cpu().numpy()[0]
        
        # Анхны координатаар буцаах
        face_h, face_w = face_region.shape[:2]
        landmarks = denormalize_landmarks(landmarks_norm, img_height=face_h, img_width=face_w)
        
        # Анхны зургайн координатаар өөрчлөх
        landmarks = landmarks.reshape(-1)  # Reshape to 1D if needed
        landmarks[0::2] += offset[0]  # x координат
        landmarks[1::2] += offset[1]  # y координат
        landmarks = landmarks.reshape(-1, 2)  # Reshape back to (5, 2)
        
        all_results.append({
            'face_id': face_idx,
            'bbox': bbox,
            'offset': offset,
            'landmarks': landmarks,
            'landmarks_norm': landmarks_norm
        })
        
        print(f"\n   ✓ Нүүр #{face_idx}")
        print(f"      Байрлал: ({x}, {y}) - ({x+width}, {y+height})")
        landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
        
        # Reshape landmarks to (5, 2) for proper indexing
        landmarks_reshaped = landmarks.reshape(-1, 2) if landmarks.ndim == 1 else landmarks
        
        for i, name in enumerate(landmark_names):
            lx = landmarks_reshaped[i, 0] if landmarks_reshaped.ndim == 2 else landmarks[i*2]
            ly = landmarks_reshaped[i, 1] if landmarks_reshaped.ndim == 2 else landmarks[i*2+1]
            print(f"      {name:15s}: ({lx:.1f}, {ly:.1f})")
    
    print(f"\n{'='*70}")
    print(f"✅ Pipeline дүүслэв!")
    print(f"{'='*70}")
    
    return image_rgb, all_results


def visualize_results(image_rgb, results):
    """Үр дүнгийг графикаар зурах"""
    image_copy = image_rgb.copy()
    
    landmark_names = ['LE', 'RE', 'N', 'LM', 'RM']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
    
    for result in results:
        x, y, w, h = result['bbox']
        
        # Nүүрний замиа зурах
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(image_copy, f"Face #{result['face_id']}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Landmark цэгүүдийг зурах
        landmarks = result['landmarks']
        
        # Reshape landmarks if it's 1D
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        
        for i, (name, color) in enumerate(zip(landmark_names, colors)):
            lx = int(landmarks[i, 0])
            ly = int(landmarks[i, 1])
            cv2.circle(image_copy, (lx, ly), 5, color, -1)
            cv2.circle(image_copy, (lx, ly), 5, (255, 255, 255), 1)
            cv2.putText(image_copy, name, (lx+10, ly-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return image_copy


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔗 TWO-STAGE FACE + LANDMARK DETECTION PIPELINE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Төхөөрөмж: {device}")
    
    # Моделийг ачаалах
    print("\n📦 Моделийг ачаалаж байна...")
    checkpoint_path = './checkpoints/resnet9_landmark_epoch3_loss0.0000.pt'
    
    if not os.path.exists(checkpoint_path):
        checkpoints = [f for f in os.listdir('./checkpoints') if f.endswith('.pt')]
        if checkpoints:
            checkpoint_path = f"./checkpoints/{checkpoints[-1]}"
    
    model = load_trained_model(checkpoint_path, device)
    print(f"✓ Модель ачаалагдлаа")
    
    # Test images
    test_images = [
        './test.png',
        '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba/000001.jpg',
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            result = pipeline_on_image(image_path, device, model)
            
            if result:
                image_rgb, all_results = result
                
                # Дүрслэх
                image_visualized = visualize_results(image_rgb, all_results)
                
                # Хадгалах
                output_path = f"pipeline_result_{os.path.basename(image_path)}"
                image_bgr = cv2.cvtColor(image_visualized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, image_bgr)
                print(f"\n💾 Үр дүн хадгалагдлаа: {output_path}")
                
                # Үзүүлэх
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.imshow(image_visualized)
                ax.set_title(f"Face + Landmark Detection: {os.path.basename(image_path)}")
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"pipeline_viz_{os.path.basename(image_path)}.png", dpi=150, bbox_inches='tight')
                plt.show()
    
    print(f"\n{'='*70}\n")
