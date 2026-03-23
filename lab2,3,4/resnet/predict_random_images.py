"""
Trained ResNet9Landmark Model - Random Image Prediction
Хадгалсан моделоор датасетээс random зураг дээр таамаглал хийнэ
"""

import torch
import torch.nn as nn
import numpy as np
import os
import random
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
        
        # Normalize to [0, 1]
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] = scaled_landmarks[:, 0] / 178
        scaled_landmarks[:, 1] = scaled_landmarks[:, 1] / 218
        scaled_landmarks = scaled_landmarks.reshape(-1)
        
        # Convert image to tensor
        image_pil = Image.fromarray(image)
        
        transform_img = Compose([
            Resize((128, 128)),
            ToTensor(),
        ])
        image_tr = transform_img(image_pil)
        
        sample = {'image_tr': image_tr, 'scaled_landmarks': scaled_landmarks}
        return sample


def load_trained_model(checkpoint_path, device):
    """
    Хадгалсан моделийг ачаалах
    """
    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def denormalize_landmarks(landmarks_normalized, img_height=218, img_width=178):
    """
    Нормалчлагдсан координатаас анхны координатаар буцаах
    landmarks_normalized: (10,) array with values in [0, 1]
    """
    landmarks = landmarks_normalized.copy()
    landmarks[0::2] = landmarks[0::2] * img_width    # x координат
    landmarks[1::2] = landmarks[1::2] * img_height   # y координат
    return landmarks


def plot_landmarks_on_image(image_array, landmarks_normalized, title="Face Landmarks"):
    """
    Зурагтай дээр facial landmarks цэгүүдийг зурах
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Зураг үзүүлэх
    ax.imshow(image_array)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Анхны зургийн хэмжээ
    img_h, img_w = image_array.shape[:2]
    
    # Нормалчлагдсан координатаас анхны координатыг авах
    landmarks = denormalize_landmarks(landmarks_normalized, img_height=img_h, img_width=img_w)
    
    # 5 цэг
    landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (name, color) in enumerate(zip(landmark_names, colors)):
        x = landmarks[i * 2]
        y = landmarks[i * 2 + 1]
        
        # Цэг зурах
        circle = patches.Circle((x, y), radius=5, color=color, fill=True, alpha=0.6)
        ax.add_patch(circle)
        
        # Цэгийн нэр
        ax.text(x + 10, y - 10, name, fontsize=10, color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def predict_on_random_image(num_images=5):
    """
    Датасетээс random зураг сонгож таамаглал хийнэ
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Төхөөрөмж: {device}")
    print("="*70)
    
    # Моделийг ачаалах
    print("\n📦 Моделийг ачаалаж байна...")
    checkpoint_path = './checkpoints/resnet9_landmark_epoch3_loss0.0000.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint файл олдсонгүй: {checkpoint_path}")
        # Бүх checkpoint файлүүдийг харуулах
        checkpoints = [f for f in os.listdir('./checkpoints') if f.endswith('.pt')]
        if checkpoints:
            print(f"💡 Боломжит checkpoints: {checkpoints}")
            checkpoint_path = f"./checkpoints/{checkpoints[-1]}"  # Хамгийн сүүлийнхийг авах
            print(f"✓ Ашиглаж байна: {checkpoint_path}")
        else:
            print("❌ Нэг ч checkpoint файл олдсонгүй!")
            return
    
    model = load_trained_model(checkpoint_path, device)
    print(f"✓ Модель ачаалагдлаа: {checkpoint_path}")
    print("="*70)
    
    # Датасетийг ачаалах
    print("\n📂 Датасет ачаалаж байна...")
    full_dataset_root = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba'
    dataset_landmarks = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/landmark/list_landmarks_align_celeba.txt'
    
    try:
        dataset = FaceLandmarksDataset(dataset_landmarks, full_dataset_root)
        print(f"✓ Датасет ачаалагдлаа: {len(dataset)} зураг")
    except Exception as e:
        print(f"❌ Датасет ачаалахад алдаа: {e}")
        return
    
    print("="*70)
    
    # Random зураг сонгож таамаглал хийнэ
    print(f"\n🎲 {num_images} random зургийн хувьд таамаглал хийж байна...")
    print("="*70)
    
    random_indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    predicted_landmarks_list = []
    images_list = []
    
    for idx_in_list, random_idx in enumerate(random_indices, 1):
        sample = dataset[random_idx]
        image_tr = sample['image_tr'].unsqueeze(0).float().to(device)  # (1, 3, 128, 128)
        
        with torch.no_grad():
            predicted_landmarks = model(image_tr)  # (1, 10)
        
        predicted_landmarks = predicted_landmarks.cpu().numpy()[0]  # (10,)
        
        # Анхны зурагийг ачаалах (нормалчлагдаагүй)
        img_name = os.path.join(full_dataset_root, dataset.landmarks_frame[random_idx][0])
        original_image = io.imread(img_name)
        
        predicted_landmarks_list.append(predicted_landmarks)
        images_list.append(original_image)
        
        print(f"\n🖼️  Зураг {idx_in_list}/{len(random_indices)}")
        print(f"   Index: {random_idx}")
        print(f"   Зураг: {dataset.landmarks_frame[random_idx][0]}")
        print(f"   Таамаглагдсан координатаууд (нормалчлагдсан):")
        for i, name in enumerate(['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']):
            x, y = predicted_landmarks[i*2], predicted_landmarks[i*2+1]
            print(f"      {name:15s}: x={x:.4f}, y={y:.4f}")
    
    print("\n" + "="*70)
    print(f"✅ Таамаглал дүүслэв!")
    
    # Графикаар үзүүлэх
    print("\n📊 Зургуудыг дүрслэх хэрэгтүүлж байна...")
    
    if len(images_list) == 1:
        fig = plot_landmarks_on_image(images_list[0], predicted_landmarks_list[0], 
                                     f"Prediction - Image {random_indices[0]}")
        plt.savefig('single_prediction.png', dpi=150, bbox_inches='tight')
        print("✓ Файлдаа хадгалагдлаа: single_prediction.png")
    else:
        fig, axes = plt.subplots(2, (len(images_list) + 1) // 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (image, landmarks) in enumerate(zip(images_list, predicted_landmarks_list)):
            ax = axes[idx]
            ax.imshow(image)
            
            # Координатаар буцаах
            img_h, img_w = image.shape[:2]
            landmarks_denorm = denormalize_landmarks(landmarks, img_height=img_h, img_width=img_w)
            
            # Цэгүүдийг зурах
            landmark_names = ['LE', 'RE', 'N', 'LM', 'RM']
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (name, color) in enumerate(zip(landmark_names, colors)):
                x = landmarks_denorm[i * 2]
                y = landmarks_denorm[i * 2 + 1]
                circle = patches.Circle((x, y), radius=4, color=color, fill=True, alpha=0.7)
                ax.add_patch(circle)
                ax.text(x + 5, y - 5, name, fontsize=8, color=color, fontweight='bold')
            
            ax.set_title(f"Image {random_indices[idx]}", fontsize=10)
            ax.axis('off')
        
        # Хэрэглэгдээгүй субплотуудыг нуух
        for idx in range(len(images_list), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('random_predictions.png', dpi=150, bbox_inches='tight')
        print(f"✓ Файлдаа хадгалагдлаа: random_predictions.png")
    
    try:
        plt.show()
    except:
        pass
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔮 FACIAL LANDMARK PREDICTION ON RANDOM IMAGES")
    print("="*70 + "\n")
    
    predict_on_random_image(num_images=6)
