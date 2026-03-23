"""
Single Batch Overfit Test for ResNet9Landmark
Проверяет, что модель может переобучаться на одном батче
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
import os
import time
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


def single_batch_overfit_test(num_iterations=1000, batch_size=8):
    """
    Сургадаг нэг batch-г дахин дахин сургаж, 
    алдаа 0 рүү ойртож байгаа эсэхийг шалгana
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Төхөөрөмж: {device}")
    print("="*70)
    
    # Датасет ачаалаад нэг batch авна
    print("\n📂 Датасет ачаалаж байна...")
    full_dataset_root = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba'
    dataset_landmarks = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/landmark/list_landmarks_align_celeba.txt'
    
    try:
        full_dataset = FaceLandmarksDataset(dataset_landmarks, full_dataset_root)
        data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Нэг batch авна
        single_batch = next(iter(data_loader))
        print(f"✓ Batch авлаа: {single_batch['image_tr'].shape}")
    except Exception as e:
        print(f"⚠️  Датасет ачаалахад алдаа: {e}")
        print("💡 Сургалтын өгөгдөл байхгүй, санамсаргүй датасет ашиглана...")
        
        # Санамсаргүй датасет ашиглана
        images = torch.randn(batch_size, 3, 128, 128)
        landmarks = torch.rand(batch_size, 10)
        single_batch = {
            'image_tr': images,
            'scaled_landmarks': landmarks
        }
    
    # Моделийг үүсгэнэ
    print("\n🧠 Модель үүсгэж байна...")
    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✓ Модель үүсэгдэв")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Параметрүүд: {total_params:,}")
    print("="*70)
    
    print(f"\n🚀 Нэг batch-д {num_iterations} удаа сургалт эхэлнэ...")
    print("="*70)
    
    # Batch-д овerfит хийнэ
    images = single_batch['image_tr'].float().to(device)
    landmarks = single_batch['scaled_landmarks'].float().to(device)
    
    losses = []
    start_time = time.time()
    
    model.train()
    
    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, landmarks)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iteration % 100 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            print(f"Итерация {iteration:4d} | Алдаа: {loss.item():.8f} | ⏱️  {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    print("="*70)
    print(f"\n✅ Сургалт дүүслэв!")
    print(f"⏱️  Нийт хугацаа: {elapsed:.1f}s ({elapsed/num_iterations*1000:.2f}ms/iteration)")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"\n📈 Ахиц:")
    print(f"   Эхний алдаа:  {initial_loss:.8f}")
    print(f"   Сүүлийн алдаа: {final_loss:.8f}")
    print(f"   Буурсан хэмжээ: {initial_loss - final_loss:.8f} ({(1 - final_loss/initial_loss)*100:.1f}%)")
    
    # Overfit амжилттай болсон эсэхийг шалгана
    print("\n" + "="*70)
    if final_loss < 0.01:
        print("🎉 АМЖИЛТТАЙ! Модель нэг batch-д сайн overfitted-лөл!")
        print("   → Сүлжээ архитектур сайн ажилж байна ✓")
        print("   → Сургалт цикл зөв ажилж байна ✓")
    elif final_loss < 0.1:
        print("⚠️  Overfit хэсэгчилсэн. Алдаа хэвээр их байна.")
        print("   → Learning rate ихэсгэх хэрэгтэй байж магадшна")
    else:
        print("❌ Overfit амжилтгүй болсон. Алдаа ихэсч байна.")
        print("   → Модель эсвэл сургалт цикл хэрэгтэй байж магадшна")
    
    print("="*70 + "\n")
    
    return {
        'model': model,
        'losses': losses,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'elapsed_time': elapsed
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 SINGLE BATCH OVERFIT TEST")
    print("="*70 + "\n")
    
    results = single_batch_overfit_test(num_iterations=500, batch_size=8)
    
    # График үзүүллэх сонголт
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['losses'], linewidth=2)
        plt.xlabel('Итерация')
        plt.ylabel('MSE Алдаа')
        plt.title('Нэг Batch-д Overfitting')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(results['losses'][:50], linewidth=2, marker='o')
        plt.xlabel('Итерация')
        plt.ylabel('MSE Алдаа')
        plt.title('Эхний 50 Итерация (Нарийвчилсан)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('single_batch_overfit_test.png', dpi=150, bbox_inches='tight')
        print("📊 График файл хадгалагдлаа: single_batch_overfit_test.png\n")
        plt.show()
    except ImportError:
        print("💡 Matplotlib суулгаагүй байна. График үзүүлэх боломжгүй.\n")
