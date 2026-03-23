import torch
import torch.nn as nn
import numpy as np


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResNet9Landmark(nn.Module):
    """
    ResNet-9 — CelebA landmark regression
    Оролт : (B, 3, 128, 128)
    Гаралт: (B, 10)  →  5 цэгийн x,y координат
             [left_eye_x, left_eye_y,
              right_eye_x, right_eye_y,
              nose_x, nose_y,
              left_mouth_x, left_mouth_y,
              right_mouth_x, right_mouth_y]
    """
    def __init__(self, in_channels=3, num_landmarks=5):
        super().__init__()

        out_coords = num_landmarks * 2  # x, y тул 2 дахин

        self.conv1 = ConvBlock(in_channels, 64)           # 128 → 128
        self.conv2 = ConvBlock(64, 128, pool=True)        # 128 → 64
        self.res1  = ResidualBlock(128, 128)

        self.conv3 = ConvBlock(128, 256, pool=True)       # 64 → 32
        self.res2  = ResidualBlock(256, 256)

        self.conv4 = ConvBlock(256, 512, pool=True)       # 32 → 16

        # 16×16 → AdaptiveAvgPool → 1×1
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # MaxPool2d(4) → AdaptiveAvgPool: хэмжээнээс үл хамаарна
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_coords),
            nn.Sigmoid(),              # координатыг [0, 1] хүрээнд хадгална
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.regressor(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device, log_file=None):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        images = data['image_tr'].float().to(device)
        landmarks = data['scaled_landmarks'].float().to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, landmarks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            msg = f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}'
            print(msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(msg + '\n')
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, log_file=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            images = data['image_tr'].float().to(device)
            landmarks = data['scaled_landmarks'].float().to(device)
            predictions = model(images)
            loss = criterion(predictions, landmarks)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


if __name__ == "__main__":
    # Test model
    model_test = ResNet9Landmark(in_channels=3, num_landmarks=5)
    dummy = torch.randn(4, 3, 128, 128)
    out = model_test(dummy)
    print("Output shape:", out.shape)   # (4, 10)
    print("Value range: ", out.min().item(), "~", out.max().item())  # 0 ~ 1

    total = sum(p.numel() for p in model_test.parameters())
    print(f"Parameters: {total:,}\n")

    # Training setup
    import time
    import os
    from torch.utils.data import Dataset, DataLoader
    from skimage import io
    import random
    
    # Create directories for logs and checkpoints
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Log file path
    log_file = f'logs/training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    
    # Dataset class
    transform = torch.nn.Identity()  # Will be loaded as tensors
    
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
            
            # Convert image to tensor (resize to 128×128)
            from torchvision.transforms import Resize, ToTensor, Compose
            from PIL import Image
            
            # Convert numpy array to PIL Image first
            image_pil = Image.fromarray(image)
            
            transform_img = Compose([
                Resize((128, 128)),
                ToTensor(),
            ])
            image_tr = transform_img(image_pil)
            
            sample = {'image_tr': image_tr, 'scaled_landmarks': scaled_landmarks}
            return sample
    
    # Dataset paths (update these to your paths)
    full_dataset_root = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba'
    dataset_landmarks = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/landmark/list_landmarks_align_celeba.txt'
    
    print("Loading dataset...")
    full_dataset = FaceLandmarksDataset(dataset_landmarks, full_dataset_root, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train set: {len(train_set)}, Val set: {len(test_set)}\n")
    
    # Estimate training time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("Estimating training time...")
    print("="*60)
    
    # Measure one batch processing time
    model_temp = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model_temp.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_temp.parameters(), lr=0.001)
    
    start_time_estimate = time.time()
    for i, data in enumerate(train_loader):
        images = data['image_tr'].float().to(device)
        landmarks = data['scaled_landmarks'].float().to(device)
        predictions = model_temp(images)
        
        loss = criterion(predictions, landmarks)
        loss.backward()
        optimizer.step()
        if i >= 4:  # measure 5 batches
            break
    
    time_per_batch = (time.time() - start_time_estimate) / 5
    num_epochs = 3
    total_batches = len(train_loader) * num_epochs
    estimated_time = total_batches * time_per_batch
    
    print(f"⏱️  Time per batch: {time_per_batch:.2f}s")
    print(f"📊 Total batches: {total_batches}")
    print(f"⏳ Estimated training time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
    print(f"📁 Log file: {log_file}")
    print(f"💾 Model folder: ./checkpoints/")
    print("="*60 + "\n")
    
    del model_temp, optimizer
    
    # Training
    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Write header to log
    with open(log_file, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Estimated time: {estimated_time/60:.1f} minutes\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write("="*60 + "\n\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, log_file)
        val_loss = validate(model, val_loader, criterion, device, log_file)
        scheduler.step()
        
        msg = f'\nEpoch {epoch}/{num_epochs}'
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
        
        msg = f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}'
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = f'checkpoints/resnet9_landmark_epoch{epoch}_loss{val_loss:.4f}.pt'
            torch.save(model.state_dict(), model_path)
            msg = f'✓ Model saved to {model_path}'
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                msg = f'Early stopping at epoch {epoch}'
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + '\n')
                break
    
    elapsed = time.time() - start_time
    msg = f'\nTraining complete in {elapsed/60:.1f} minutes ({elapsed:.0f}s)'
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    
    msg = f'Best validation loss: {best_val_loss:.6f}'
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    
    print(f'\n📝 Log file: {log_file}\n')