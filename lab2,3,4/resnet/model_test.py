import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, ToTensor, Compose
from skimage import io
import glob

# Dataset class
class FaceLandmarksDataset:
    def __init__(self, csv_file, img_root_dir):
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
        
        # Convert to PIL and resize
        image_pil = Image.fromarray(image)
        transform_img = Compose([
            Resize((128, 128)),
            ToTensor(),
        ])
        image_tr = transform_img(image_pil)
        
        sample = {'image_tr': image_tr, 'image_original': image, 'scaled_landmarks': scaled_landmarks, 'landmarks': landmarks}
        return sample


# Load dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


# Model definition (copy from resnet9.py)
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
    def __init__(self, in_channels=3, num_landmarks=5):
        super().__init__()
        out_coords = num_landmarks * 2
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1  = ResidualBlock(128, 128)
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.res2  = ResidualBlock(256, 256)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_coords),
            nn.Sigmoid(),
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


def test_model(model_path, dataset_path, landmarks_path, device='cuda'):
    """Test the model on validation set"""
    
    print("="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Load model
    model = ResNet9Landmark(in_channels=3, num_landmarks=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    
    # Load dataset
    dataset = FaceLandmarksDataset(landmarks_path, dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)
    print(f"✓ Test set loaded: {len(test_set)} samples\n")
    
    # Test metrics
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    predictions_list = []
    targets_list = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images = data['image_tr'].float().to(device)
            landmarks = data['scaled_landmarks'].float().to(device)
            
            predictions = model(images)
            
            # Calculate metrics
            mse = F.mse_loss(predictions, landmarks, reduction='mean').item()
            mae = F.l1_loss(predictions, landmarks, reduction='mean').item()
            
            total_mse += mse * images.shape[0]
            total_mae += mae * images.shape[0]
            total_samples += images.shape[0]
            
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(landmarks.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}, MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Average MSE Loss: {avg_mse:.6f}")
    print(f"Average MAE Loss: {avg_mae:.6f}")
    print(f"RMSE: {np.sqrt(avg_mse):.6f}")
    print("="*60 + "\n")
    
    # Visualize predictions
    print("Generating visualization...")
    visualize_predictions(model, test_loader, device, num_samples=8)


def visualize_predictions(model, test_loader, device, num_samples=8):
    """Visualize model predictions"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    sample_count = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            images = data['image_tr'].float().to(device)
            images_orig = data['image_original']
            landmarks_true = data['landmarks']
            
            predictions = model(images).cpu().numpy()
            
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Get image
                img = images_orig[i]
                
                # Convert predicted landmarks from normalized coords to pixel coords
                pred_lm = predictions[i].reshape(-1, 2)
                pred_lm[:, 0] = pred_lm[:, 0] * 178
                pred_lm[:, 1] = pred_lm[:, 1] * 218
                
                # True landmarks
                true_lm = landmarks_true[i].numpy()
                
                # Plot
                ax = axes[sample_count]
                ax.imshow(img)
                ax.scatter(true_lm[:, 0], true_lm[:, 1], s=50, marker='o', c='green', label='True', alpha=0.7)
                ax.scatter(pred_lm[:, 0], pred_lm[:, 1], s=50, marker='x', c='red', label='Predicted', alpha=0.7)
                ax.set_title(f'Sample {sample_count + 1}')
                ax.legend()
                ax.axis('off')
                
                sample_count += 1
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: predictions_visualization.png\n")
    plt.show()


if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Find latest model checkpoint
    checkpoint_dir = 'checkpoints'
    model_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    
    if not model_files:
        print(f"❌ No model files found in {checkpoint_dir}")
        exit(1)
    
    # Get latest model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Using latest model: {latest_model}\n")
    
    # Dataset paths
    dataset_root = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/img_align_celeba'
    landmarks_file = '/home/tr1bo/Documents/1. School/1. 3B/DeepLearning/lab2,3,4/data/landmark/list_landmarks_align_celeba.txt'
    
    # Test
    test_model(latest_model, dataset_root, landmarks_file, device)
