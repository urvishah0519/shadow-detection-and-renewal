import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ISTDDataset(Dataset):
    def __init__(self, root_dir, phase='train'):
        self.root_dir = os.path.join(root_dir, phase)
        self.image_files = [f for f in os.listdir(os.path.join(self.root_dir, 'A')) if f.endswith('.png')]
        
        # Image transforms (for both shadow and shadow-free images)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Mask transform (no normalization needed for binary masks)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        shadow_img = Image.open(os.path.join(self.root_dir, 'A', img_name)).convert('RGB')
        shadow_mask = Image.open(os.path.join(self.root_dir, 'B', img_name)).convert('L')
        shadow_free = Image.open(os.path.join(self.root_dir, 'C', img_name)).convert('RGB')
        
        return {
            'shadow_img': self.image_transform(shadow_img),
            'shadow_mask': self.mask_transform(shadow_mask),
            'shadow_free': self.image_transform(shadow_free),
            'name': img_name
        }

def load_data(data_path, phase='train', batch_size=4, shuffle=True):
    dataset = ISTDDataset(data_path, phase=phase)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader