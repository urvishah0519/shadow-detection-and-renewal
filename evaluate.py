import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shadow_desdnet import ShadowDESDNet
from models.shadow_removal import ShadowRemoval
from models.utils import load_data

class ShadowEvaluator:
    def __init__(self, model_path, data_path, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.data_path = data_path
        
        # Load models from checkpoint
        self.load_models(model_path)
        
        # Setup data loader
        self.test_loader = load_data(
            data_path=data_path,
            phase='test',
            batch_size=batch_size,
            shuffle=False
        )
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs/eval', exist_ok=True)
        self.writer = SummaryWriter('logs/eval')

    def load_models(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.detector = ShadowDESDNet().to(self.device)
        self.remover = ShadowRemoval().to(self.device)
        
        self.detector.load_state_dict(checkpoint['detector_state'])
        self.remover.load_state_dict(checkpoint['remover_state'])
        
        self.detector.eval()
        self.remover.eval()

    def denormalize(self, tensor):
        """Convert normalized tensor back to image"""
        return tensor * 0.5 + 0.5

    def save_results(self, batch, pred_masks, pred_free, batch_idx):
        """Save visualization of results for all images in batch"""
        shadow_imgs = self.denormalize(batch['shadow_img'].cpu())
        shadow_masks = batch['shadow_mask'].cpu()
        shadow_free = self.denormalize(batch['shadow_free'].cpu())
        
        # Create directory for this batch
        batch_dir = f'results/batch_{batch_idx}'
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save each image individually
        for i in range(len(shadow_imgs)):
            # Create grid for this single image
            results = [
                shadow_imgs[i],                     # Input shadow image
                pred_masks[i].cpu().repeat(3,1,1),  # Predicted mask (as RGB)
                pred_free[i].cpu(),                 # Predicted shadow-free
                shadow_masks[i].repeat(3,1,1),      # Ground truth mask
                shadow_free[i]                      # Ground truth shadow-free
            ]
            
            grid = make_grid(results, nrow=5, padding=2, normalize=True)
            plt.figure(figsize=(20, 4))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            
            # Save with original image index if available, otherwise use batch position
            img_name = batch.get('img_name', [f'img_{i}'] * len(shadow_imgs))[i]
            plt.savefig(f'{batch_dir}/{img_name}.png', bbox_inches='tight')
            plt.close()

    def evaluate(self):
        total_loss = 0
        self.detector.eval()
        self.remover.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Move data to device
                shadow_img = batch['shadow_img'].to(self.device)
                shadow_mask = batch['shadow_mask'].to(self.device)
                shadow_free = batch['shadow_free'].to(self.device)
                
                # Forward pass
                pred_mask = self.detector(shadow_img)
                pred_free = self.remover(shadow_img, pred_mask)
                
                # Calculate loss using nn.functional
                loss = nn.functional.binary_cross_entropy(pred_mask, shadow_mask) + \
                       nn.functional.l1_loss(pred_free, shadow_free)
                total_loss += loss.item()
                
                # Save results for all images in this batch
                self.save_results(batch, pred_mask, pred_free, batch_idx)
        
        avg_loss = total_loss / len(self.test_loader)
        print(f"\nEvaluation Complete")
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/eval', avg_loss, 0)
        return avg_loss

if __name__ == "__main__":
    evaluator = ShadowEvaluator(
        model_path='checkpoints/best_model.pth',
        data_path='data/ISTD_dataset',
        batch_size=8
    )
    evaluator.evaluate()