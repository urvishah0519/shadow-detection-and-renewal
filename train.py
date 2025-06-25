import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shadow_desdnet import ShadowDESDNet
from models.shadow_removal import ShadowRemoval
from models.utils import load_data

class ShadowTrainer:
    def __init__(self, data_path, batch_size=4, epochs=20, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize models
        self.detector = ShadowDESDNet().to(self.device)
        self.remover = ShadowRemoval().to(self.device)
        
        # Optimizers
        self.opt_det = optim.Adam(self.detector.parameters(), lr=lr, weight_decay=1e-5)
        self.opt_rem = optim.Adam(self.remover.parameters(), lr=lr, weight_decay=1e-5)
        
        # Loss functions
        self.criterion_det = nn.BCEWithLogitsLoss()
        self.criterion_rem = nn.L1Loss()
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Data loaders
        self.train_loader = load_data(data_path, 'train', batch_size)
        self.val_loader = load_data(data_path, 'test', max(1, batch_size//2), shuffle=False)
        
        # Logging
        os.makedirs('logs/train', exist_ok=True)
        self.writer = SummaryWriter('logs/train')

    def train_epoch(self, epoch):
        self.detector.train()
        self.remover.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            # Clear cache periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            shadow_img = batch['shadow_img'].to(self.device, non_blocking=True)
            shadow_mask = batch['shadow_mask'].to(self.device, non_blocking=True)
            shadow_free = batch['shadow_free'].to(self.device, non_blocking=True)
            
            with autocast():
                # Forward passes
                pred_mask = self.detector(shadow_img)
                pred_free = self.remover(shadow_img, pred_mask.detach())
                
                # Loss calculation
                loss_det = self.criterion_det(pred_mask, shadow_mask)
                loss_rem = self.criterion_rem(pred_free, shadow_free)
                loss = loss_det + loss_rem
            
            # Backpropagation
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % 2 == 0:
                self.scaler.step(self.opt_det)
                self.scaler.step(self.opt_rem)
                self.scaler.update()
                self.opt_det.zero_grad()
                self.opt_rem.zero_grad()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)
            
            # Memory cleanup
            del shadow_img, shadow_mask, shadow_free, pred_mask, pred_free
            torch.cuda.empty_cache()
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.detector.eval()
        self.remover.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                shadow_img = batch['shadow_img'].to(self.device)
                shadow_mask = batch['shadow_mask'].to(self.device)
                shadow_free = batch['shadow_free'].to(self.device)
                
                pred_mask = self.detector(shadow_img)
                pred_free = self.remover(shadow_img, pred_mask)
                
                loss = self.criterion_det(pred_mask, shadow_mask) + \
                       self.criterion_rem(pred_free, shadow_free)
                total_loss += loss.item()
                
                # Cleanup
                del shadow_img, shadow_mask, shadow_free, pred_mask, pred_free
                torch.cuda.empty_cache()
                
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'detector_state': self.detector.state_dict(),
            'remover_state': self.remover.state_dict(),
            'opt_det_state': self.opt_det.state_dict(),
            'opt_rem_state': self.opt_rem.state_dict()
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(state, f'checkpoints/checkpoint_epoch{epoch}.pth')
        
        if is_best:
            torch.save(state, 'checkpoints/best_model.pth')

    def train(self):
        best_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            self.save_checkpoint(epoch, is_best)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/ISTD_dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    trainer = ShadowTrainer(**vars(args))
    trainer.train()

