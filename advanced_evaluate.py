import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shadow_desdnet import ShadowDESDNet
from models.shadow_removal import ShadowRemoval
from models.utils import load_data

class MetricEvaluator:
    def __init__(self, model_path='checkpoints/best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models(model_path)
        
    def load_models(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.detector = ShadowDESDNet().to(self.device)
        self.remover = ShadowRemoval().to(self.device)
        self.detector.load_state_dict(checkpoint['detector_state'])
        self.remover.load_state_dict(checkpoint['remover_state'])
        self.detector.eval()
        self.remover.eval()
    
    def calculate_metrics(self, y_true, y_pred_prob, threshold=0.5):
        # Convert probabilities to binary predictions
        y_pred = (y_pred_prob > threshold).astype(int)
        y_true = y_true.astype(int)
        
        if np.sum(y_true) == 0:  # Handle no-shadow cases
            return [1.0, 1.0, 1.0, 1.0]
        
        return [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        ]
    
    def evaluate(self, data_path='data/ISTD_dataset', batch_size=8):
        test_loader = load_data(data_path, 'test', batch_size, shuffle=False)
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                shadow_img = batch['shadow_img'].to(self.device)
                shadow_mask = batch['shadow_mask'].to(self.device)
                
                # Get predictions
                pred_mask_prob = self.detector(shadow_img)
                
                # Convert to numpy arrays
                true_mask = shadow_mask.cpu().numpy().flatten()
                pred_prob = pred_mask_prob.cpu().numpy().flatten()
                
                # Calculate metrics
                batch_metrics = self.calculate_metrics(true_mask, pred_prob)
                all_metrics.append(batch_metrics)
        
        # Calculate averages
        avg_metrics = np.mean(all_metrics, axis=0)
        return {
            'accuracy': avg_metrics[0],
            'precision': avg_metrics[1],
            'recall': avg_metrics[2],
            'f1': avg_metrics[3]
        }

def main():
    evaluator = MetricEvaluator()
    metrics = evaluator.evaluate()
    
    print("\nShadow Detection Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Visualize sample predictions
    sample_loader = load_data('data/ISTD_dataset', 'test', batch_size=3, shuffle=True)
    batch = next(iter(sample_loader))
    
    with torch.no_grad():
        shadow_img = batch['shadow_img'].to(evaluator.device)
        pred_mask = evaluator.detector(shadow_img)
        pred_free = evaluator.remover(shadow_img, pred_mask)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    for i in range(min(3, len(batch['shadow_img']))):  # Show up to 3 samples
        plt.subplot(3, 4, i*4+1)
        plt.imshow(batch['shadow_img'][i].permute(1,2,0).numpy()*0.5+0.5)
        plt.title("Input")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4+2)
        plt.imshow(pred_mask[i][0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title("Pred Mask")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4+3)
        plt.imshow(pred_free[i].permute(1,2,0).cpu().numpy()*0.5+0.5)
        plt.title("Shadow-Free")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4+4)
        plt.imshow(batch['shadow_mask'][i][0].cpu().numpy(), cmap='gray')
        plt.title("GT Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/metrics_evaluation.png', bbox_inches='tight')
    print("\nVisualization saved to 'results/metrics_evaluation.png'")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()