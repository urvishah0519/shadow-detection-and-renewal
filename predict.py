import torch
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.shadow_desdnet import ShadowDESDNet
from models.shadow_removal import ShadowRemoval

def remove_shadow(image_path, model_path='checkpoints/best_model.pth'):
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = ShadowDESDNet().to(device)
    remover = ShadowRemoval().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    detector.load_state_dict(checkpoint['detector_state'])
    remover.load_state_dict(checkpoint['remover_state'])
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process
    with torch.no_grad():
        shadow_mask = detector(image_tensor)
        shadow_free = remover(image_tensor, shadow_mask)
    
    # Convert back to image
    shadow_free = (shadow_free.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
    result = transforms.ToPILImage()(shadow_free)
    
    # Save result
    output_path = os.path.splitext(image_path)[0] + '_shadowfree.png'
    result.save(output_path)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--model_path', default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    remove_shadow(args.image_path, args.model_path)