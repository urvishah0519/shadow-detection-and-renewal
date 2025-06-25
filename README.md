# Shadow_detection_and_removal
# Shadow Detection and Removal System
## OUTPUT
![shadow_detected](https://github.com/user-attachments/assets/67ea42e3-06ae-4ae2-a8ac-e0a6940cc24a)
![shadow_mask](https://github.com/user-attachments/assets/65c287ca-3d4c-4a5f-81fb-9d2ffb9240e6)
![shadow_removed](https://github.com/user-attachments/assets/8a9fc392-7b33-414d-9ca8-6bac5c59e45d)




A deep learning system for detecting and removing shadows from images, built with PyTorch. The project consists of two main components:
1. Shadow Detection Network (ShadowDESDNet) - Identifies shadow regions in images
2. Shadow Removal Network - Removes shadows while preserving image content

## Features

- Dual-network architecture for end-to-end shadow processing
- Dynamic attention mechanism for improved shadow detection
- Memory-efficient design for practical usage
- Comprehensive training and evaluation pipeline
- Pretrained ResNeXt50 backbone for feature extraction

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- CUDA 11.3+ (for GPU acceleration)
- Other dependencies in `requirements.txt`

## Installation

bash
git clone https://github.com/yourusername/shadow-removal.git
cd shadow-removal
pip install -r requirements.txt

## DATASET PREPARATION: 
data/ISTD_dataset/
    ├── train/
    │   ├── A/  # Shadow images
    │   ├── B/  # Shadow masks
    │   └── C/  # Shadow-free images
    └── test/
        ├── A/
        ├── B/
        └── C/
        
## TRAINING: 
python main.py train \
    --data_path data/ISTD_dataset \
    --batch_size 8 \
    --epochs 20 \
    --lr 1e-4
    
## EVALUATION:
python main.py evaluate \
    --model_path checkpoints/best_model.pth \
    --data_path data/ISTD_dataset \
    --batch_size 8
    
## Inference on Single Image:
python predict.py --image_path path/to/your/image.jpg

## PROJECT STRUCTURE:
├── models/
│   ├── shadow_desdnet.py       # Shadow detection network
│   ├── shadow_removal.py       # Shadow removal network
│   └── utils.py                # Dataset and loader utilities
├── train.py                    # Training script
├── evaluate.py                 # Basic evaluation
├── advanced_evaluate.py        # Metrics evaluation
├── predict.py                  # Inference script
├── main.py                     # Main CLI interface
├── checkpoints/                # Model weights
├── data/                       # Dataset
├── results/                    # Output visualizations
└── logs/                       # Training logs

## RESULTS:
The system achieves the following metrics on the ISTD test set:

Metric	Score
Accuracy	0.95
Precision	0.91
Recall	0.89
F1 Score	0.90

## Pretrained Models:
Download pretrained models from Releases and place them in the checkpoints/ director

## Contributing:
Contributions are welcome! Please open an issue or submit a pull request.

## License:
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation: 
If you use this code in your research, please cite:

bibtex
@misc{shadowremoval2023,
  author = Manas Dutt,
  title = {Shadow Detection and Removal System},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/shadow-removal}}
}

## Acknowledgments:
ISTD dataset providers

PyTorch community

Original ResNeXt authors


 Key Features to Highlight:

1. Clear Project Description: Immediately explains what the project does
2. Visual Example: Shows sample results upfront
3. Structured Installation/Usage: Step-by-step instructions
4. Dataset Requirements: Explains expected data structure
5. Comprehensive CLI: Documents all command options
6. Performance Metrics: Shows quantitative results
7. Academic Citation: Ready for research use
8. Clean Project Structure: Helps contributors navigate




