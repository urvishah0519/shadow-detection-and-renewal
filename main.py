import argparse
from train import ShadowTrainer
from evaluate import ShadowEvaluator

def main():
    parser = argparse.ArgumentParser(description='Shadow Detection and Removal System')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the shadow detection and removal models')
    train_parser.add_argument('--data_path', default='data/ISTD_dataset', help='Path to dataset directory')
    train_parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.set_defaults(func=lambda args: ShadowTrainer(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    ).train())

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--model_path', default='checkpoints/best_model.pth', 
                           help='Path to trained model checkpoint')
    eval_parser.add_argument('--data_path', default='data/ISTD_dataset', 
                           help='Path to dataset directory')
    eval_parser.add_argument('--batch_size', type=int, default=8, 
                           help='Batch size for evaluation')
    eval_parser.set_defaults(func=lambda args: ShadowEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size
    ).evaluate())

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()