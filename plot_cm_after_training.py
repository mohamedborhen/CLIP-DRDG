#!/usr/bin/env python3
"""
Simple script to plot confusion matrices after training is complete.
This script loads a trained model and creates confusion matrices for all environments.

Usage:
    python plot_cm_after_training.py --model_path ./output/model.pkl
    python plot_cm_after_training.py --model_path ./output/model_best_f1_w_step_2750.pkl --test_env 1
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from domainbed import datasets, algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrices from trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pkl file)')
    parser.add_argument('--test_env', type=int, default=0,
                        help='Test environment (default: 0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as model directory)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Data directory containing the DR dataset')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), 'confusion_matrices_final')
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR Model file not found: {args.model_path}")
        return
    
    print(f"INFO Loading model from: {args.model_path}")
    print(f"INFO Test environment: {args.test_env}")
    print(f"INFO Output directory: {args.output_dir}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO Using device: {device}")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        model_hparams = checkpoint['model_hparams']
        
        # Load dataset
        dataset = datasets.DR(args.data_dir, [args.test_env], model_hparams)
        
        # Compute weights for balance (needed for algorithm initialization)
        train_loader = FastDataLoader(
            dataset=dataset.datasets[0],  # Use first environment for weights
            batch_size=64,
            num_workers=0
        )
        
        # Calculate class weights
        # First find the actual number of classes in the data
        all_labels = []
        for x, y in train_loader:
            all_labels.extend(y.tolist())
        
        actual_num_classes = max(all_labels) + 1
        class_counts = torch.zeros(actual_num_classes)
        
        # Reset the loader and count again
        train_loader = FastDataLoader(
            dataset=dataset.datasets[0],
            batch_size=64,
            num_workers=0
        )
        
        for x, y in train_loader:
            for class_idx in y:
                class_counts[class_idx] += 1
        
        weights_for_balance = 1.0 / (class_counts + 1e-8)  # Add small epsilon to avoid division by zero
        weights_for_balance = weights_for_balance / weights_for_balance.sum()
        
        # Load algorithm
        algorithm_class = algorithms.get_algorithm_class(checkpoint['args']['algorithm'])
        algorithm = algorithm_class(
            checkpoint['model_input_shape'],
            checkpoint['model_num_classes'],
            checkpoint['model_num_domains'],
            model_hparams,
            weights_for_balance
        )
        algorithm.load_state_dict(checkpoint['model_dict'])
        algorithm.to(device)
        algorithm.eval()
        
        print("SUCCESS Model loaded successfully!")
        
        # Create data splits
        in_splits = []
        out_splits = []
        
        for env_i, env in enumerate(dataset):
            out, in_ = misc.split_dataset(env, int(len(env) * 0.2), misc.seed_hash(0, env_i))
            
            if model_hparams.get('class_balanced', False):
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
            else:
                in_weights, out_weights = None, None
                
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
        
        # Create evaluation loaders
        eval_loaders = [FastDataLoader(dataset=env, batch_size=64, num_workers=0) 
                       for env, _ in (in_splits + out_splits)]
        eval_weights = [weights for _, weights in (in_splits + out_splits)]
        eval_loader_names = [f'env{i}_in' for i in range(len(in_splits))] + \
                           [f'env{i}_out' for i in range(len(out_splits))]
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"\n📊 Creating confusion matrices for {len(eval_loaders)} environments...")
        
        # Plot confusion matrices for each environment
        for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
            print(f"  - Processing {name}...")
            
            cm = misc.plot_confusion_matrix_custom(
                network=algorithm,
                loader=loader,
                weights=weights,
                device=device,
                class_names=["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
                output_dir=args.output_dir,
                step="final",
                env_name=name
            )
            
            print(f"    ✓ Confusion matrix saved (shape: {cm.shape}, total predictions: {cm.sum()})")
        
        # Create summary plot
        print("\n📊 Creating summary plot...")
        misc.create_training_summary_plots(
            algorithm=algorithm,
            eval_loaders=eval_loaders,
            eval_loader_names=eval_loader_names,
            eval_weights=eval_weights,
            device=device,
            output_dir=args.output_dir,
            step="final"
        )
        
        print(f"\nSUCCESS All confusion matrices saved to: {args.output_dir}")
        print(f"🔍 Files created:")
        for name in eval_loader_names:
            print(f"  - confusion_matrix_{name}_step_final.png")
        print(f"  - confusion_matrices_summary_step_final.png")
        
    except Exception as e:
        print(f"ERROR Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
