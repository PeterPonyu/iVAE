#!/usr/bin/env python

"""
Command-line interface for iVAE.

This module provides the command-line interface for training iVAE models
on single-cell gene expression data stored in h5ad format.
"""

import argparse
import os
from .agent import agent
import torch
import scanpy as sc
import numpy as np

def main():
    """
    Main entry point for the iVAE command-line interface.
    
    This function parses command-line arguments, loads the data, initializes
    the iVAE model, trains it, and saves the results.
    
    The CLI supports customization of all model hyperparameters including:
    - Neural network architecture (hidden_dim, latent_dim, i_dim)
    - Training parameters (epochs, lr, batch size via percent)
    - Loss weights (beta, irecon, dip, tc, info)
    
    Results are saved as NumPy arrays in the specified output directory.
    """
    parser = argparse.ArgumentParser(
        description='Train iVAE model on single-cell gene expression data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate for Adam optimizer')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data.h5ad', 
                       help='Path to the input h5ad file')
    parser.add_argument('--layer', type=str, default='counts', 
                       help='Layer name in AnnData object to use (e.g., "counts", "X")')
    parser.add_argument('--percent', type=float, default=0.01, 
                       help='Fraction of data to use per batch (0 < percent <= 1)')
    
    # Architecture parameters
    parser.add_argument('--hidden_dim', type=int, default=128, 
                       help='Dimension of hidden layers in encoder/decoder')
    parser.add_argument('--latent_dim', type=int, default=10, 
                       help='Dimension of the main latent space')
    parser.add_argument('--i_dim', type=int, default=2, 
                       help='Dimension of interpretative latent space (bottleneck)')
    
    # Loss weight parameters
    parser.add_argument('--beta', type=float, default=1.0, 
                       help='Weight for KL divergence term')
    parser.add_argument('--irecon', type=float, default=0.0, 
                       help='Weight for interpretative reconstruction loss')
    parser.add_argument('--dip', type=float, default=0.0, 
                       help='Weight for DIP (disentanglement) loss')
    parser.add_argument('--tc', type=float, default=0.0, 
                       help='Weight for total correlation loss')
    parser.add_argument('--info', type=float, default=0.0, 
                       help='Weight for InfoVAE MMD loss')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='iVAE_output', 
                       help='Directory to save output files')

    args = parser.parse_args()

    # Load single-cell data
    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Data shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize the agent object
    print("Initializing iVAE model...")
    ag = agent(
        adata=adata,
        layer=args.layer,
        percent=args.percent,
        irecon=args.irecon,
        beta=args.beta,
        dip=args.dip,
        tc=args.tc,
        info=args.info,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        i_dim=args.i_dim,
        lr=args.lr,
        device=device
    )

    # Train the model
    print(f"Training for {args.epochs} epochs...")
    ag.fit(epochs=args.epochs)

    # Get the latent space representations
    print("Extracting latent representations...")
    iembed = ag.get_iembed()
    latent = ag.get_latent()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the results
    iembed_path = os.path.join(args.output_dir, 'iembed.npy')
    latent_path = os.path.join(args.output_dir, 'latent.npy')
    
    np.save(iembed_path, iembed)
    np.save(latent_path, latent)

    print(f"\nTraining complete!")
    print(f"Results saved to '{args.output_dir}/':")
    print(f"  - iembed.npy: shape {iembed.shape}")
    print(f"  - latent.npy: shape {latent.shape}")

if __name__ == '__main__':
    main()
