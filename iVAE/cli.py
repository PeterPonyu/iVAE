#!/usr/bin/env python

import argparse
import os
from agent import agent
import torch
from anndata import read_h5ad
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run the agent class training script')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--layer', type=str, default='counts', help='Layer to use from adata')
    parser.add_argument('--percent', type=float, default=0.01, help='Percent parameter value')
    parser.add_argument('--irecon', type=float, default=0.0, help='Irecon parameter value')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter value')
    parser.add_argument('--dip', type=float, default=0.0, help='Dip parameter value')
    parser.add_argument('--tc', type=float, default=0.0, help='TC parameter value')
    parser.add_argument('--info', type=float, default=0.0, help='Info parameter value')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension size')
    parser.add_argument('--i_dim', type=int, default=2, help='i dimension size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data.h5ad', help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='iVAE_output', help='Directory to save the results')

    args = parser.parse_args()

    # Load data
    adata = read_h5ad(args.data_path)

    # Initialize the agent object
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
        device=args.device if hasattr(args, 'device') else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    # Train the model
    ag.fit(epochs=args.epochs)

    # Get the latent space representations
    iembed = ag.get_iembed()
    latent = ag.get_latent()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the results
    np.save(os.path.join(args.output_dir, 'iembed.npy'), iembed)
    np.save(os.path.join(args.output_dir, 'latent.npy'), latent)

    print(f"Results have been saved to the '{args.output_dir}' directory.")

if __name__ == '__main__':
    main()