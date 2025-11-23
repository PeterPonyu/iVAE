## About

iVAE is an enhanced representation learning method designed for capturing lineage features and gene expression patterns in single-cell transcriptomics. Compared to a standard VAE, iVAE incorporates a pivotal interpretative module that increases the correlation between latent components. This enhanced correlation helps the model learn gene expression patterns in single-cell data where correlations are present.

<img src='source/_static/fig.png' width='600' align='center'>

## Key Features

- **Interpretative Module**: A bottleneck layer that encourages learning of correlated patterns in gene expression
- **Multiple Loss Functions**: Supports various regularization techniques (KL, DIP, TC, MMD) for improved disentanglement
- **Single-Cell Optimized**: Uses negative binomial distribution for count data modeling
- **Easy to Use**: Simple Python API and command-line interface
- **GPU Accelerated**: Automatic GPU detection and usage when available

## Installation

[![PyPI](https://img.shields.io/pypi/v/iVAE.svg?color=brightgreen&style=flat)](https://pypi.org/project/iVAE/)

You can install the `iVAE` package using:

```bash
pip install iVAE
```

This repository is hosted at [iVAE GitHub Repository](https://github.com/PeterPonyu/iVAE).

### Requirements

- Python >= 3.9
- PyTorch >= 1.13.1
- scanpy >= 1.10.4
- scikit-learn
- numpy
- tqdm

All dependencies are automatically installed with pip.

## Quick Start

### Command-Line Interface

The simplest way to use iVAE is through the command-line interface:

```bash
# Basic usage with default parameters
iVAE --data_path data.h5ad --epochs 500 --output_dir results

# With custom parameters
iVAE --data_path data.h5ad \
     --epochs 1000 \
     --latent_dim 15 \
     --i_dim 3 \
     --beta 2.0 \
     --irecon 0.5 \
     --output_dir results
```

### Python API

You can also use iVAE directly in Python scripts:

```python
import scanpy as sc
from iVAE import agent

# Load your single-cell data
adata = sc.read_h5ad('data.h5ad')

# Initialize and train the model
model = agent(
    adata=adata,
    layer='counts',       # Layer to use from AnnData
    latent_dim=10,        # Number of latent dimensions
    i_dim=2,              # Interpretative dimension
    epochs=500
)
model.fit(epochs=500)

# Extract representations
latent = model.get_latent()      # Main latent representation
iembed = model.get_iembed()      # Interpretative embedding

# Use for downstream analysis
adata.obsm['X_ivae'] = latent
sc.pp.neighbors(adata, use_rep='X_ivae')
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

## Usage

### Command-Line Arguments

You can customize the behavior of the script by providing additional arguments:

#### Data Parameters
- `--data_path`: Path to the h5ad data file (default: 'data.h5ad')
- `--layer`: Layer to use from the AnnData object (default: 'counts'). Common options: 'counts', 'X', or custom layer names
- `--output_dir`: Directory to save the results (default: 'iVAE_output')

#### Training Parameters
- `--epochs`: Number of training epochs (default: 1000). More epochs = better convergence but longer training
- `--lr`: Learning rate for Adam optimizer (default: 1e-4)
- `--percent`: Fraction of cells per training batch (default: 0.01). Range: (0, 1]. Smaller = more updates but noisier gradients

#### Architecture Parameters
- `--hidden_dim`: Hidden layer dimension in encoder/decoder (default: 128). Larger = more capacity
- `--latent_dim`: Main latent space dimension (default: 10). Should roughly match expected cell types/states
- `--i_dim`: Interpretative latent space dimension (default: 2). Creates bottleneck for learning correlated patterns. Should be < latent_dim

#### Loss Weight Parameters
- `--beta`: Weight for KL divergence term (default: 1.0). Higher values encourage latent codes closer to prior distribution
- `--irecon`: Weight for interpretative reconstruction loss (default: 0.0). If > 0, penalizes reconstruction from interpretative bottleneck
- `--dip`: Weight for DIP (Disentangled Inferred Prior) loss (default: 0.0). If > 0, encourages diagonal covariance for disentanglement
- `--tc`: Weight for Total Correlation loss (default: 0.0). If > 0, encourages independent latent dimensions
- `--info`: Weight for InfoVAE MMD loss (default: 0.0). If > 0, matches aggregated posterior to prior

### Example Commands

Basic training with default parameters:

```bash
iVAE --data_path 'path/to/your/data.h5ad' --epochs 500 --output_dir 'results'
```

Training with interpretative reconstruction and disentanglement:

```bash
iVAE --epochs 1000 \
     --layer 'counts' \
     --data_path 'data.h5ad' \
     --latent_dim 15 \
     --i_dim 3 \
     --beta 2.0 \
     --irecon 0.5 \
     --dip 1.0 \
     --output_dir 'ivae_output'
```

### Output

After running the script, the latent space representations are saved in the specified output directory (`iVAE_output` by default):

- **`iembed.npy`**: Interpretative embedding from the bottleneck layer (shape: n_cells × i_dim). This compressed representation captures the most important correlated patterns.
- **`latent.npy`**: Main latent representation (shape: n_cells × latent_dim). This is the primary encoded representation for downstream analysis.

These files are NumPy arrays that can be loaded using `numpy.load()` for further analysis.

### Example of Loading and Using Output Data

You can load and analyze the output data using the following Python code:

```python
import numpy as np
import scanpy as sc

# Load the saved representations
iembed = np.load('iVAE_output/iembed.npy')
latent = np.load('iVAE_output/latent.npy')

# Inspect shapes
print("Interpretative embedding shape:", iembed.shape)  # (n_cells, i_dim)
print("Latent representation shape:", latent.shape)     # (n_cells, latent_dim)

# Add to AnnData object for downstream analysis
adata = sc.read_h5ad('data.h5ad')
adata.obsm['X_ivae_latent'] = latent
adata.obsm['X_ivae_iembed'] = iembed

# Perform clustering and visualization
sc.pp.neighbors(adata, use_rep='X_ivae_latent')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Visualize results
sc.pl.umap(adata, color=['leiden', 'cell_type'])

# Analyze interpretative embedding
import pandas as pd
iembed_df = pd.DataFrame(iembed, columns=[f'iDim{i}' for i in range(iembed.shape[1])])
print(iembed_df.describe())
```

## Advanced Usage

### Using Different Regularization Techniques

iVAE supports multiple regularization methods that can be combined:

```python
from iVAE import agent

# Beta-VAE style training (disentanglement via KL weighting)
model = agent(adata, beta=4.0, latent_dim=10)

# DIP-VAE style training (disentanglement via covariance penalty)
model = agent(adata, dip=10.0, latent_dim=10)

# TC-VAE style training (total correlation penalty)
model = agent(adata, tc=5.0, latent_dim=10)

# InfoVAE style training (MMD between posterior and prior)
model = agent(adata, info=1.0, latent_dim=10)

# Combined approach with interpretative reconstruction
model = agent(
    adata,
    beta=2.0,      # KL regularization
    irecon=0.5,    # Interpretative reconstruction
    dip=1.0,       # Disentanglement
    latent_dim=15,
    i_dim=3
)
model.fit(epochs=1000)
```

### Hyperparameter Tuning Tips

- **latent_dim**: Start with the expected number of cell types/states. Can be higher for complex datasets.
- **i_dim**: Should be significantly smaller than latent_dim (e.g., 2-3 for latent_dim=10).
- **irecon**: Start with 0.0, gradually increase to 0.5-1.0 if you want stronger interpretability.
- **beta**: Values > 1 encourage disentanglement but may reduce reconstruction quality. Try 1.0-4.0.
- **percent**: Use 0.01-0.1 depending on dataset size. Larger datasets can use smaller values.
- **epochs**: 500-2000 epochs usually sufficient. Monitor loss curves to determine convergence.

## API docs
[![Documentation Status](https://readthedocs.org/projects/ivae/badge/?version=latest)](https://ivae.readthedocs.io/en/latest/?badge=latest)

The documentation is available. Click the [doc](https://ivae.readthedocs.io/en/latest/) for details.

## License
[![PyPI](https://img.shields.io/github/license/PeterPonyu/iVAE?style=flat-square&color=brightgreen)](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License. See the LICENSE file for details.

## Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15686686.svg)](https://doi.org/10.5281/zenodo.15686686)

## Contact

For questions or issues, please contact Zeyu Fu at [fuzeyu99@126.com](mailto:fuzeyu99@126.com) or [fuzeyu09@gmail.com](mailto:fuzeyu09@gmail.com).

## Cite

- Fu, Z., Chen, C., Wang, S. et al. iVAE: an interpretable representation learning framework enhances clustering performance for single-cell data. *BMC Biol* **23**, 213 (2025). https://doi.org/10.1186/s12915-025-02315-7
        
        
        
        



---
