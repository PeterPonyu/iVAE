## About

iVAE is an enhanced representation learning method designed for capturing lineage features and gene expression patterns in single-cell transcriptomics. Compared to a standard VAE, iVAE incorporates a pivotal interpretative module that increases the correlation between latent components. This enhanced correlation helps the model learn gene expression patterns in single-cell data where correlations are present.

<img src='source/_static/fig.png' width='300' align='center'>

## Installation

[![PyPI](https://img.shields.io/pypi/v/iVAE.svg?color=brightgreen&style=flat)](https://pypi.org/project/iVAE/)

You can install the `iVAE` package using:

```bash
pip install iVAE
```

This repository is hosted at [iVAE GitHub Repository](https://github.com/PeterPonyu/iVAE).

## Usage

You can customize the behavior of the script by providing additional arguments:

- `--epochs`: Number of training epochs (default: 1000)
- `--layer`: Layer to use from the AnnData object (default: 'counts')
- `--percent`: Percent parameter value (default: 0.01)
- `--irecon`: Irecon parameter value (default: 0.0)
- `--beta`: Beta parameter value (default: 1.0)
- `--dip`: Dip parameter value (default: 0.0)
- `--tc`: TC parameter value (default: 0.0)
- `--info`: Info parameter value (default: 0.0)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--latent_dim`: Latent dimension size (default: 10)
- `--i_dim`: i dimension size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--data_path`: Path to the data file (default: 'data.h5ad')
- `--output_dir`: Directory to save the results (default: 'iVAE_output')

Example of running with custom parameters:

```bash
iVAE --epochs 500 --layer 'counts' --data_path 'path/to/your/data.h5ad' --output_dir 'iVAE_output'
```

### Output

After running the script, the latent space representations are saved in the specified output directory (`iVAE_output` by default):

- `iembed.npy`: Contains the output from the `get_iembed()` function.
- `latent.npy`: Contains the output from the `get_latent()` function.

These files are NumPy arrays that can be loaded using `numpy.load()` for further analysis.

### Example of Loading Output Data

You can load and analyze the output data using the following Python code:

```python
import numpy as np

# Load the iembed data
iembed = np.load('iVAE_output/iembed.npy')

# Load the latent data
latent = np.load('iVAE_output/latent.npy')

# Perform your analysis
print("iembed shape:", iembed.shape)
print("latent shape:", latent.shape)
```

## License
[![PyPI](https://img.shields.io/github/license/PeterPonyu/iVAE?style=flat-square&color=brightgreen)](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License. See the LICENSE file for details.


## Contact

For questions or issues, please contact Zeyu Fu at [fuzeyu99@126.com](mailto:fuzeyu99@126.com).

---
