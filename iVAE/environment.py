"""
Environment class for managing data and training workflow.

This module provides the Env class that manages the training environment,
including data loading, batching, and evaluation scoring.
"""

from .model import iVAE
from .mixin import envMixin
import numpy as np
from sklearn.cluster import KMeans


class Env(iVAE, envMixin):
    """
    Training environment for the iVAE model.
    
    This class manages the training workflow including data preprocessing,
    batch sampling, model updates, and performance evaluation. It inherits
    from both iVAE (the model) and envMixin (evaluation utilities).
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell gene expression data.
    layer : str
        Name of the layer in adata.layers to use for training.
    percent : float
        Proportion of data to use in each training batch (0 < percent <= 1).
    irecon : float
        Weight for interpretative reconstruction loss.
    beta : float
        Weight for KL divergence term.
    dip : float
        Weight for DIP loss.
    tc : float
        Weight for total correlation loss.
    info : float
        Weight for InfoVAE MMD loss.
    hidden_dim : int
        Dimension of hidden layers in the neural network.
    latent_dim : int
        Dimension of the main latent space.
    i_dim : int
        Dimension of the interpretative latent space.
    lr : float
        Learning rate for optimization.
    device : torch.device
        Device for computation (CPU or CUDA).
    
    Attributes
    ----------
    X : numpy.ndarray
        Log-normalized gene expression matrix of shape (n_cells, n_genes).
    n_obs : int
        Number of cells (observations).
    n_var : int
        Number of genes (variables).
    batch_size : int
        Number of samples per training batch.
    labels : numpy.ndarray
        Reference cluster labels for evaluation (from K-means on input data).
    score : list
        Training history of evaluation scores.
    """
    
    def __init__(
        self,
        adata,
        layer,
        percent,
        irecon,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        lr,
        device,
        *args,
        **kwargs
    ):
        self._register_anndata(adata, layer, latent_dim)
        self.batch_size = int(percent * self.n_obs)
        super().__init__(
            irecon     = irecon,
            beta       = beta,
            dip        = dip,
            tc         = tc,
            info       = info,
            state_dim  = self.n_var,
            hidden_dim = hidden_dim, 
            latent_dim = latent_dim,
            i_dim      = i_dim,
            lr         = lr, 
            device     = device
        )
        self.score = []
    
    def load_data(
        self,
    ):
        """
        Load a random batch of data for training.
        
        Returns
        -------
        numpy.ndarray
            A batch of gene expression data of shape (batch_size, n_genes).
        """
        data, idx = self._sample_data()
        self.idx = idx
        return data
        
    def step(
        self,
        data
    ):
        """
        Perform one training step: update model and evaluate performance.
        
        Parameters
        ----------
        data : numpy.ndarray
            Batch of gene expression data.
        """
        self.update(data)
        latent = self.take_latent(data)
        score = self._calc_score(latent)
        self.score.append(score)
    
    def _sample_data(
        self,
        
    ):
        """
        Sample a random batch from the dataset.
        
        Returns
        -------
        data : numpy.ndarray
            Batch of gene expression data.
        idx_ : numpy.ndarray
            Indices of the sampled cells.
        """
        idx = np.random.permutation(self.n_obs)
        idx_ = np.random.choice(idx, self.batch_size)
        data = self.X[idx_,:]
        return data, idx_

    def _register_anndata(
        self,
        adata,
        layer: str,
        latent_dim
    ):
        """
        Register and preprocess AnnData object.
        
        This method extracts the specified layer from the AnnData object,
        applies log1p transformation, and creates reference labels for evaluation.
        
        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix.
        layer : str
            Name of the layer to use (must exist in adata.layers).
        latent_dim : int
            Number of clusters to create for reference labels.
        """
        self.X = np.log1p(adata.layers[layer].A)
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        return 