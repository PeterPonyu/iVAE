"""
Main agent class for training and using iVAE models.

This module provides the user-facing agent class that wraps the complete
iVAE training pipeline with a simple interface.
"""

from .environment import Env
from anndata import AnnData
import torch
import tqdm

class agent(Env):
    """  
    High-level interface for training and using iVAE models.
    
    The agent class provides a user-friendly interface for training interpretable
    Variational Autoencoders (iVAE) on single-cell transcriptomics data. It handles
    data preprocessing, model training, and extraction of learned representations.
    
    iVAE enhances standard VAE by incorporating an interpretative module that
    increases correlation between latent components, helping capture biologically
    meaningful gene expression patterns in single-cell data.

    Parameters  
    ----------  
    adata : AnnData  
        Annotated data matrix containing single-cell gene expression data.
        Should have at least one layer (e.g., 'counts', 'X') with raw or normalized counts.
    layer : str, optional  
        The layer of the AnnData object to use, by default 'counts'.
        Common options: 'counts', 'X', or custom layer names.
    percent : float, optional  
        Fraction of cells to use per training batch (0 < percent <= 1), by default 0.01.
        Smaller values = smaller batches, more frequent updates. Larger values = more stable gradients.
    irecon : float, optional  
        Weight for interpretative reconstruction loss, by default 0.0.
        If > 0, penalizes reconstruction errors from the interpretative bottleneck,
        encouraging more interpretable latent representations.
    beta : float, optional  
        Weight for KL divergence term (beta-VAE), by default 1.0.
        Higher values encourage latent codes closer to prior (standard normal),
        potentially improving disentanglement but may reduce reconstruction quality.
    dip : float, optional  
        Weight for DIP (Disentangled Inferred Prior) loss, by default 0.0.
        If > 0, encourages diagonal covariance in latent space for disentanglement.
    tc : float, optional  
        Weight for Total Correlation (TC) loss from Beta-TC VAE, by default 0.0.
        If > 0, encourages factorized (independent) latent dimensions.
    info : float, optional  
        Weight for InfoVAE MMD (Maximum Mean Discrepancy) loss, by default 0.0.
        If > 0, matches aggregated posterior to prior using kernel-based distance.
    hidden_dim : int, optional  
        Dimension of hidden layers in encoder/decoder networks, by default 128.
        Larger values increase model capacity but require more data and computation.
    latent_dim : int, optional  
        Dimension of the main latent space, by default 10.
        Should roughly match the expected number of cell types or states.
    i_dim : int, optional  
        Dimension of the interpretative latent space (bottleneck), by default 2.
        This compressed representation encourages learning of correlated patterns.
        Should be smaller than latent_dim.
    lr : float, optional  
        Learning rate for Adam optimizer, by default 1e-4.
    device : torch.device, optional  
        Device to run computations on, by default uses GPU if available, otherwise CPU.

    Methods  
    -------  
    fit(epochs=1000)  
        Train the model on the data for a specified number of epochs.
        Returns the trained agent instance.
    get_iembed()  
        Extract the interpretative embedding (intermediate bottleneck representation).
        Returns a NumPy array of shape (n_cells, i_dim).
    get_latent()  
        Extract the main latent representation.
        Returns a NumPy array of shape (n_cells, latent_dim).
        
    Examples
    --------
    Basic usage with default parameters:
    
    >>> import scanpy as sc
    >>> from iVAE import agent
    >>> 
    >>> # Load single-cell data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> 
    >>> # Train iVAE model
    >>> ag = agent(adata, layer='counts', latent_dim=10)
    >>> ag.fit(epochs=500)
    >>> 
    >>> # Extract representations
    >>> latent = ag.get_latent()
    >>> iembed = ag.get_iembed()
    
    With custom regularization:
    
    >>> # Train with interpretative reconstruction and disentanglement
    >>> ag = agent(
    ...     adata, 
    ...     layer='counts',
    ...     latent_dim=10,
    ...     i_dim=3,
    ...     irecon=0.5,    # Enable interpretative reconstruction
    ...     beta=2.0,       # Stronger KL regularization  
    ...     dip=1.0         # Enable disentanglement
    ... )
    >>> ag.fit(epochs=1000)
    """ 
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        percent: float = .01,
        irecon: float = .0,
        beta: float = 1.,
        dip: float = .0,
        tc: float = .0,
        info: float = .0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        lr: float = 1e-4,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            lr=lr,
            device=device
        )
        
    def fit(
        self,
        epochs:int=1000
    ):
        """  
        Train the iVAE model on the data.
        
        This method trains the model for the specified number of epochs, updating
        parameters via stochastic gradient descent on random batches. Progress is
        displayed with a progress bar showing loss and clustering metrics.

        Parameters  
        ----------  
        epochs : int, optional  
            Number of training epochs, by default 1000.
            One epoch = one pass through a random batch of the data.

        Returns  
        -------  
        agent  
            The fitted agent instance (self), allowing method chaining.
            
        Notes
        -----
        During training, the following metrics are displayed every 10 epochs:
        - Loss: Total loss value
        - ARI: Adjusted Rand Index (clustering agreement with reference)
        - NMI: Normalized Mutual Information
        - ASW: Average Silhouette Width (cluster cohesion)
        - C_H: Calinski-Harabasz score (cluster separation)
        - D_B: Davies-Bouldin score (lower is better)
        - P_C: Pearson correlation (average correlation between latent dimensions)
        """
        with tqdm.tqdm(total=int(epochs), desc='Training iVAE', ncols=150) as pbar:
            for i in range(int(epochs)):
                data = self.load_data()
                self.step(data)
                if (i+1) % 10 == 0:
                    pbar.set_postfix({
                        'Loss':f'{self.loss[-1][0]:.2f}',
                        'ARI':f'{(self.score[-1][0]):.2f}',
                        'NMI':f'{(self.score[-1][1]):.2f}',
                        'ASW':f'{(self.score[-1][2]):.2f}',
                        'C_H':f'{(self.score[-1][3]):.2f}',
                        'D_B':f'{(self.score[-1][4]):.2f}',
                        'P_C':f'{(self.score[-1][5]):.2f}'
                    })
                pbar.update(1)
        return self
    
    def get_iembed(
        self
    ):
        """  
        Extract the interpretative embedding from the trained model.
        
        The interpretative embedding is the compressed representation from the
        bottleneck layer (dimension i_dim). This representation captures the most
        important correlated patterns learned by the model.

        Returns  
        -------  
        numpy.ndarray  
            The interpretative embedding as a NumPy array of shape (n_cells, i_dim).
            Each row represents one cell, each column is one interpretative dimension.
            
        Notes
        -----
        This embedding passes through the interpretative bottleneck (latent_encoder),
        forcing the model to learn a more compact and interpretable representation.
        """
        return self.nn(torch.tensor(self.X, dtype=torch.float).to(self.device))[-3].detach().cpu().numpy()    
        
    def get_latent(
        self,
    ):
        """  
        Extract the main latent representation from the trained model.
        
        The latent representation is the primary compressed encoding of the input
        data (dimension latent_dim). This can be used for downstream analysis like
        clustering, visualization, or as features for other models.

        Returns  
        -------  
        numpy.ndarray  
            The latent representation as a NumPy array of shape (n_cells, latent_dim).
            Each row represents one cell, each column is one latent dimension.
            
        Notes
        -----
        This is the standard VAE latent code before passing through the
        interpretative module. It typically has higher dimension than iembed
        and captures more detailed information about the data.
        """
        q_z = self.take_latent(self.X)
        return q_z
