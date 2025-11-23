"""
Neural network modules for the iVAE model.

This module contains the encoder, decoder, and VAE architectures used in iVAE.
The VAE includes an interpretative module that enhances correlation between latent components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def weight_init(m):
    """
    Initialize weights for linear layers using Xavier normal initialization.
    
    Parameters
    ----------
    m : nn.Module
        The module to initialize. If it's a Linear layer, weights are initialized
        with Xavier normal and biases are set to 0.01.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, .01)

class Encoder(nn.Module):
    """
    Encoder network for the VAE that maps input data to latent space.
    
    The encoder uses a three-layer neural network to encode input gene expression
    data into a latent representation following a Gaussian distribution.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the input state (number of genes).
    hidden_dim : int
        Dimension of the hidden layers.
    action_dim : int
        Dimension of the latent space (output dimension).
    
    Notes
    -----
    The network outputs both mean (q_m) and log-variance (q_s) parameters
    of the latent Gaussian distribution. The actual latent code is sampled
    using the reparameterization trick.
    """
    def __init__(
        self, 
        state_dim, 
        hidden_dim, 
        action_dim
    ):
        super(
            Encoder, 
            self
        ).__init__()
        self.nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim*2)
        )
        self.apply(weight_init)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, state_dim).
        
        Returns
        -------
        q_z : torch.Tensor
            Sampled latent code of shape (batch_size, action_dim).
        q_m : torch.Tensor
            Mean of the latent distribution of shape (batch_size, action_dim).
        q_s : torch.Tensor
            Log-variance of the latent distribution of shape (batch_size, action_dim).
        """
        output = self.nn(x)
        q_m = output[:,:int(output.shape[-1]/2)]
        q_s = output[:,int(output.shape[-1]/2):]     
        s = F.softplus(q_s) + 1e-6
        n = Normal(q_m, s)
        q_z = n.rsample()
        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder network for the VAE that reconstructs data from latent space.
    
    The decoder uses a three-layer neural network to decode latent representations
    back to gene expression space. It uses a negative binomial distribution for
    reconstruction, which is appropriate for count data like gene expression.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the output state (number of genes).
    hidden_dim : int
        Dimension of the hidden layers.
    action_dim : int
        Dimension of the latent space (input dimension).
    
    Attributes
    ----------
    disp : nn.Parameter
        Dispersion parameter for the negative binomial distribution,
        learned during training for each gene.
    """
    def __init__(
        self, 
        state_dim, 
        hidden_dim, 
        action_dim
    ):
        super(
            Decoder, 
            self
        ).__init__()
        self.nn = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softmax(dim=-1)
        )
        self.disp = nn.Parameter(torch.randn(state_dim))
        self.apply(weight_init)

    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent representation of shape (batch_size, action_dim).
        
        Returns
        -------
        output : torch.Tensor
            Reconstructed gene expression proportions of shape (batch_size, state_dim).
            Values are normalized via softmax to sum to 1.
        """
        output = self.nn(x)
        return output
        
class VAE(nn.Module):
    """
    Interpretable Variational Autoencoder (iVAE) with an interpretative module.
    
    This VAE architecture includes a special interpretative module that compresses
    and then expands the latent representation. This module increases the correlation
    between latent components, helping the model capture gene expression patterns
    where correlations are biologically meaningful.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the input/output state (number of genes).
    hidden_dim : int
        Dimension of the hidden layers in encoder/decoder.
    action_dim : int
        Dimension of the main latent space.
    i_dim : int
        Dimension of the interpretative latent space (typically smaller than action_dim).
        This bottleneck encourages learning of correlated patterns.
    
    Attributes
    ----------
    encoder : Encoder
        Neural network that encodes input to latent space.
    decoder : Decoder
        Neural network that decodes latent space to reconstructed output.
    latent_encoder : nn.Linear
        Compresses latent space to interpretative dimension (action_dim -> i_dim).
    latent_decoder : nn.Linear
        Expands interpretative space back to latent dimension (i_dim -> action_dim).
    
    Notes
    -----
    The interpretative module (latent_encoder + latent_decoder) acts as an autoencoder
    within the VAE, creating a bottleneck that forces the model to learn more
    interpretable and correlated latent representations.
    """
    def __init__(
        self, 
        state_dim, 
        hidden_dim, 
        action_dim,
        i_dim
    ):
        super(
            VAE, 
            self
        ).__init__()
        self.encoder = Encoder(state_dim, hidden_dim, action_dim)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim)
        self.latent_encoder = nn.Linear(action_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, action_dim)
        
    def forward(
        self, 
        x
    ):
        """
        Forward pass through the iVAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data of shape (batch_size, state_dim).
        
        Returns
        -------
        q_z : torch.Tensor
            Sampled latent representation of shape (batch_size, action_dim).
        q_m : torch.Tensor
            Mean of latent distribution of shape (batch_size, action_dim).
        q_s : torch.Tensor
            Log-variance of latent distribution of shape (batch_size, action_dim).
        pred_x : torch.Tensor
            Reconstructed output from latent code of shape (batch_size, state_dim).
        le : torch.Tensor
            Interpretative embedding of shape (batch_size, i_dim).
        ld : torch.Tensor
            Decoded interpretative embedding of shape (batch_size, action_dim).
        pred_xl : torch.Tensor
            Reconstructed output from interpretative code of shape (batch_size, state_dim).
        """
        q_z, q_m, q_s = self.encoder(x)
        
        # Interpretative module: compress and expand latent code
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)
        
        # Reconstruct from both original and interpretative latent codes
        pred_x = self.decoder(q_z)
        pred_xl = self.decoder(ld)
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl