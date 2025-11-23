"""
iVAE model implementation with multiple loss functions.

This module implements the core iVAE model that combines a VAE architecture
with various regularization techniques including KL divergence, DIP (Disentangled
Inferred Prior), Beta-TC VAE, and InfoVAE losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE

class iVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Interpretable Variational Autoencoder model for single-cell data.
    
    This class combines multiple loss functions and regularization techniques to learn
    interpretable latent representations of single-cell gene expression data. It uses
    a negative binomial distribution for reconstruction and supports various disentanglement
    objectives.
    
    Parameters
    ----------
    irecon : float
        Weight for the interpretative reconstruction loss. If > 0, penalizes the
        difference between reconstruction from interpretative and original latent codes.
    beta : float
        Weight for the KL divergence term. Higher values encourage latent codes
        to match the prior distribution (standard normal).
    dip : float
        Weight for the DIP (Disentangled Inferred Prior) loss. Encourages diagonal
        covariance structure in the latent space.
    tc : float
        Weight for the total correlation (TC) term from Beta-TC VAE. Encourages
        factorized latent representations.
    info : float
        Weight for the InfoVAE MMD (Maximum Mean Discrepancy) loss. Encourages
        the latent distribution to match the prior.
    state_dim : int
        Dimension of the input/output state (number of genes).
    hidden_dim : int
        Dimension of the hidden layers.
    latent_dim : int
        Dimension of the main latent space.
    i_dim : int
        Dimension of the interpretative latent space.
    lr : float
        Learning rate for the Adam optimizer.
    device : torch.device
        Device to run computations on (CPU or CUDA).
    
    Attributes
    ----------
    nn : VAE
        The VAE neural network model.
    nn_optimizer : torch.optim.Adam
        Optimizer for training the VAE.
    loss : list
        Training history storing loss components for each update.
    """
    def __init__(
        self,
        irecon,
        beta,
        dip,
        tc,
        info,
        state_dim, 
        hidden_dim, 
        latent_dim,
        i_dim,
        lr,
        device,
        *args, 
        **kwargs
    ):
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.nn = VAE(state_dim, hidden_dim, latent_dim, i_dim).to(device)
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.device = device
        self.loss = []
    
    def take_latent(
        self, 
        state
    ):
        """
        Extract latent representations from input data.
        
        Parameters
        ----------
        state : numpy.ndarray or torch.Tensor
            Input gene expression data of shape (n_samples, n_genes).
        
        Returns
        -------
        numpy.ndarray
            Latent representations of shape (n_samples, latent_dim).
        """
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, _, _, _, _, _, _ = self.nn(state)
        return q_z.detach().cpu().numpy()
        
    def update(
        self, 
        states
    ):
        """
        Perform one training step with the given batch of data.
        
        This method computes the total loss (reconstruction + regularization terms)
        and updates the model parameters via backpropagation.
        
        Parameters
        ----------
        states : numpy.ndarray or torch.Tensor
            Batch of gene expression data of shape (batch_size, n_genes).
        
        Notes
        -----
        The total loss is composed of:
        - Negative binomial reconstruction loss
        - Optional interpretative reconstruction loss (if irecon > 0)
        - KL divergence (weighted by beta)
        - Optional DIP loss (if dip > 0)
        - Optional TC loss (if tc > 0)
        - Optional MMD loss (if info > 0)
        """
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, pred_x, le, ld, pred_xl = self.nn(states)
       
        # Scale predictions by library size (total count per cell)
        l = states.sum(-1).view(-1,1)
        pred_x = pred_x * l
        
        # Compute negative binomial reconstruction loss
        disp = torch.exp(self.nn.decoder.disp)
        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()
        
        # Compute interpretative reconstruction loss if enabled
        if self.irecon:
            pred_xl = pred_xl * l
            irecon_loss = - self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
        else:
            irecon_loss = torch.zeros(1).to(self.device)
        
        # Compute KL divergence from standard normal prior
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        
        # Compute DIP loss if enabled (encourages diagonal covariance)
        if self.dip:
            dip_loss = self.dip * self._dip_loss(q_m ,q_s)
        else:
            dip_loss = torch.zeros(1).to(self.device)
        
        # Compute total correlation loss if enabled (encourages factorization)
        if self.tc:
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m ,q_s)
        else:
            tc_loss = torch.zeros(1).to(self.device)
        
        # Compute MMD loss if enabled (matches prior distribution)
        if self.info:
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z))
        else:
            mmd_loss = torch.zeros(1).to(self.device)
        
        # Combine all loss terms
        total_loss = recon_loss + irecon_loss + kl_div + dip_loss + tc_loss + mmd_loss
            
        # Backpropagation and optimization step
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        self.nn_optimizer.step()

        # Record loss components for monitoring
        self.loss.append((
            total_loss.item(),
            recon_loss.item(), 
            irecon_loss.item(),
            kl_div.item(), 
            dip_loss.item(),
            tc_loss.item(), 
            mmd_loss.item()
        ))