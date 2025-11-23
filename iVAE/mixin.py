"""
Mixin classes providing various loss functions and utilities for iVAE.

This module contains mixin classes that implement different loss functions and
regularization techniques used in the iVAE model, as well as environment-related
utilities for data handling and evaluation.
"""

import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score


class scviMixin:
    """
    Mixin providing scVI-style loss functions.
    
    This mixin implements the KL divergence and negative binomial log-likelihood
    functions used in the scVI model for single-cell data.
    """
    
    def _normal_kl(
        self, 
        mu1, 
        lv1, 
        mu2, 
        lv2
    ):
        """
        Compute KL divergence between two Gaussian distributions.
        
        Parameters
        ----------
        mu1 : torch.Tensor
            Mean of the first Gaussian distribution.
        lv1 : torch.Tensor
            Log-variance of the first Gaussian distribution.
        mu2 : torch.Tensor
            Mean of the second Gaussian distribution.
        lv2 : torch.Tensor
            Log-variance of the second Gaussian distribution.
        
        Returns
        -------
        torch.Tensor
            KL divergence KL(N(mu1, exp(lv1)) || N(mu2, exp(lv2))).
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2)**2.) / (2. * v2) - 0.5
        return kl
    
    def _log_nb(
        self, 
        x, 
        mu, 
        theta, 
        eps=1e-8
    ):
        """
        Compute log-likelihood of negative binomial distribution.
        
        The negative binomial distribution is commonly used for count data like
        gene expression, as it can model over-dispersion.
        
        Parameters
        ----------
        x : torch.Tensor
            Observed count data.
        mu : torch.Tensor
            Mean parameter of the negative binomial distribution.
        theta : torch.Tensor
            Dispersion parameter of the negative binomial distribution.
            Higher values indicate less over-dispersion.
        eps : float, optional
            Small constant for numerical stability, by default 1e-8.
        
        Returns
        -------
        torch.Tensor
            Log-likelihood values for each observation.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res


class betatcMixin:
    """
    Mixin providing Beta-TC VAE loss functions.
    
    This mixin implements the total correlation (TC) decomposition from the
    Beta-TC VAE paper, which encourages factorized latent representations.
    """
    
    def _betatc_compute_gaussian_log_density(
        self, 
        samples, 
        mean, 
        log_var
    ):
        """
        Compute log density of Gaussian distribution.
        
        Parameters
        ----------
        samples : torch.Tensor
            Sampled points from the distribution.
        mean : torch.Tensor
            Mean of the Gaussian distribution.
        log_var : torch.Tensor
            Log-variance of the Gaussian distribution.
        
        Returns
        -------
        torch.Tensor
            Log density values.
        """
        import math
        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)    
    
    def _betatc_compute_total_correlation(
        self, 
        z_sampled, 
        z_mean, 
        z_logvar
    ):
        """
        Compute the total correlation (TC) term.
        
        Total correlation measures the mutual information between latent dimensions,
        quantifying how much they deviate from being independent.
        
        Parameters
        ----------
        z_sampled : torch.Tensor
            Sampled latent codes of shape (batch_size, latent_dim).
        z_mean : torch.Tensor
            Mean of latent distribution of shape (batch_size, latent_dim).
        z_logvar : torch.Tensor
            Log-variance of latent distribution of shape (batch_size, latent_dim).
        
        Returns
        -------
        torch.Tensor
            Total correlation value (scalar).
        """
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1), 
            z_mean.unsqueeze(dim=0), 
            z_logvar.unsqueeze(dim=0)
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """
    Mixin providing InfoVAE loss functions.
    
    This mixin implements Maximum Mean Discrepancy (MMD) from InfoVAE, which
    matches the aggregated posterior to the prior distribution.
    """
    
    def _compute_mmd(
        self, 
        z_posterior_samples, 
        z_prior_samples
    ):
        """
        Compute Maximum Mean Discrepancy (MMD) between posterior and prior.
        
        MMD is a kernel-based distance metric between two distributions. It's used
        to encourage the aggregated posterior to match the prior distribution.
        
        Parameters
        ----------
        z_posterior_samples : torch.Tensor
            Samples from the posterior distribution.
        z_prior_samples : torch.Tensor
            Samples from the prior distribution.
        
        Returns
        -------
        torch.Tensor
            MMD value (scalar).
        """
        mean_pz_pz = self._compute_unbiased_mean(self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True)
        mean_pz_qz = self._compute_unbiased_mean(self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False)
        mean_qz_qz = self._compute_unbiased_mean(self._compute_kernel(z_posterior_samples, z_posterior_samples), unbaised=True)
        mmd = mean_pz_pz - 2*mean_pz_qz + mean_qz_qz
        return mmd
    
    def _compute_unbiased_mean(
        self, 
        kernel, 
        unbaised
    ):
        """
        Compute (unbiased) mean of kernel matrix.
        
        Parameters
        ----------
        kernel : torch.Tensor
            Kernel matrix.
        unbaised : bool
            If True, exclude diagonal elements to get unbiased estimator.
        
        Returns
        -------
        torch.Tensor
            Mean kernel value.
        """
        N, M = kernel.shape
        if unbaised:
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(kernel, dim1=0, dim2=1).sum(dim=-1)
            mean_kernel = sum_kernel / (N*(N-1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))
        return mean_kernel
    
    def _compute_kernel(
        self, 
        z0, 
        z1
    ):
        """
        Compute RBF kernel matrix between two sets of samples.
        
        Parameters
        ----------
        z0 : torch.Tensor
            First set of samples of shape (batch_size, z_dim).
        z1 : torch.Tensor
            Second set of samples of shape (batch_size, z_dim).
        
        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_size, batch_size).
        """
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(-2)
        z1 = z1.unsqueeze(-3)
        z0 = z0.expand(batch_size, batch_size, z_size) 
        z1 = z1.expand(batch_size, batch_size, z_size) 
        kernel = self._kernel_rbf(z0, z1)
        return kernel
    
    def _kernel_rbf(
        self, 
        x, 
        y
    ):
        """
        Compute Radial Basis Function (RBF) kernel.
        
        Parameters
        ----------
        x : torch.Tensor
            First set of points.
        y : torch.Tensor
            Second set of points.
        
        Returns
        -------
        torch.Tensor
            RBF kernel values.
        """
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel

class dipMixin:
    """
    Mixin providing DIP (Disentangled Inferred Prior) loss functions.
    
    This mixin implements the DIP loss that encourages the covariance matrix
    of the latent distribution to be diagonal with unit variance.
    """
    
    def _dip_loss(
        self,
        q_m,
        q_s
    ):
        """
        Compute DIP loss for disentanglement.
        
        DIP loss penalizes:
        1. Diagonal elements deviating from 1 (unit variance)
        2. Off-diagonal elements deviating from 0 (independence)
        
        Parameters
        ----------
        q_m : torch.Tensor
            Mean of latent distribution of shape (batch_size, latent_dim).
        q_s : torch.Tensor
            Log-variance of latent distribution of shape (batch_size, latent_dim).
        
        Returns
        -------
        torch.Tensor
            DIP loss value (scalar).
        """
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1)**2)  # Penalize deviation from unit variance
        dip_loss_od = torch.sum(cov_off_diag**2)   # Penalize non-zero correlations
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss
        
    def _dip_cov_matrix(
        self, 
        q_m,
        q_s
    ): 
        """
        Compute covariance matrix of the latent distribution.
        
        Parameters
        ----------
        q_m : torch.Tensor
            Mean of latent distribution.
        q_s : torch.Tensor
            Log-variance of latent distribution.
        
        Returns
        -------
        torch.Tensor
            Covariance matrix.
        """
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.diag(q_s.exp()), dim=0)
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class envMixin:
    """
    Mixin providing environment utilities for data handling and evaluation.
    
    This mixin implements functions for clustering, scoring, and correlation
    analysis of latent representations.
    """
    
    def _calc_score(
        self,
        latent
    ):
        """
        Calculate clustering performance scores for latent representations.
        
        Parameters
        ----------
        latent : numpy.ndarray
            Latent representations of shape (n_samples, latent_dim).
        
        Returns
        -------
        tuple
            A tuple of scores: (ARI, NMI, ASW, C_H, D_B, P_C).
        """
        n = latent.shape[1]
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels)
        return scores
        
    def _calc_label(
        self,
        latent
    ):
        """
        Perform K-means clustering on latent representations.
        
        Parameters
        ----------
        latent : numpy.ndarray
            Latent representations.
        
        Returns
        -------
        numpy.ndarray
            Cluster labels.
        """
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels
    
    def _calc_corr(
        self,
        latent
    ):
        """
        Calculate average absolute correlation between latent dimensions.
        
        This metric quantifies how correlated the latent dimensions are, which
        is desirable in iVAE as it indicates the model has learned meaningful
        relationships between features.
        
        Parameters
        ----------
        latent : numpy.ndarray
            Latent representations.
        
        Returns
        -------
        float
            Average absolute correlation (excluding self-correlation).
        """
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1
        
    def _metrics(
        self,
        latent,
        labels
    ):
        """
        Compute multiple clustering evaluation metrics.
        
        Parameters
        ----------
        latent : numpy.ndarray
            Latent representations.
        labels : numpy.ndarray
            Predicted cluster labels.
        
        Returns
        -------
        tuple
            Metrics: (ARI, NMI, ASW, C_H, D_B, P_C) where:
            - ARI: Adjusted Rand Index (agreement with true labels)
            - NMI: Normalized Mutual Information (agreement with true labels)
            - ASW: Average Silhouette Width (cluster cohesion)
            - C_H: Calinski-Harabasz score (cluster separation)
            - D_B: Davies-Bouldin score (cluster compactness, lower is better)
            - P_C: Pearson Correlation (average correlation between dimensions)
        """
        ARI = adjusted_mutual_info_score(self.labels[self.idx], labels)
        NMI = normalized_mutual_info_score(self.labels[self.idx], labels)
        ASW = silhouette_score(latent, labels)
        C_H = calinski_harabasz_score(latent, labels)
        D_B = davies_bouldin_score(latent, labels)
        P_C = self._calc_corr(latent)
        return ARI, NMI, ASW, C_H, D_B, P_C






