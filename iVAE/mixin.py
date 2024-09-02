import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score


class scviMixin:
    
    def _normal_kl(
        self, 
        mu1, 
        lv1, 
        mu2, 
        lv2
    ):
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
    
    def _betatc_compute_gaussian_log_density(
        self, 
        samples, 
        mean, 
        log_var
    ):
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
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1), 
            z_mean.unsqueeze(dim=0), 
            z_logvar.unsqueeze(dim=0)
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class infoMixin:
    
    def _compute_mmd(
        self, 
        z_posterior_samples, 
        z_prior_samples
    ):
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
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel

class dipMixin:
    
    def _dip_loss(
        self,
        q_m,
        q_s
    ):
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1)**2)  
        dip_loss_od = torch.sum(cov_off_diag**2)   
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss
        
    def _dip_cov_matrix(
        self, 
        q_m,
        q_s
    ): 
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.diag(q_s.exp()), dim=0)
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class envMixin:
    
    def _calc_score(
        self,
        latent
    ):
        
        n = latent.shape[1]
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels)
        return scores
        
    def _calc_label(
        self,
        latent
    ):
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels
    
    def _calc_corr(
        self,
        latent
    ):
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1
        
    def _metrics(
        self,
        latent,
        labels
    ):
        ARI = adjusted_mutual_info_score(self.labels[self.idx], labels)
        NMI = normalized_mutual_info_score(self.labels[self.idx], labels)
        ASW = silhouette_score(latent, labels)
        C_H = calinski_harabasz_score(latent, labels)
        D_B = davies_bouldin_score(latent, labels)
        P_C = self._calc_corr(latent)
        return ARI, NMI, ASW, C_H, D_B, P_C






