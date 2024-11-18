from .model import iVAE
from .mixin import envMixin
import numpy as np
from sklearn.cluster import KMeans


class Env(iVAE, envMixin):
    
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
        data, idx = self._sample_data()
        self.idx = idx
        return data
        
    def step(
        self,
        data
    ):
        self.update(data)
        latent = self.take_latent(data)
        score = self._calc_score(latent)
        self.score.append(score)
    
    def _sample_data(
        self,
        
    ):
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
        self.X = np.log1p(adata.layers[layer].A)
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        return 