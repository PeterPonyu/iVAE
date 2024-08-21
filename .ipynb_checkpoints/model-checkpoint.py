import torch
import torch.nn as nn
import torch.optim as optim
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE

class iVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
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
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, _, _, _, _, _, _ = self.nn(state)
        return q_z.detach().cpu().numpy()
        
    def update(
        self, 
        states
    ):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, pred_x, le, ld, pred_xl = self.nn(states)
       
        l = states.sum(-1).view(-1,1)
        pred_x = pred_x * l
        
        disp = torch.exp(self.nn.decoder.disp)
        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()
        
        if self.irecon:
            pred_xl = pred_xl * l
            irecon_loss = - self.irecon * self._log_nb(states, pred_xl, disp).sum(-1).mean()
        else:
            irecon_loss = torch.zeros(1).to(self.device)
        
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        
        if self.dip:
            dip_loss = self.dip * self._dip_loss(q_m ,q_s)
        else:
            dip_loss = torch.zeros(1).to(self.device)
        
        if self.tc:
            tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m ,q_s)
        else:
            tc_loss = torch.zeros(1).to(self.device)
        
        if self.info:
            mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z))
        else:
            mmd_loss = torch.zeros(1).to(self.device)
        
        total_loss = recon_loss + irecon_loss + kl_div + dip_loss + tc_loss + mmd_loss
            
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        self.nn_optimizer.step()

        self.loss.append((
            total_loss.item(),
            recon_loss.item(), 
            irecon_loss.item(),
            kl_div.item(), 
            dip_loss.item(),
            tc_loss.item(), 
            mmd_loss.item()
        ))