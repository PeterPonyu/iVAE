import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, .01)

class Encoder(nn.Module):
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
        output = self.nn(x)
        q_m = output[:,:int(output.shape[-1]/2)]
        q_s = output[:,int(output.shape[-1]/2):]     
        s = F.softplus(q_s) + 1e-6
        n = Normal(q_m, s)
        q_z = n.rsample()
        return q_z, q_m, q_s


class Decoder(nn.Module):
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
        output = self.nn(x)
        return output
        
class VAE(nn.Module):
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
        
        q_z, q_m, q_s = self.encoder(x)
        
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)
        
        pred_x = self.decoder(q_z)
        pred_xl = self.decoder(ld)
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl