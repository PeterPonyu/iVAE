from .environment import Env
from anndata import AnnData
import torch
import tqdm

class agent(Env):
    """  
    An agent class for modeling single-cell transcriptomics data using a variational autoencoder approach.  

    Parameters  
    ----------  
    adata : AnnData  
        Annotated data matrix.  
    layer : str, optional  
        The layer of the AnnData object to use, by default 'counts'.  
    percent : float, optional  
        Percent parameter value, by default 0.01.  
    irecon : float, optional  
        Irecon parameter value, by default 0.0.  
    beta : float, optional  
        Beta parameter value, by default 1.0.  
    dip : float, optional  
        Dip parameter value, by default 0.0.  
    tc : float, optional  
        TC parameter value, by default 0.0.  
    info : float, optional  
        Info parameter value, by default 0.0.  
    hidden_dim : int, optional  
        Hidden dimension size, by default 128.  
    latent_dim : int, optional  
        Latent dimension size, by default 10.  
    i_dim : int, optional  
        I dimension size, by default 2.  
    lr : float, optional  
        Learning rate, by default 1e-4.  
    device : torch.device, optional  
        Device to run the model on, by default uses GPU if available, otherwise CPU.  

    Methods  
    -------  
    fit(epochs=1000)  
        Fits the model to the data for a specified number of epochs.  
    get_iembed()  
        Returns the intermediate embedding from the neural network.  
    get_latent()  
        Returns the latent representation of the data.  
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
        Fits the model to the data for a specified number of epochs.  

        Parameters  
        ----------  
        epochs : int, optional  
            Number of training epochs, by default 1000.  

        Returns  
        -------  
        agent  
            The fitted agent instance.  
        """
        with tqdm.tqdm(total=int(epochs), desc='Fitting', ncols=150) as pbar:
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
        Returns the intermediate embedding from the neural network.  

        Returns  
        -------  
        numpy.ndarray  
            The intermediate embedding as a NumPy array.  
        """
        return self.nn(torch.tensor(self.X, dtype=torch.float).to(self.device))[-3].detach().cpu().numpy()    
        
    def get_latent(
        self,
    ):
        """  
        Returns the latent representation of the data.  

        Returns  
        -------  
        numpy.ndarray  
            The latent representation as a NumPy array.  
        """
        q_z = self.take_latent(self.X)
        return q_z
