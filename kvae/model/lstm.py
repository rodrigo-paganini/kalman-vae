
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvae.model.config import KVAEConfig


class DynamicsParameterNetwork(nn.Module):
    """
    LSTM network that outputs mixture weights alpha_t for the LGSSM
    alpha_t depends on history a_{0:t-1}
    """
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.a_dim,
            hidden_size=config.dynamics_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.fc_alpha = nn.Linear(config.dynamics_hidden_dim, config.num_modes)
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture weights for each time step
        
        Args:
            a: Latent encodings [batch, T, a_dim]
        Returns:
            alpha: Mixture weights [batch, T, K]
        """
        batch_size, T, _ = a.shape
        
        # Shift a by one time step (alpha_t depends on a_{0:t-1})
        a_shifted = torch.zeros_like(a)
        a_shifted[:, 1:] = a[:, :-1]
        
        # LSTM forward pass
        h, _ = self.lstm(a_shifted)
        
        # Compute mixture weights with softmax
        alpha = F.softmax(self.fc_alpha(h), dim=-1)
        
        return alpha