from torch import nn
import torch

from typing import Optional, Tuple, Dict

from kvae.model.config import KVAEConfig

#TODO: THIS FUNCTION IS NOT USED
class LGSSM(nn.Module):
    """
    Linear Gaussian State Space Model with time-varying parameters
    Models: z_t = A_t * z_{t-1} + B_t * u_t + eps_z
            a_t = C_t * z_t + eps_a
    """
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        self.a_dim = config.a_dim
        self.z_dim = config.z_dim
        self.K = config.num_modes
        
        # Learn K sets of transition, control, and emission matrices
        self.A_matrices = nn.Parameter(torch.randn(self.K, self.z_dim, self.z_dim) * 0.1)
        self.B_matrices = nn.Parameter(torch.randn(self.K, self.z_dim, self.z_dim) * 0.1)
        self.C_matrices = nn.Parameter(torch.randn(self.K, self.a_dim, self.z_dim) * 0.1)
        
        # Process and measurement noise (diagonal covariances)
        self.log_Q = nn.Parameter(torch.zeros(self.z_dim))
        self.log_R = nn.Parameter(torch.zeros(self.a_dim))
        
        # Initial state
        self.z0_mean = nn.Parameter(torch.zeros(self.z_dim))
        self.z0_logvar = nn.Parameter(torch.zeros(self.z_dim))
    
    @property
    def Q(self):
        """Process noise covariance"""
        return torch.diag(torch.exp(self.log_Q))
    
    @property
    def R(self):
        """Measurement noise covariance"""
        return torch.diag(torch.exp(self.log_R))
    
    def get_matrices(self, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute time-varying matrices as weighted combination of K modes
        
        Args:
            alpha: Mixture weights [batch, T, K]
        Returns:
            A_t, B_t, C_t: Time-varying matrices
        """
        batch_size, T, K = alpha.shape
        
        # Weighted combination: sum_k alpha_k * Matrix_k
        # A_t: [batch, T, z_dim, z_dim]
        A_t = torch.einsum('btk,kij->btij', alpha, self.A_matrices)
        B_t = torch.einsum('btk,kij->btij', alpha, self.B_matrices)
        C_t = torch.einsum('btk,kij->btij', alpha, self.C_matrices)
        
        return A_t, B_t, C_t
    
    def kalman_filter(self, a_obs: torch.Tensor, alpha: torch.Tensor,
                     u: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass: Kalman filtering
        
        Args:
            a_obs: Observed encodings [batch, T, a_dim]
            alpha: Mixture weights [batch, T, K]
            u: Control inputs [batch, T, z_dim] (optional)
        
        Returns:
            Dictionary with filtered means and covariances
        """
        batch_size, T, _ = a_obs.shape
        device = a_obs.device
        
        if u is None:
            u = torch.zeros(batch_size, T, self.z_dim, device=device)
        
        # Get time-varying matrices
        A_t, B_t, C_t = self.get_matrices(alpha)
        Q = self.Q
        R = self.R
        
        # Initialize
        z_mean = self.z0_mean.unsqueeze(0).expand(batch_size, -1)
        z_cov = torch.diag(torch.exp(self.z0_logvar)).unsqueeze(0).expand(batch_size, -1, -1)
        
        filtered_means = []
        filtered_covs = []
        
        for t in range(T):
            # Prediction step
            if t > 0:
                z_mean = torch.bmm(A_t[:, t], z_mean.unsqueeze(-1)).squeeze(-1) + \
                        torch.bmm(B_t[:, t], u[:, t].unsqueeze(-1)).squeeze(-1)
                z_cov = torch.bmm(torch.bmm(A_t[:, t], z_cov), A_t[:, t].transpose(1, 2)) + Q
            
            # Update step
            a_pred = torch.bmm(C_t[:, t], z_mean.unsqueeze(-1)).squeeze(-1)
            S = torch.bmm(torch.bmm(C_t[:, t], z_cov), C_t[:, t].transpose(1, 2)) + R
            
            # Kalman gain
            K = torch.bmm(torch.bmm(z_cov, C_t[:, t].transpose(1, 2)), 
                         torch.inverse(S))
            
            # Update
            innovation = a_obs[:, t] - a_pred
            z_mean = z_mean + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
            z_cov = z_cov - torch.bmm(torch.bmm(K, C_t[:, t]), z_cov)
            
            filtered_means.append(z_mean)
            filtered_covs.append(z_cov)
        
        return {
            'means': torch.stack(filtered_means, dim=1),
            'covs': torch.stack(filtered_covs, dim=1)
        }
    
    def kalman_smoother(self, filtered_means: torch.Tensor,
                       filtered_covs: torch.Tensor,
                       alpha: torch.Tensor,
                       u: Optional[torch.Tensor] = None) -> Dict:
        """
        Backward pass: Kalman smoothing
        
        Args:
            filtered_means: From forward pass [batch, T, z_dim]
            filtered_covs: From forward pass [batch, T, z_dim, z_dim]
            alpha: Mixture weights [batch, T, K]
            u: Control inputs [batch, T, z_dim] (optional)
        
        Returns:
            Dictionary with smoothed means and covariances
        """
        batch_size, T, z_dim = filtered_means.shape
        device = filtered_means.device
        
        if u is None:
            u = torch.zeros(batch_size, T, self.z_dim, device=device)
        
        # Get time-varying matrices
        A_t, B_t, _ = self.get_matrices(alpha)
        Q = self.Q
        
        # Initialize with filtered estimates
        smoothed_means = [filtered_means[:, -1]]
        smoothed_covs = [filtered_covs[:, -1]]
        
        # Backward recursion
        for t in range(T - 2, -1, -1):
            # Predicted next state
            z_pred = torch.bmm(A_t[:, t+1], filtered_means[:, t].unsqueeze(-1)).squeeze(-1) + \
                    torch.bmm(B_t[:, t+1], u[:, t+1].unsqueeze(-1)).squeeze(-1)
            P_pred = torch.bmm(torch.bmm(A_t[:, t+1], filtered_covs[:, t]), 
                              A_t[:, t+1].transpose(1, 2)) + Q
            
            # Smoother gain
            J = torch.bmm(torch.bmm(filtered_covs[:, t], A_t[:, t+1].transpose(1, 2)),
                         torch.inverse(P_pred))
            
            # Smoothed estimates
            z_smooth = filtered_means[:, t] + \
                      torch.bmm(J, (smoothed_means[0] - z_pred).unsqueeze(-1)).squeeze(-1)
            P_smooth = filtered_covs[:, t] + \
                      torch.bmm(torch.bmm(J, smoothed_covs[0] - P_pred), J.transpose(1, 2))
            
            smoothed_means.insert(0, z_smooth)
            smoothed_covs.insert(0, P_smooth)
        
        return {
            'means': torch.stack(smoothed_means, dim=1),
            'covs': torch.stack(smoothed_covs, dim=1)
        }