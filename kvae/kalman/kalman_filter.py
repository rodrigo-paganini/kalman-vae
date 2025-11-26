import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
        
class KalmanFilter(nn.Module):
    def __init__(self, std_dyn, std_obs, mu0, Sigma0, dyn_params):
        super().__init__()
        # Dynamics parameter network
        self.dyn_params = dyn_params

        n = dyn_params.A.size(1)
        m = dyn_params.B.size(2)
        p = dyn_params.C.size(1)
        self.n, self.m, self.p = n, m, p

        dev = Sigma0.device
        dtp = Sigma0.dtype

        # NOTE:Fixed Q,R
        self.register_buffer("Q", (std_dyn**2) * torch.eye(n, dtype=dtp, device=dev))  # [n,n]
        self.register_buffer("R", (std_obs**2) * torch.eye(p, dtype=dtp, device=dev))  # [p,p]
        self.register_buffer("I", torch.eye(n, dtype=dtp, device=dev)) 

        # Initial belief  TODO: possibly handle them in filter function?
        self.register_buffer("mu0",    mu0.clone())     # [n]
        self.register_buffer("Sigma0", Sigma0.clone())  # [n,n]
    

    def filter_step(self, mu_t_t, Sigma_t_t, y_t, u_t, A, B, C):
        # [B,n,n], [B,n,m], [B,p,n]
        batch = y_t.size(0)

        Q = self.Q.expand(batch, -1, -1)
        R = self.R.expand(batch, -1, -1)

        # Inputs: mean and covariance of the filtered believe at t-1
        if mu_t_t.dim() == 2:
            mu_tprev_tprev = mu_t_t.unsqueeze(-1) # [B,n,1]
        else:
            mu_tprev_tprev = mu_t_t               # [B,n,1]
        if u_t.dim() == 2:
            u_t = u_t.unsqueeze(-1)               # [B,m,1]
        else:
            u_t = u_t                             # [B,m,1]
        if y_t.dim() == 2:
            y_t = y_t.unsqueeze(-1)               # [B,p,1]
        else:
            y_t = y_t                             # [B,p,1]

        # Prediction step - Compute beliefs at time t given data up to t-1
        Sigma_tprev_tprev = Sigma_t_t       # [B,n,n]
        # [B,n,1] = [B,n,n] @ [B,n,1] + [B,n,n] @ [B,m,1]
        mu_t_tprev    = (A @ mu_tprev_tprev) + (B @ u_t)
        # [B,n,n] = [B,n,n] @ [B,n,n] @ [B,n,n].T + [B,n,n]
        Sigma_t_tprev = A @ Sigma_tprev_tprev @ A.mT + Q

        # Measurement update step
        # Compute beliefs at time t given data up to t
        # Predicted observation
        # [B,p,1] = [B,p,n] @ [B,n,1]
        y_pred = C @ mu_t_tprev
        # Innovation (residual) [B,p,1]
        r = y_t - y_pred
        # Innovation covariance with jitter
        # [B,p,p] = [B,p,n] @ [B,n,n] @ [B,p,n].T + [B,p,p]
        S = C @ Sigma_t_tprev @ C.mT + R
        S = 0.5 * (S + S.mT) 
        # Kalman gain (to avoid the inverse .solve is used)
        # [B,n,p] = [B,n,n] @ [B,p,n].T
        PCT = Sigma_t_tprev @ C.mT
        # [B,n,p] : (solve([B,p,p], [B,n,p].T) -> [B,p,n]).T
        K   = torch.linalg.solve(S, PCT.mT).mT

        # Posterior mean
        # [B,n] = [B,n] + [B,n,p] @ [B,p,1]
        mu_t_t    = mu_t_tprev + K @ r
        # Posterior covariance Joseph form (more stable)
        # [B,n,n] = [B,n,n] - [B,n,p] @ [B,n,p].T
        I_KC = self.I - K @ C
        Sigma_t_t = I_KC @ Sigma_t_tprev @ I_KC.mT + K @ R @ K.mT
        Sigma_t_t = 0.5 * (Sigma_t_t + Sigma_t_t.mT)  # ensures symmetry

        # Return updated beliefs at time t and predicted beliefs at time t
        return mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev, A, B, C


    def filter(self, Y, U):
        """
        The forward pass of the Kalman filter computes the posterior
        at a given t given the observations up to time t.
        p(z_t | y_1:t, u_1:t) = N(z_t| mu_t,t-1 , Sigma_t,t-1)
        NOTE: if the LGSSM is fixed we can use batch solver
        Args:
            Y: [B,T,p] observations
            U: [B,T,m] controls
        Returns:
            mus_filt:    [B,T,n] filtered means (mu_t,t)
            Sigmas_filt: [B,T,n,n] filtered covariances (Sigma_t,t)
            mus_pred:    [B,T,n] predicted means (mu_t,t-1)
            Sigmas_pred: [B,T,n,n] predicted covariances (Sigma_t,t-1)
        """
        batch, T, _ = Y.shape
        mu    = self.mu0.expand(batch, -1)            # [B,n]
        Sigma = self.Sigma0.expand(batch, -1, -1)     # [B,n,n]

        # Get initial A, B, C matrices NOTE: using zeros as priming input
        A, B, C = self.dyn_params.compute_step(torch.zeros((batch, self.p), device=Y.device, dtype=Y.dtype))

        A_list = []
        B_list = []
        C_list = []
        mus_filt = []
        Sigmas_filt = []
        mus_pred = []
        Sigmas_pred = []
        for t in range(T):
            # Get current observation and control
            y_t = Y[:, t]
            u_t = U[:, t]
            
            # Compute filter step
            mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev, _, _, _ = self.filter_step(mu, Sigma, y_t, u_t, A, B, C)

            # Store results
            mus_filt.append(mu_t_t)
            Sigmas_filt.append(Sigma_t_t)
            mus_pred.append(mu_t_tprev)
            Sigmas_pred.append(Sigma_t_tprev)
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)

            # Update for next step
            mu, Sigma = mu_t_t, Sigma_t_t
            # Compute A, B, C at time t (to be used at next step)
            A, B, C = self.dyn_params.compute_step(y_t)  


        return torch.stack(mus_filt, 1), torch.stack(Sigmas_filt, 1), torch.stack(mus_pred, 1), torch.stack(Sigmas_pred, 1), \
                    torch.stack(A_list, 1), torch.stack(B_list, 1), torch.stack(C_list, 1)


    def smooth_step(self, Sigma_t_t, Sigma_tpost_t, Sigma_tpost_T,
                      mu_t_t, mu_tpost_t, mu_tpost_T, A):
        """
        Kalman smoother backward step computes p(z_t | y_1:T, u_1:T) by
        conditioning on past and future data, which reduces uncertainty.
        This function implements the Rauch-Tung-Striebel RTS smoother.
        Args:
            Sigma_t_t:      [B,n,n]   Sigma_t|t   filtered covariance at time t
            Sigma_tpost_t:  [B,n,n]   Sigma_t+1|t predicted covariance at time t+1
            Sigma_tpost_T:  [B,n,n]   Sigma_t|T   smoothed covariance at time t+1
            mu_t_t:         [B,n]     mu_t|t      filtered mean at time t
            mu_tpost_t:     [B,n]     mu_t+1|t    predicted mean at time t+1
            mu_tpost_T:     [B,n]     mu_t|T      smoothed mean at time t+1
        Returns:
            mu_tpost_T:     [B,n]     mu_t|T      smoothed mean at time t+1
            Sigma_tpost_T:  [B,n,n]   Sigma_t|T   smoothed covariance at time t+1
        """
        # Smoother gain J_t = Sigma_{t|t} A^T (Sigma_{t+1|t})^{-1}
        # [B,n,n] 
        J_t = torch.linalg.solve(Sigma_tpost_t.mT, (Sigma_t_t @ A.mT).mT).mT
        # Smoothed mean
        # [B,n] + [B,n] <- ([B,n,n] @ [B,n,1]) 
        mu_tpost_T = mu_t_t + (J_t @ (mu_tpost_T - mu_tpost_t))
        # Smoothed covariance
        Sigma_tpost_T = Sigma_t_t + J_t @ (Sigma_tpost_T - Sigma_tpost_t) @ J_t.mT
        Sigma_tpost_T = 0.5 * (Sigma_tpost_T + Sigma_tpost_T.mT) 
        
        return mu_tpost_T, Sigma_tpost_T    


    def smooth(self, Y, U):
        """

        """
        batch, T, _ = Y.shape

        # First run the filter to get filtered and predicted estimates
        mus_filt, Sigmas_filt, mus_pred, Sigmas_pred, A_list, B_list, C_list = self.filter(Y, U)

        # Initialize smoothing with last filtered estimate
        # mu_T-1|T-1, Sigma_T-1|T-1 
        mu_T, Sigma_T = mus_filt[:, -1], Sigmas_filt[:, -1]

        mus_smooth = [mu_T]
        Sigmas_smooth = [Sigma_T]
        for t in range(T-2, -1, -1):      # t = T-2, …, 0
            mu_t_T, Sigma_t_T = self.smooth_step(
                Sigmas_filt[:, t],  # Sigma_t_t
                Sigmas_pred[:, t+1],# Sigma_tpost_t
                Sigma_T,            # Sigma_tpost_T
                mus_filt[:, t],     # mu_t_t
                mus_pred[:, t+1],   # mu_tpost_t
                mu_T,               # mu_tpost_T
                A_list[:, t]      # A_t
                )
            # Update for next step
            mu_T, Sigma_T = mu_t_T, Sigma_t_T
            # Store results
            mus_smooth.append(mu_t_T)
            Sigmas_smooth.append(Sigma_t_T)

        mus_smooth.reverse()
        Sigmas_smooth.reverse()
        return torch.stack(mus_smooth, 1), torch.stack(Sigmas_smooth, 1), A_list, B_list, C_list


    def _safe_cholesky(self, Sigma, max_tries=5, jitter_init=1e-6): #TODO: unsure 
        dev, dtp = Sigma.device, Sigma.dtype
        n = Sigma.size(-1)

        # Force symmetry
        Sigma = 0.5 * (Sigma + Sigma.mT)

        eye = torch.eye(n, device=dev, dtype=dtp)
        jitter = jitter_init
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(Sigma + jitter * eye)
                return L
            except torch._C._LinAlgError:
                jitter *= 10.0  # increase jitter and try again

        # Final fallback: use only the (clamped) diagonal
        diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        diag = torch.clamp(diag, min=1e-6)
        L = torch.diag_embed(torch.sqrt(diag))
        return L
    

    def elbo(self, mu_t_T, Sigma_t_T, y_t, u_t, A_list, B_list, C_list):
        """
        Compute the Evidence Lower Bound (ELBO)
        Args:
            mu_t_T:     [B,T,n]   smoothed means
            Sigma_t_T:  [B,T,n,n] smoothed covariances
            y_t:        [B,T,p]   observations
            u_t:        [B,T,m]   controls
        Returns:
            elbo:       [B]       ELBO per batch element
        """
        # TODO: review dimensions
        B = y_t.size(0)
        T = y_t.size(1)

        # Reshape mu_t_T to [B,T,n]
        if mu_t_T.dim() == 4:
            mu_t_T = mu_t_T.squeeze(-1)   
        else:
            mu_t_T = mu_t_T         
        # Reshape u_t to [B,T,m,1]
        if u_t.dim() == 3:
            u_t = u_t.unsqueeze(-1)

        # Sample from the smoothed distribution - to ensure positive definiteness
        dev, dtp = self.Q.device, self.Q.dtype
        L = self._safe_cholesky(Sigma_t_T)
        mvn_smooth = MultivariateNormal(mu_t_T, scale_tril=L)
        # Sample using reparameterization trick (keep gradients) [B,T,n,1] 
        z_t_samp = mvn_smooth.rsample()

        z_tprev = z_t_samp[:, :-1].unsqueeze(-1) # z_0 to z_{T-1} 
        # Transition likelihood - prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        z_t = z_t_samp[:, 1:]           # z_1 to z_T  
        u_t = u_t[:, 1:]                # u_1 to u_T  
        Az  = A_list[:, 1:] @ z_tprev   # [B,T-1,n,1]
        Bu  = B_list[:, 1:] @ u_t       # [B,T-1,n,1]
        
        zero_n = torch.zeros(self.n, device=dev, dtype=dtp)
        zero_p = torch.zeros(self.p, device=dev, dtype=dtp)
        mu_trans = (Az + Bu).squeeze(-1)  # [B,T-1,n]
        # TODO: why trick with z_t_transition - mu_transition? 
        mvn_trans = MultivariateNormal(zero_n, self.Q)
        # [B,T-1]
        log_prob_trans = mvn_trans.log_prob((z_t - mu_trans).squeeze(-1)) 

        # Emission likelihood - prod_{t=1}^T p(y_t|z_t) [B,T,p]
        mu_emiss = (C_list @ z_t_samp.unsqueeze(-1)).squeeze(-1)  # [B,T,p]  
        mvn_emiss = MultivariateNormal(zero_p, self.R)
        log_prob_emiss = mvn_emiss.log_prob(y_t - mu_emiss) # [B,T]

        # Initial term [B]
        mvn_init = MultivariateNormal(self.mu0, self.Sigma0)
        log_prob_init = mvn_init.log_prob(z_t_samp[:,0,:])

        # Entropy term [B,T]
        entropy = -mvn_smooth.log_prob(z_t_samp)

        # ELBO computation as sum per frame 
        denom = B * T
        elbo = (
            log_prob_trans.sum() +
            log_prob_emiss.sum() +
            log_prob_init.sum() +
            entropy.sum()
        ) / denom
        return elbo