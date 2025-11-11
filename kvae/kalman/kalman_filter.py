import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


class DynamicsParameter(nn.Module):
    def __init__(self, A, B, C, hidden_lstm=32):
        """
        A0: [K, n,n] initial state transition matrix
        B0: [K, n,m] initial control input matrix
        C0: [K, p,n] initial observation matrix
        """
        super().__init__()

        K = A.size(0) 
        n = A.size(1)
        m = B.size(2)
        p = C.size(1)
        self.n, self.m, self.p = n, m, p

        self.A = nn.Parameter(A.clone()) 
        self.B = nn.Parameter(B.clone())
        self.C = nn.Parameter(C.clone())

        # LSTM hidden state TODO: hardcoded so far
        self.lstm_state = None 

        self.lstm = nn.LSTM(
            input_size=self.p, 
            hidden_size=hidden_lstm, 
            num_layers=1, 
            batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_lstm, hidden_lstm), 
            nn.Tanh(),
            )
        
        self.head_w = nn.Linear(hidden_lstm, K)

    def reset_state(self):
        self.lstm_state = None

    def compute_step(self, a_tprev):
        # If a_tprev is [B,_], unsqueeze to [B,1,_]
        if a_tprev.dim() == 2: 
            a_tprev = a_tprev.unsqueeze(1)  

        # Apply LSTM, MLP and head
        h, new_state = self.lstm(a_tprev, self.lstm_state)
        f = self.mlp(h)                 
        self.w = torch.softmax(self.head_w(f), dim=-1).squeeze(1)  
        
        # Update LSTM state
        self.lstm_state = new_state

        # Compute weighted sum of matrices
        A = torch.einsum('bk,kij->bij', self.w, self.A)
        B = torch.einsum('bk,knm->bnm', self.w, self.B)
        C = torch.einsum('bk,kpn->bpn', self.w, self.C)

        return A, B, C 
        

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
    
    def filter_step(self, mu_t_t, Sigma_t_t, y_t, u_t):
        # [B,n,n], [B,n,m], [B,p,n]
        batch = y_t.size(0)
        A, B, C = self.dyn_params.compute_step(y_t)  

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

        # Reset dynamic parameter network
        self.dyn_params.reset_state()

        A_list = []
        B_list = []
        C_list = []
        mus_filt = []
        Sigmas_filt = []
        mus_pred = []
        Sigmas_pred = []
        for t in range(T):
            mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev, A, B, C = self.filter_step(mu, Sigma, Y[:, t], U[:, t])
            # Update for next step
            mu, Sigma = mu_t_t, Sigma_t_t
            # Store results
            mus_filt.append(mu_t_t)
            Sigmas_filt.append(Sigma_t_t)
            mus_pred.append(mu_t_tprev)
            Sigmas_pred.append(Sigma_t_tprev)
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)

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
                A_list[:, t+1]      # A_t
                            )
            # Update for next step
            mu_T, Sigma_T = mu_t_T, Sigma_t_T
            # Store results
            mus_smooth.append(mu_t_T)
            Sigmas_smooth.append(Sigma_t_T)

        mus_smooth.reverse()
        Sigmas_smooth.reverse()
        return torch.stack(mus_smooth, 1), torch.stack(Sigmas_smooth, 1), A_list, B_list, C_list


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
        if u_t.dim() == 3: # [B,T,m,1]  
            u_t = u_t.unsqueeze(-1)      
        else:   
            u_t = u_t 

        # Sample from the smoothed distribution - to ensure positive definiteness
        L = torch.linalg.cholesky(Sigma_t_T)
        mvn_smooth = MultivariateNormal(mu_t_T, scale_tril=L)
        # Sample using reparameterization trick (keep gradients) [B,T,n,1]
        z_t = mvn_smooth.rsample() 

        # Transition likelihood - prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        Az = A_list[:, 1:] @ z_t[:, :-1].unsqueeze(-1)  # [B,T-1,n,1]
        Bu = (B_list[:, 1:, :, :] @ u_t[:, 1:, :])      # [B,T-1,n,1]
        
        mu_trans = (Az + Bu).squeeze(-1)  # [B,T-1,n]
        # TODO: why trick with z_t_transition - mu_transition? 
        mvn_trans = MultivariateNormal(torch.zeros(self.n), self.Q)
        # [B,T-1]
        log_prob_trans = mvn_trans.log_prob((z_t[:, :-1, :] - mu_trans).squeeze(-1)) 

        # Emission likelihood - prod_{t=1}^T p(y_t|z_t) [B,T,p]
        mu_emiss = (C_list @ z_t.unsqueeze(-1)).squeeze(-1)  # [B,T,p]  
        mvn_emiss = MultivariateNormal(torch.zeros(self.p), self.R)
        log_prob_emiss = mvn_emiss.log_prob(y_t - mu_emiss) # [B,T]

        # Initial term [B]
        mvn_init = MultivariateNormal(self.mu0, self.Sigma0)
        log_prob_init = mvn_init.log_prob(z_t[:,0,:])

        # Entropy term [B,T]
        entropy = -mvn_smooth.log_prob(z_t)

        # ELBO computation as sum per frame 
        denom = B * T
        elbo = (
            log_prob_trans.sum() +
            log_prob_emiss.sum() +
            log_prob_init.sum() +
            entropy.sum()
        ) / denom
        return elbo