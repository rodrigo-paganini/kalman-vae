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
    

    def filter_step(self, mu_t_t, Sigma_t_t, y_t, u_t, A, B, C, Q, mask_t=None):
        # [B,n,n], [B,n,m], [B,p,n]
        batch = y_t.size(0)

        if Q.dim() == 2:
            Q = Q.expand(batch, -1, -1)
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

        # Shape mask 
        if mask_t is None:
            mask_t = torch.ones(batch, device=y_t.device, dtype=y_t.dtype)
        else:
            mask_t = mask_t.to(device=y_t.device, dtype=y_t.dtype)
            if mask_t.dim() == 0:
                mask_t = mask_t.expand(batch)
        mask_exp = mask_t.view(batch, 1, 1)   

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
        if S.device.type == 'mps':
            S_cpu = S.cpu()
            PCT_cpu = PCT.cpu()
            K = torch.linalg.solve(S_cpu, PCT_cpu.mT).mT.to(S.device)
        else:
            K = torch.linalg.solve(S, PCT.mT).mT
        # Apply mask to Kalman gain
        # For missing values, set to 0 the Kalman gain matrix
        K = mask_exp * K

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


    def filter(self, Y, U, mask=None):
        """
        The forward pass of the Kalman filter computes the posterior
        at a given t given the observations up to time t.
        p(z_t | y_1:t, u_1:t) = N(z_t| mu_t,t-1 , Sigma_t,t-1)
        NOTE: if the LGSSM is fixed we can use batch solver
        Args:
            Y:    [B,T,p] observations
            U:    [B,T,m] controls
            mask: [B, T] with 1 = observed, 0 = missing
        Returns:
            mus_filt:    [B,T,n] filtered means (mu_t,t)
            Sigmas_filt: [B,T,n,n] filtered covariances (Sigma_t,t)
            mus_pred:    [B,T,n] predicted means (mu_t,t-1)
            Sigmas_pred: [B,T,n,n] predicted covariances (Sigma_t,t-1)
        """
        batch, T, _ = Y.shape
        mu    = self.mu0.expand(batch, -1)            # [B,n]
        Sigma = self.Sigma0.expand(batch, -1, -1)     # [B,n,n]

        # Mask shape
        if mask is None:
            mask_tens = torch.ones(batch, T, device=Y.device, dtype=Y.dtype)
        else:
            mask_tens = mask.to(device=Y.device, dtype=Y.dtype)
            if mask_tens.shape != (batch, T):
                mask_tens = mask_tens.view(batch, T)

        use_batch_dyn = self.dyn_params.use_switching_dynamics
        if use_batch_dyn:
            A_seq, B_seq, C_seq, Q_seq = self.dyn_params.compute_batch(
                Y, is_training=self.training
            )
        else:
            A_seq = B_seq = C_seq = Q_seq = None
            y_for_dyn = torch.zeros((batch, self.p), device=Y.device, dtype=Y.dtype)

        A_list = []
        B_list = []
        C_list = []
        mus_filt = []
        Sigmas_filt = []
        mus_pred = []
        Sigmas_pred = []
        for t in range(T):
            # Get current observation and control
            if use_batch_dyn:
                A = A_seq[:, t]
                B = B_seq[:, t]
                C = C_seq[:, t]
                Q_t = Q_seq[:, t]
            else:
                A, B, C = self.dyn_params.compute_step(y_for_dyn)
                Q_t = self.Q
            y_t = Y[:, t]
            u_t = U[:, t]
            m_t = mask_tens[:, t]
            
            # Compute filter step
            mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev, _, _, _ = self.filter_step(
                mu, Sigma, y_t, u_t, A, B, C, Q_t, mask_t=m_t
            )

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
            y_pred = (C @ mu_t_tprev).squeeze(-1)
            m_col = m_t.view(batch, 1)
            y_for_dyn = m_col * y_t + (1.0 - m_col) * y_pred

        # Stack regime sequences
        if not use_batch_dyn:
            state_seq = self.dyn_params.state_seq
            if isinstance(state_seq, list) and len(state_seq) > 0:
                self.dyn_params.state_seq = torch.stack(state_seq, 1)

        return (
            torch.stack(mus_filt, 1),
            torch.stack(Sigmas_filt, 1),
            torch.stack(mus_pred, 1),
            torch.stack(Sigmas_pred, 1),
            torch.stack(A_list, 1),
            torch.stack(B_list, 1),
            torch.stack(C_list, 1),
        )
    

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
        # WORKAROUND for MPS bug
        if Sigma_tpost_t.device.type == 'mps':
            Sigma_tpost_t_cpu = Sigma_tpost_t.cpu()
            temp_cpu = (Sigma_t_t @ A.mT).cpu()
            J_t = torch.linalg.solve(Sigma_tpost_t_cpu.mT, temp_cpu.mT).mT.to(Sigma_tpost_t.device)
        else:
            J_t = torch.linalg.solve(Sigma_tpost_t.mT, (Sigma_t_t @ A.mT).mT).mT
        # Smoothed mean
        # [B,n] + [B,n] <- ([B,n,n] @ [B,n,1]) 
        mu_tpost_T = mu_t_t + (J_t @ (mu_tpost_T - mu_tpost_t))
        # Smoothed covariance
        Sigma_tpost_T = Sigma_t_t + J_t @ (Sigma_tpost_T - Sigma_tpost_t) @ J_t.mT
        Sigma_tpost_T = 0.5 * (Sigma_tpost_T + Sigma_tpost_T.mT) 
        
        return mu_tpost_T, Sigma_tpost_T    


    def smooth(self, Y, U, mask=None):
        """

        """
        batch, T, _ = Y.shape

        # First run the filter to get filtered and predicted estimates
        mus_filt, Sigmas_filt, mus_pred, Sigmas_pred, A_list, B_list, C_list = self.filter(Y, U, mask=mask)

        # Initialize smoothing with last filtered estimate
        # mu_T-1|T-1, Sigma_T-1|T-1 
        mu_T, Sigma_T = mus_filt[:, -1], Sigmas_filt[:, -1]

        mus_smooth = torch.zeros_like(mus_filt)
        mus_smooth[:, -1] = mu_T
        Sigmas_smooth = torch.zeros_like(Sigmas_filt)
        Sigmas_smooth[:, -1] = Sigma_T
        for t in range(T-2, -1, -1):      # t = T-2, …, 0
            A_t = A_list[:, t+1]
            mu_t_T, Sigma_t_T = self.smooth_step(
                Sigmas_filt[:, t],  # Sigma_t_t
                Sigmas_pred[:, t+1],# Sigma_tpost_t
                Sigma_T,            # Sigma_tpost_T
                mus_filt[:, t],     # mu_t_t
                mus_pred[:, t+1],   # mu_tpost_t
                mu_T,               # mu_tpost_T
                A_t                 # A_t 
                )
            # Update for next step
            mu_T, Sigma_T = mu_t_T, Sigma_t_T
            # Store results
            mus_smooth[:, t] = mu_t_T
            Sigmas_smooth[:, t] = Sigma_t_T

        return (
            mus_smooth, Sigmas_smooth,
            mus_filt, Sigmas_filt,
            mus_pred, Sigmas_pred,
            A_list, B_list, C_list,
        )


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
    

    def elbo(self, mu_t_T, Sigma_t_T, y_t, u_t, A_list, B_list, C_list, Q_list=None, mask=None):
        """
        Compute the Evidence Lower Bound (ELBO)
        Args:
            mu_t_T:     [B,T,n]   smoothed means
            Sigma_t_T:  [B,T,n,n] smoothed covariances
            y_t:        [B,T,p]   observations
            u_t:        [B,T,m]   controls
            Q_list:     [B,T,n,n] process noise covariances
            mask:       [B,T]     with 1 = observed, 0 = missing
        Returns:
            elbo:       scalar    ELBO averaged over batch
        """
        # TODO: review dimensions
        B = y_t.size(0)
        T = y_t.size(1)

        # Observation mask
        if mask is None:
            mask_tens = torch.ones(B, T, device=y_t.device, dtype=y_t.dtype)
        else:
            mask_tens = mask.to(device=y_t.device, dtype=y_t.dtype)
            if mask_tens.shape != (B, T):
                mask_tens = mask_tens.view(B, T)

        dev, dtp = y_t.device, y_t.dtype

        # Reshape mu_t_T to [B,T,n]
        if mu_t_T.dim() == 4:
            mu_t_T = mu_t_T.squeeze(-1)   
        else:
            mu_t_T = mu_t_T         
        # Reshape u_t to [B,T,m,1]
        if u_t.dim() == 3:
            u_t = u_t.unsqueeze(-1)

        # Process noise
        if Q_list is None:
            Q_list = getattr(self.dyn_params, "Q_seq", None)
        if Q_list is None:
            Q_list = self.Q.to(device=dev, dtype=dtp).expand(B, T, -1, -1)

        # Sample from the smoothed distribution - to ensure positive definiteness
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

        Q_trans = Q_list[:, 1:]  # [B,T-1,n,n]
        L_Q = self._safe_cholesky(Q_trans)
        # TODO: why trick with z_t_transition - mu_transition? 
        mvn_trans = MultivariateNormal(zero_n, scale_tril=L_Q.reshape(-1, self.n, self.n))
        # [B,T-1]
        log_prob_trans = mvn_trans.log_prob((z_t - mu_trans).reshape(-1, self.n)).view(B, T-1)

        # Emission likelihood - prod_{t=1}^T p(y_t|z_t) [B,T,p]
        mu_emiss = (C_list @ z_t_samp.unsqueeze(-1)).squeeze(-1)  # [B,T,p]  
        mvn_emiss = MultivariateNormal(zero_p, self.R)
        log_prob_emiss = mvn_emiss.log_prob(y_t - mu_emiss) # [B,T]

        # Mask out missing observations in the emission term
        log_prob_emiss = log_prob_emiss * mask_tens

        # Initial term [B]
        mvn_init = MultivariateNormal(self.mu0, self.Sigma0)
        log_prob_init = mvn_init.log_prob(z_t_samp[:,0,:])
        if self.dyn_params.use_switching_dynamics:
            log_qseq, log_pseq = self.dyn_params.elbo_terms()
        else:
            zeros = torch.zeros(B, T, device=y_t.device, dtype=y_t.dtype)
            log_qseq, log_pseq = zeros, zeros

        # Entropy term [B,T]
        entropy = -mvn_smooth.log_prob(z_t_samp)

        # ELBO computation normalized by total observed frames
        num_el = mask_tens.sum().clamp(min=1.0)
        elbo = (
            log_prob_trans.sum() +
            log_prob_emiss.sum() +
            log_prob_init.sum() +
            log_qseq.sum() +
            log_pseq.sum() +
            entropy.sum()
        ) / num_el
        return elbo
