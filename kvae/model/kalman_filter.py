import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


class KalmanFilter(nn.Module):
    def __init__(self, A, B, C, q_scale, r_scale, mu0, Sigma0):
        super().__init__()
        n = A.size(0)
        m = B.size(1)
        p = C.size(0)
        self.n, self.m, self.p = n, m, p

        # Trainable parameters
        # n = state dim
        # m = control dim
        # p = observation dim
        self.A = nn.Parameter(A.clone()) # [n,n]
        self.B = nn.Parameter(B.clone()) # [n,m]
        self.C = nn.Parameter(C.clone()) # [p,n]

        # NOTE:Fixed Q,R
        self.register_buffer("Q", (q_scale**2) * torch.eye(n)) # [n,n]
        self.register_buffer("R", (r_scale**2) * torch.eye(p)) # [p,p]

        # Initial belief  TODO: possibly handle them in filter function?
        self.register_buffer("mu0",    mu0.clone())     # [n]
        self.register_buffer("Sigma0", Sigma0.clone())  # [n,n]

        self.register_buffer("I", torch.eye(n, dtype=Sigma0.dtype, device=Sigma0.device))

    
    def filter_step(self, mu_t_t, Sigma_t_t, y_t, u_t):
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R

        # Inputs: mean and covariance of the filtered believe at t-1
        mu_tprev_tprev = mu_t_t       # [B,n]
        Sigma_tprev_tprev = Sigma_t_t # [B,n,n]

        # Prediction step - Compute beliefs at time t given data up to t-1
        # [B,n] = [B,n] @ [n,n].T + [B,m] @ [n,m].T
        mu_t_tprev    = (mu_tprev_tprev @ A.T) + (u_t @ B.T)
        # [B,n,n] = [n,n] @ [B,n,n] @ [n,n].T + [n,n]
        Sigma_t_tprev = A @ Sigma_tprev_tprev @ A.mT + Q

        # Measurement update step
        # Compute beliefs at time t given data up to t
        # Predicted observation
        # [B,p] = [B,n] @ [p,n].T
        y_pred = mu_t_tprev @ C.T
        # Innovation (residual)
        # [B,p,1] = ([B,p] - [B,p]).unsqueeze(-1)
        r = (y_t - y_pred).unsqueeze(-1)
        # Innovation covariance with jitter
        # [B,p,p] = [p,n] @ [B,n,n] @ [p,n].T + [p,p]
        S = C @ Sigma_t_tprev @ C.mT + R
        S = 0.5 * (S + S.mT) 
        # Kalman gain (to avoid the inverse .solve is used)
        # [B,n,p] = [B,n,n] @ [p,n].T
        PCT = Sigma_t_tprev @ C.mT
        # [B,n,p] : (solve([B,p,p], [B,n,p].T) -> [B,p,n]).T
        K   = torch.linalg.solve(S, PCT.mT).mT

        # Posterior mean
        # [B,n] = [B,n] + ([B,n,p] @ [B,p,1])
        mu_t_t    = mu_t_tprev + (K @ r).squeeze(-1)
        # Posterior covariance Joseph form (more stable)
        # [B,n,n] = [B,n,n] - [B,n,p] @ [B,n,p].T
        I_KC = self.I - K @ C
        Sigma_t_t = I_KC @ Sigma_t_tprev @ I_KC.mT + K @ R @ K.mT
        Sigma_t_t = 0.5 * (Sigma_t_t + Sigma_t_t.mT)  # ensures symmetry

        # Return updated beliefs at time t and predicted beliefs at time t
        return mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev
    

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

        mus_filt = []
        Sigmas_filt = []
        mus_pred = []
        Sigmas_pred = []
        for t in range(T):
            mu_t_t, Sigma_t_t, mu_t_tprev, Sigma_t_tprev = self.filter_step(mu, Sigma, Y[:, t], U[:, t])
            # Update for next step
            mu, Sigma = mu_t_t, Sigma_t_t
            # Store results
            mus_filt.append(mu_t_t)
            Sigmas_filt.append(Sigma_t_t)
            mus_pred.append(mu_t_tprev)
            Sigmas_pred.append(Sigma_t_tprev)

        return torch.stack(mus_filt, 1), torch.stack(Sigmas_filt, 1), torch.stack(mus_pred, 1), torch.stack(Sigmas_pred, 1)


    def smooth_step(self, Sigma_t_t, Sigma_tpost_t, Sigma_tpost_T,
                      mu_t_t, mu_tpost_t, mu_tpost_T):
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
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R

        # Smoother gain J_t = Sigma_{t|t} A^T (Sigma_{t+1|t})^{-1}
        # [B,n,n] 
        J_t = torch.linalg.solve(Sigma_tpost_t.mT, (Sigma_t_t @ A.mT).mT).mT
        # Smoothed mean
        # [B,n] + [B,n] <- ([B,n,n] @ [B,n,1]) 
        mu_tpost_T = mu_t_t + (J_t @ (mu_tpost_T - mu_tpost_t).unsqueeze(-1)).squeeze(-1)
        # Smoothed covariance
        Sigma_tpost_T = Sigma_t_t + J_t @ (Sigma_tpost_T - Sigma_tpost_t) @ J_t.mT
        Sigma_tpost_T = 0.5 * (Sigma_tpost_T + Sigma_tpost_T.mT) 
        
        return mu_tpost_T, Sigma_tpost_T    


    def smooth(self, Y, U):
        """

        """
        batch, T, _ = Y.shape

        # First run the filter to get filtered and predicted estimates
        mus_filt, Sigmas_filt, mus_pred, Sigmas_pred = self.filter(Y, U)

        # Initialize smoothing with last filtered estimate
        # mu_T-1|T-1, Sigma_T-1|T-1 
        mu_T, Sigma_T = mus_filt[:, -1], Sigmas_filt[:, -1]

        mus_smooth, Sigmas_smooth = [mu_T], [Sigma_T]
        for t in range(T-2, -1, -1):      # t = T-2, …, 0
            mu_t_T, Sigma_t_T = self.smooth_step(
                Sigmas_filt[:, t],  # Sigma_t_t
                Sigmas_pred[:, t+1],# Sigma_tpost_t
                Sigma_T,            # Sigma_tpost_T
                mus_filt[:, t],     # mu_t_t
                mus_pred[:, t+1],   # mu_tpost_t
                mu_T                # mu_tpost_T
                            )
            # Update for next step
            mu_T, Sigma_T = mu_t_T, Sigma_t_T
            # Store results
            mus_smooth.append(mu_t_T)
            Sigmas_smooth.append(Sigma_t_T)

        mus_smooth.reverse()
        Sigmas_smooth.reverse()
        return torch.stack(mus_smooth, 1), torch.stack(Sigmas_smooth, 1)

    def elbo(self, mu_t_T, Sigma_t_T, y_t, u_t):
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
        B, T, n = mu_t_T.shape
        
        # Sample from the smoothed distribution - to ensure positive definiteness
        L = torch.linalg.cholesky(Sigma_t_T)
        mvn_smooth = MultivariateNormal(mu_t_T, scale_tril=L)
        # Sample using reparameterization trick (keep gradients) [B,T,n]
        z_t = mvn_smooth.rsample()  

        # Transition likelihood - prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        Az = z_t[:, :-1, :] @ self.A.mT
        Bu = u_t[:,  1:, :] @ self.B.mT
        mu_trans = Az + Bu  # [B,T-1,n]
        # TODO: why trick with z_t_transition - mu_transition? 
        mvn_trans = MultivariateNormal(torch.zeros(self.n), self.Q)
        # [B,T-1]
        log_prob_trans = mvn_trans.log_prob(z_t[:, :-1, :] - mu_trans) 

        # Emission likelihood - prod_{t=1}^T p(y_t|z_t) [B,T,p]
        mu_emiss = z_t @ self.C.mT 
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
    

## =================   TEST   ================= ##
if __name__ == '__main__':
    torch.manual_seed(0)

    # ----- Rocket example - Simple Kalman filter  -----
    dt = 0.1
    g  = -9.81
    T  = 10.0
    N  = int(T/dt)
    t  = np.arange(N)*dt

    # Generate data
    x = np.zeros((N,2))
    for n in range(N-1):
        a = (20.0 if t[n] < 6.0 else 0.0) + g
        x[n+1,0] = x[n,0] + x[n,1]*dt + 0.5*a*dt*dt
        x[n+1,1] = x[n,1] + a*dt

    a_spec = np.r_[((x[1:,1]-x[:-1,1])/dt - g)[0], (x[1:,1]-x[:-1,1])/dt - g]
    u_meas = a_spec + g + np.random.randn(N)*2.8    
    z_meas = x[:,0] + np.random.randn(N)*4.0        

    # Problem setup
    A = torch.tensor([[1.0, dt],
                      [0.0, 1.0]], dtype=torch.float32)
    B = torch.tensor([[0.5*dt**2],
                      [dt]], dtype=torch.float32)
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    eps_a = 2
    Q = (eps_a**2) * torch.tensor([[dt**4/4, dt**3/2],
                                   [dt**3/2, dt**2]], dtype=torch.float32)
    R = torch.tensor([[3.0**2]], dtype=torch.float32)

    mu0    = torch.zeros(2, dtype=torch.float32)                     
    Sigma0 = torch.diag(torch.tensor([1000.0, 1000.0], dtype=torch.float32))

    kf = KalmanFilter(A, B, C, eps_a, 3.0, mu0, Sigma0)

    model_dtype = A.dtype
    model_device = A.device
    Y = torch.from_numpy(z_meas).to(dtype=model_dtype, device=model_device).unsqueeze(0).unsqueeze(-1)  # [1,N,1]
    U = torch.from_numpy(u_meas).to(dtype=model_dtype, device=model_device).unsqueeze(0).unsqueeze(-1)  # [1,N,1]

    # # Compute filter
    # mus_filt, Sigmas_filt, mus_pred, Sigmas_pred = kf.filter(Y, U)
    # # Compute smoother
    # mus_smooth, Sigmas_smooth = kf.smooth(Y, U)
    # # Compute ELBO
    # elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U)

    # print(f"ELBO: {elbo.item():.2f}")

    # # Convert tensors to numpy
    # mu_np = mus_filt[0].detach().cpu().numpy()
    # P_np  = Sigmas_filt[0].detach().cpu().numpy()
    # pos_mean = mu_np[:, 0]
    # vel_mean = mu_np[:, 1]
    # pos_std  = np.sqrt(P_np[:, 0, 0])
    # vel_std  = np.sqrt(P_np[:, 1, 1])
    # mu_f_s  = mus_smooth[0].detach().cpu().numpy()       
    # P_f_s   = Sigmas_smooth[0].detach().cpu().numpy()    
    # pos_mean_s = mu_f_s[:, 0]
    # vel_mean_s = mu_f_s[:, 1]
    # pos_std_s  = np.sqrt(P_f_s[:, 0, 0])
    # vel_std_s  = np.sqrt(P_f_s[:, 1, 1])

    # # --- Altitude plot ---
    # plt.figure()
    # plt.plot(t, x[:,0], label="reference (altitude)")
    # plt.plot(t, pos_mean, label="KF mean (altitude)")
    # plt.plot(t, pos_mean_s, label="RTS smoothed mean")
    # plt.fill_between(t, pos_mean - pos_std, pos_mean + pos_std, alpha=0.2, label="±1σ")
    # plt.fill_between(t, pos_mean_s - pos_std_s, pos_mean_s + pos_std_s, alpha=0.15, label="smooth ±1σ")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Altitude [m]")
    # plt.title("Altitude: reference vs KF mean ±1σ")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # --- Velocity plot ---
    # plt.figure()
    # plt.plot(t, x[:,1], label="reference (velocity)")
    # plt.plot(t, vel_mean, label="KF mean (velocity)")
    # plt.plot(t, vel_mean_s, label="RTS smoothed mean")
    # plt.fill_between(t, vel_mean - vel_std, vel_mean + vel_std, alpha=0.2, label="±1σ")
    # plt.fill_between(t, vel_mean_s - vel_std_s, vel_mean_s + vel_std_s, alpha=0.15, label="smooth ±1σ")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Velocity [m/s]")
    # plt.title("Velocity: reference vs KF mean ±1σ")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    def simulate_rocket_batch(B, T, dt, sigma_a_meas=2.0, sigma_z=4.0):
        """
        T: number of time steps
        dt: sample time
        g: gravity (negative if up is positive)
        Returns Y:[B,T,1], U:[B,T,1], X:[B,T,2]
        """
        rng = np.random.default_rng(0)
        t = np.arange(T) * dt
        g = -9.81

        X = np.zeros((B, T, 2))  # [pos, vel]
        Y = np.zeros((B, T, 1))  # altitude
        U = np.zeros((B, T, 1))  # control = a_meas + g

        for b in range(B):
            thrust = rng.uniform(15.0, 25.0)   # m/s^2
            burn   = rng.uniform(3.0, 7.0)     # s
            x0, v0 = rng.normal(0.0, 2.0), rng.normal(0.0, 2.0)

            x = np.zeros((T, 2))
            x[0] = [x0, v0]
            for n in range(T - 1):
                # engine cutoff sequence
                a_true = (thrust if t[n] < burn else 0.0) + g
                x[n+1, 0] = x[n, 0] + x[n, 1]*dt + 0.5 * a_true * dt**2
                x[n+1, 1] = x[n, 1] + a_true * dt

            # accelerometer specific force and noisy measurement
            a_spec = (x[1:, 1] - x[:-1, 1]) / dt - g
            a_spec = np.r_[a_spec[0], a_spec]  # pad first sample
            a_meas = a_spec + rng.normal(0, sigma_a_meas, size=T)

            u_meas = a_meas + g                      # control = a_meas + g ≈ a_true + noise
            z_meas = x[:, 0] + rng.normal(0, sigma_z, size=T)

            X[b] = x
            Y[b, :, 0] = z_meas
            U[b, :, 0] = u_meas

        return (torch.tensor(Y, dtype=torch.float32),
                torch.tensor(U, dtype=torch.float32),
                torch.tensor(X, dtype=torch.float32))
    

    q_scale = 0.5     # sqrt of process magnitude 
    r_scale = 4.0     # sqrt of emiss noise      
    
    # Data (batch, time)
    dt = 0.1
    n_batch, T = 5000, 100
    Y, U, X = simulate_rocket_batch(n_batch, T, dt, q_scale=2.0, r_scale=4.0)
        

    A_true = torch.tensor([[1., dt],[0., 1.]], dtype=torch.float32)
    B_true = torch.tensor([[0.5*dt*dt],[dt]], dtype=torch.float32)
    C_true = torch.tensor([[1., 0.]], dtype=torch.float32)

    # Initial random guesses
    A0 = torch.randn_like(A_true)
    B0 = torch.randn_like(B_true)
    C0 = torch.randn_like(C_true)

    mu0    = torch.zeros(2)
    Sigma0 = torch.diag(torch.tensor([100., 100.], dtype=torch.float32))

    kf = KalmanFilter(A0.clone(), B0.clone(), C0.clone(), q_scale, r_scale, mu0, Sigma0)
    
    # Testing all functions
    # Compute filter
    mus_filt, Sigmas_filt, mus_pred, Sigmas_pred = kf.filter(Y, U)
    # Compute smoother
    mus_smooth, Sigmas_smooth = kf.smooth(Y, U)
    # Compute ELBO
    elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U)

    # Optimizer
    opt = torch.optim.Adam([kf.A, kf.B, kf.C], lr=1e-2, weight_decay=0.0)
    kf.train()
    n_epochs = 200
    losses = []

    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()

        # Forward: smooth then ELBO
        mus_smooth, Sigmas_smooth = kf.smooth(Y, U)            # [B,T,n], [B,T,n,n]
        elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U)        # scalar (per-frame normalized)
        loss = -elbo                                           # maximize ELBO

        # Backward + step
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"[epoch {epoch:03d}]  loss = {loss.item():.4f}  ELBO = {elbo.item():.4f}")

    # Print results 
    def rel_err(est, true):
        return (torch.linalg.norm(est - true) / torch.linalg.norm(true)).item()

    # TODO: not matching, differences in learned base
    print("\nLearned parameters vs ground truth:")
    print("A (learned):\n", kf.A.detach().cpu().numpy())
    print("A (true):\n", A.detach().cpu().numpy())
    print(f"rel ||Â-A|| / ||A|| = {rel_err(kf.A, A):.4e}")

    print("\nB (learned):\n", kf.B.detach().cpu().numpy())
    print("B (true):\n", B.detach().cpu().numpy())
    print(f"rel ||B̂-B|| / ||B|| = {rel_err(kf.B, B):.4e}")

    print("\nC (learned):\n", kf.C.detach().cpu().numpy())
    print("C (true):\n", C.detach().cpu().numpy())
    print(f"rel ||Ĉ-C|| / ||C|| = {rel_err(kf.C, C):.4e}")

    print("Debug")



