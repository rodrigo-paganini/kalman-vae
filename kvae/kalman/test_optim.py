import torch
from torch.distributions import MultivariateNormal
import numpy as np
from kalman_filter import KalmanFilter, DynamicsParameter
import matplotlib.pyplot as plt

def simulate_rocket_batch(B, T, dt, std_dyn=2.0, std_meas=4.0):
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
    U = np.zeros((B, T, 1))  # control

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
        a_meas = a_spec + rng.normal(0, std_dyn, size=T)

        u_meas = a_meas + g    
        z_meas = x[:, 0] + rng.normal(0, std_meas, size=T)

        X[b] = x
        Y[b, :, 0] = z_meas
        U[b, :, 0] = u_meas

    return (torch.tensor(Y, dtype=torch.float32),
            torch.tensor(U, dtype=torch.float32),
            torch.tensor(X, dtype=torch.float32))

# Parameters
K = 1
dt = 0.1
n_batch = 5000
T = 100
std_dyn = 0.5     
std_obs = 4.0     

# Simulate batch data [B,T,1], [B,T,1], [B,T,2]
Y, U, X = simulate_rocket_batch(n_batch, T, dt, std_dyn,std_obs)

# Show one sample
sample = 0
x = X[sample,:,0].numpy()
z_meas = Y[sample,:,0].numpy()
t = np.arange(T)*dt
plt.figure()
plt.plot(t, x, label="Reference (altitude)", color="grey", linestyle="--")
plt.scatter(t, z_meas, label="Observations", marker="x", color="blue", alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Rocket altitude and measurements")
plt.legend()
plt.grid(True)
plt.show()

# Ground truth matrices
A_true = torch.tensor([[1., dt],[0., 1.]], dtype=torch.float32).repeat(K,1,1)
B_true = torch.tensor([[0.5*dt*dt],[dt]], dtype=torch.float32).repeat(K,1,1)
C_true = torch.tensor([[1., 0.]], dtype=torch.float32).repeat(K,1,1)

# Initial random guesses
A0 = torch.eye(2, dtype=torch.float32) + 0.1 * torch.randn(2, 2)
A0 = A0.repeat(K, 1, 1)
B0 = 0.1 * torch.randn(2, 1, dtype=torch.float32)
B0 = B0.repeat(K, 1, 1)
C0 = torch.tensor([[1., 0.]], dtype=torch.float32) + 0.1 * torch.randn(1, 2)
C0 = C0.repeat(K, 1, 1)

mu0    = torch.zeros(2)
Sigma0 = torch.diag(torch.tensor([1., 1.], dtype=torch.float32))

dyn_params = DynamicsParameter(A0.clone(), B0.clone(), C0.clone(), hidden_lstm=32)
kf = KalmanFilter(std_dyn, std_obs, mu0, Sigma0, dyn_params)

# Testing all functions
# Compute filter
mus_filt, Sigmas_filt, mus_pred, Sigmas_pred, A_list, B_list, C_list = kf.filter(Y, U)
# Compute smoother
mus_smooth, Sigmas_smooth, A_list, B_list, C_list = kf.smooth(Y, U)
# Compute ELBO
elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U, A_list, B_list, C_list)
print(f"ELBO: {elbo.item():.2f}")

# Optimizer
opt = torch.optim.Adam(dyn_params.parameters(), lr=1e-2, weight_decay=0.0)
kf.train()
n_epochs = 100
losses = []

for epoch in range(1, n_epochs + 1):
    opt.zero_grad()
    # Reset LSTM state within dynamics parameter network
    dyn_params.reset_state()

    # Forward: smooth then ELBO
    mus_smooth, Sigmas_smooth, A_list, B_list, C_list = kf.smooth(Y, U)            # [B,T,n], [B,T,n,n]
    elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U, A_list, B_list, C_list)        # scalar (per-frame normalized)
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


# Test after training
dyn_params.reset_state()
sample = 10

x_ref_pos = X[sample, :, 0].cpu().numpy()
x_ref_vel = X[sample, :, 1].cpu().numpy()
z_meas    = Y[sample, :, 0].cpu().numpy()

mus_filt, Sigmas_filt, mus_pred, Sigmas_pred, A_list, B_list, C_list = kf.filter(Y, U)
mus_smooth, Sigmas_smooth, A_list, B_list, C_list = kf.smooth(Y, U)
elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U, A_list, B_list, C_list)
print(f"\nAfter training ELBO: {elbo.item():.2f}")

# Evaluate in measurement space: y_hat = C_t @ mu
with torch.no_grad():
    mu_f = mus_filt[sample]        
    P_f  = Sigmas_filt[sample]     
    mu_s = mus_smooth[sample]      
    P_s  = Sigmas_smooth[sample]   
    C_t  = C_list[sample]          

    # Predicted measurement mean over time
    y_mean_f = (C_t @ mu_f).squeeze(-1)      
    y_mean_s = (C_t @ mu_s).squeeze(-1)      

    # Predicted measurement std (channel 0) via C P C^T
    c0 = C_t[:, 0, :]                        
    var_y_f = torch.einsum('ti,tij,tj->t', c0, P_f, c0)  
    var_y_s = torch.einsum('ti,tij,tj->t', c0, P_s, c0)  

    y_mean_f_np = y_mean_f.detach().cpu().numpy()
    y_mean_s_np = y_mean_s.detach().cpu().numpy()
    pos_std_f   = var_y_f.clamp_min(0).sqrt().cpu().numpy()
    pos_std_s   = var_y_s.clamp_min(0).sqrt().cpu().numpy()

# Altitude plot 
plt.figure()
plt.plot(t, x_ref_pos, label="Reference", linestyle="--", color="black")
plt.scatter(t, z_meas, label="Observations", marker="x", alpha=0.5)
plt.plot(t, y_mean_f_np[:, 0], label="KF predicted meas", linewidth=2, color="orange")
plt.fill_between(t,
                 y_mean_f_np[:, 0] - pos_std_f,
                 y_mean_f_np[:, 0] + pos_std_f,
                 alpha=0.2, label="KF ±1std", color="orange")
plt.plot(t, y_mean_s_np[:, 0], label="RTS predicted meas", linewidth=2, color="red")
plt.fill_between(t,
                 y_mean_s_np[:, 0] - pos_std_s,
                 y_mean_s_np[:, 0] + pos_std_s,
                 alpha=0.2, label="RTS ±1std", color="red")
plt.xlabel("Time [s]"); plt.ylabel("Altitude [m]")
plt.title("Altitude: reference vs predicted measurement (KF/RTS)")
plt.legend(); plt.grid(True); plt.show()


# GAP IN TRAJECTOY
sample = 35  
t_gap_start = 30   
t_gap_end   = 60   

t = np.arange(T) * dt
x_ref_pos = X[sample, :, 0].cpu().numpy()
z_meas    = Y[sample, :, 0].cpu().numpy()

mask_obs = np.ones(T, dtype=bool)
mask_obs[t_gap_start:t_gap_end] = False

U_sample       = U[sample:sample+1]              
mus_sample     = mus_smooth[sample:sample+1]     
Sigmas_sample  = Sigmas_smooth[sample:sample+1]  

# We want warmup = t_gap_start steps 
# then free generation until the end of the sequence.
warmup_steps = t_gap_start
gen_steps    = T - warmup_steps   # so total n_steps = T

with torch.no_grad():
    Y_gen, Z_gen, A_gen, B_gen, C_gen = kf.generate_sample(
        U_sample, mus_sample, Sigmas_sample,
        gen_steps=gen_steps,
        warmup_steps=warmup_steps,
        deterministic=True   
    )
    # Y_gen: [1, T, p]
    y_gen_np = Y_gen[0].cpu().numpy()[:, 0]  # [T]

plt.figure()
plt.plot(t, x_ref_pos, label="Reference", linestyle="--", color="black")
plt.scatter(
    t[mask_obs], z_meas[mask_obs],
    label="Observed measurements", marker="x", alpha=0.6
)
plt.plot(
    t[t_gap_start:t_gap_end],
    y_gen_np[t_gap_start:t_gap_end],
    label="Generated (imputation via KF)", color="red"
)
plt.plot(t, y_gen_np, label="Generated full trajectory", color="red", linestyle=":")
plt.axvspan(t[t_gap_start], t[t_gap_end-1], color="grey", alpha=0.15, label="Gap region")
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Gap in trajectory & generative Kalman model imputation")
plt.legend()
plt.grid(True)
plt.show()


# ===============================
# Completely new trajectory generation
# ===============================
n_gen = 100  # length of new sequence

# 1) Generate a *new* physical rocket trajectory using the SAME simulator
Y_real, U_real, X_real = simulate_rocket_batch(
    B=1, T=n_gen, dt=dt,
    std_dyn=std_dyn,
    std_meas=std_obs,
)

# Move to same device/dtype as the KF
Y_real = Y_real.to(kf.mu0.device)
U_real = U_real.to(kf.mu0.device)
X_real = X_real.to(kf.mu0.device)

# Extract "reality"
x_true = X_real[0, :, 0].cpu().numpy()   # true altitude
z_true = Y_real[0, :, 0].cpu().numpy()   # noisy measurements
t_gen  = np.arange(n_gen) * dt

U_new = U_real.clone()  

n_mc = 50
samples = []

with torch.no_grad():
    for _ in range(n_mc):
        Y_mc, _, _, _, _ = kf.generate_sample(
            U_new,
            mus_t=None,         
            Sigmas_t=None,
            gen_steps=n_gen,
            warmup_steps=0,
            deterministic=False  
        )
        samples.append(Y_mc[0, :, 0].cpu().numpy())

samples = np.stack(samples, axis=0)  
mean_y = samples.mean(axis=0)
std_y  = samples.std(axis=0)

plt.figure()
plt.plot(t_gen, x_true, label="True altitude (simulated)", color="black", linestyle="--")
plt.plot(t_gen, mean_y, label="KVAE generated altitude (mean)", color="purple")
plt.fill_between(
    t_gen,
    mean_y - std_y,
    mean_y + std_y,
    alpha=0.2,
    color="purple",
    label="KVAE ±1 std (MC)"
)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("New trajectory: true rocket vs KVAE generative model")
plt.legend()
plt.grid(True)
plt.show()


print("Debug")


