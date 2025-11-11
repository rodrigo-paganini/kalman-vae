import torch
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

Y, U, X = simulate_rocket_batch(n_batch, T, dt, std_dyn,std_obs)

# Show one sample
sample = 100
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
A0 = torch.randn_like(A_true).repeat(K,1,1)
B0 = torch.randn_like(B_true).repeat(K,1,1)
C0 = torch.randn_like(C_true).repeat(K,1,1)

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
n_epochs = 200
losses = []

for epoch in range(1, n_epochs + 1):
    opt.zero_grad()
    # Reset LSTM state within dynamics parameter network
    dyn_params.reset_state()

    # Forward: smooth then ELBO
    mus_smooth, Sigmas_smooth, A_list, B_list, C_list = kf.smooth(Y, U)            # [B,T,n], [B,T,n,n]
    elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U, A_list, B_list, C_list)        # scalar (per-frame normalized)
    loss = -elbo                                           # maximize ELBO

    # Alignment loss
    Z = mus_smooth.detach().reshape(-1, dyn_params.n)     
    X_true = X.reshape(-1, dyn_params.n)
    S = torch.linalg.lstsq(Z, X_true).solution   
    align_loss = torch.mean((mus_smooth.reshape(-1, dyn_params.n) @ S - X_true)**2)
    loss = -elbo + 1e-2 * align_loss  

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
print("A (learned):\n", dyn_params.A.detach().cpu().numpy())
print("A (true):\n", A_true.detach().cpu().numpy())
print(f"rel ||Â-A|| / ||A|| = {rel_err(dyn_params.A, A_true):.4e}")

print("\nB (learned):\n", dyn_params.B.detach().cpu().numpy())
print("B (true):\n", B_true.detach().cpu().numpy())
print(f"rel ||B̂-B|| / ||B|| = {rel_err(dyn_params.B, B_true):.4e}")
print("\nC (learned):\n", dyn_params.C.detach().cpu().numpy())

print("C (true):\n", C_true.detach().cpu().numpy())
print(f"rel ||Ĉ-C|| / ||C|| = {rel_err(dyn_params.C, C_true):.4e}")

print("Debug")



