import numpy as np
import torch
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter, DynamicsParameter

# ----- Rocket example - Simple Kalman filter  -----
dt = 0.1
g  = -9.81
T  = 10.0
N  = int(T/dt)
t  = np.arange(N)*dt
std_obs = 4.0 # 
std_dyn = 2.0 # Q 

# Generate data
x = np.zeros((N,2))
for n in range(N-1):
    a = (20.0 if t[n] < 6.0 else 0.0) + g
    x[n+1,0] = x[n,0] + x[n,1]*dt + 0.5*a*dt*dt
    x[n+1,1] = x[n,1] + a*dt

a_spec = np.r_[((x[1:,1]-x[:-1,1])/dt - g)[0], (x[1:,1]-x[:-1,1])/dt - g]
u_meas = a_spec + g + np.random.randn(N)*std_dyn**2    
z_meas = x[:,0] + np.random.randn(N)*std_obs**2

# plt.figure()
# plt.plot(t, x[:,0], label="Reference (altitude)", color="grey", linestyle="--")
# plt.scatter(t, z_meas, label="Observations", marker="x", color="blue", alpha=0.5)
# plt.xlabel("Time [s]")
# plt.ylabel("Altitude [m]")
# plt.title("Rocket altitude and measurements")
# plt.legend()
# plt.grid(True)
# plt.show()

# Problem setup
A = torch.tensor([[1.0, dt],
                  [0.0, 1.0]], dtype=torch.float32)
B = torch.tensor([[0.5*dt**2],
                  [dt]], dtype=torch.float32)
C = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
# Q = (std_dyn**2) * torch.tensor([[dt**4/4, dt**3/2],
#                                 [dt**3/2, dt**2]], dtype=torch.float32)
# R = torch.tensor([[std_obs**2]], dtype=torch.float32)

# Assume K=1
K = 1
A_in = A.repeat(K, 1, 1) 
B_in = B.repeat(K, 1, 1) 
C_in = C.repeat(K, 1, 1) 

mu0    = torch.zeros(2, dtype=torch.float32)                     
Sigma0 = torch.diag(torch.tensor([1**2, 1**2], dtype=torch.float32))

dyn_param = DynamicsParameter(A_in, B_in, C_in)
kf = KalmanFilter(std_dyn, std_obs, mu0, Sigma0, dyn_param)

model_dtype = A.dtype
model_device = A.device
Y = torch.from_numpy(z_meas).to(dtype=model_dtype, device=model_device).unsqueeze(0).unsqueeze(-1)  # [1,N,1]
U = torch.from_numpy(u_meas).to(dtype=model_dtype, device=model_device).unsqueeze(0).unsqueeze(-1)  # [1,N,1]

# Compute filter
mus_filt, Sigmas_filt, mus_pred, Sigmas_pred, A_list, B_list, C_list = kf.filter(Y, U)
# # Compute smoother
mus_smooth, Sigmas_smooth, _, _, _ = kf.smooth(Y, U)
# # Compute ELBO
elbo = kf.elbo(mus_smooth, Sigmas_smooth, Y, U, A_list, B_list, C_list)
print(f"ELBO: {elbo.item():.2f}")

# Convert tensors to numpy
mu_np = mus_filt[0].squeeze(-1).detach().cpu().numpy()
P_np  = Sigmas_filt[0].detach().cpu().numpy()
pos_mean = mu_np[:, 0]
vel_mean = mu_np[:, 1]
pos_std  = np.sqrt(P_np[:, 0, 0])
vel_std  = np.sqrt(P_np[:, 1, 1])
mu_f_s  = mus_smooth[0].squeeze(-1).detach().cpu().numpy()       
P_f_s   = Sigmas_smooth[0].detach().cpu().numpy()    
pos_mean_s = mu_f_s[:, 0]
vel_mean_s = mu_f_s[:, 1]
pos_std_s  = np.sqrt(P_f_s[:, 0, 0])
vel_std_s  = np.sqrt(P_f_s[:, 1, 1])

# Altitude plot 
plt.figure()
plt.plot(t, x[:,0], label="Reference", color="black", linestyle="--")
plt.scatter(t, z_meas, label="Observations", marker="x", color="blue", alpha=0.5)
plt.plot(t, pos_mean, label="KF mean", color="orange", linewidth=2)
plt.fill_between(t, pos_mean - pos_std, pos_mean + pos_std, color="orange", alpha=0.2, label="±1std")
plt.plot(t, pos_mean_s, label="RTS smoothed mean", color="red", linewidth=2)
plt.fill_between(t, pos_mean_s - pos_std_s, pos_mean_s + pos_std_s, color="red", alpha=0.2, label="smooth ±1std")
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Altitude: reference vs KF mean ±1std")
plt.legend()
plt.grid(True)
plt.show()

# Velocity plot
plt.figure()
plt.plot(t, x[:,1], label="reference (velocity)", color="black", linestyle="--")
plt.plot(t, vel_mean, label="KF mean (velocity)", color="orange", linewidth=2)
plt.plot(t, vel_mean_s, label="RTS smoothed mean", color="red", linewidth=2)
plt.fill_between(t, vel_mean - vel_std, vel_mean + vel_std, color="orange", alpha=0.2, label="±1std")
plt.fill_between(t, vel_mean_s - vel_std_s, vel_mean_s + vel_std_s, color="red", alpha=0.2, label="smooth ±1std")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity: reference vs KF mean ±1std")
plt.legend()
plt.grid(True)
plt.show()