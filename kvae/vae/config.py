from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class KVAEConfig:
    # Data
    img_channels: int = 1
    img_size: int = 32          # box is 32x32

    # Latent dims
    a_dim: int = 2              # dim_a
    z_dim: int = 4              # dim_z
    u_dim: Optional[int] = None # dim_u (controls)

    # LGSSM / mixture
    num_modes: int = 3              # K
    sticky_p_stay: float = 0.9      # self-transition prob for sticky prior
    tau_init: float = 0.5           # initial Gumbel-Softmax temperature
    tau_decay_rate: float = 0.99    # multiplicative decay for tau
    tau_decay_steps: int = 1        # apply tau decay every N epochs
    use_switching_dynamics: bool = True  # toggle switching LDS
    noise_emission: float = 0.03    # measurement noise (on a)
    noise_transition: float = 0.02  # process noise (on z)  
    init_cov: float = 20.0          # initial state variance 
    init_kf_matrices: float = 0.05  # std for B,C init      

    # VAE arch (conv=True, num_filters="32,32,32", filter_size=3)
    out_distr: str = "bernoulli"  # "bernoulli" or "gaussian"
    encoder_channels: Optional[List[int]] = None
    encoder_kernel_size: int = 3
    encoder_stride: int = 2
    encoder_padding: int = 1

    decoder_channels: Optional[List[int]] = None
    decoder_kernel_size: int = 3
    decoder_stride: int = 2
    decoder_padding: int = 1

    noise_pixel_var: float = 0.1
    scale_reconstruction: float = 0.3

    # Alpha / dynamics network (alpha_rnn=True, alpha_units=50)
    dynamics_hidden_dim: int = 50      # LSTM hidden size ~ alpha_units

    # Training / optimization
    grad_clip_norm: float = 150.0      # max_grad_norm in TF
    recon_weight: float = 0.3          # if you use it anywhere

    init_lr: float = 0.001      # init_lr
    decay_rate: float = 0.85    # decay_rate
    decay_steps: int = 20       # decay_steps

    # Inputation
    generate_step: int = 5          # Number of steps to generate during imputation
    t_init_mask: int = 4            # Initial time step to start masking
    t_steps_mask: int = 12          # Number of steps to mask during imputation

    def __post_init__(self):
        if self.u_dim is None:
            self.u_dim = self.z_dim
        if self.encoder_channels is None:
            self.encoder_channels = [32, 32, 32]
        if self.decoder_channels is None:
            self.decoder_channels = [32, 32, 32]
