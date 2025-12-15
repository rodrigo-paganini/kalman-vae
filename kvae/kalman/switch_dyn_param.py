
import torch
import torch.nn as nn
from torch.distributions import Multinomial
from torch.nn.functional import gumbel_softmax

class SwitchingDynamicsParameter(nn.Module):
    def __init__(self, A, B, C, Q=None, prior=None, hidden_lstm=32, markov_regime_posterior=None):
        super().__init__()
        self.is_switching_dynamics = True
        self.K = A.size(0)
        n = A.size(1)
        m = B.size(2)
        p = C.size(1)
        self.n, self.m, self.p = n, m, p
        self.tau = 0.5  # Gumbel-softmax temperature

        if Q is None:
            Q = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(self.K, 1, 1)

        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.C = nn.Parameter(C.clone())
        self.Q = nn.Parameter(Q.clone())

        self.s_tprev = None  # To be set externally if needed
        self.prior = prior if prior is not None else StickyRegimePrior(self.K)
        self.markov_regime_posterior = markov_regime_posterior or MarkovVariationalRegimePosterior(
            self.K, input_dim=p, hidden_size=hidden_lstm
        )
        self.hidden_size = hidden_lstm
        self.state_seq = None

    def reset_state(self):
        self.state_seq = None

    def compute_batch(self, a_seq, is_training=True):
        batch, T, _ = a_seq.size()

        if self.K == 1:
            A = self.A[0].expand(batch, T, -1, -1)
            B = self.B[0].expand(batch, T, -1, -1)
            C = self.C[0].expand(batch, T, -1, -1)
            Q = self.Q[0].expand(batch, T, -1, -1)
            self.log_qseq = torch.zeros(batch, T, device=a_seq.device, dtype=a_seq.dtype)
            self.log_pseq = torch.zeros(batch, T, device=a_seq.device, dtype=a_seq.dtype)
            self.Q_seq = Q
            self.state_seq = torch.ones(batch, T, self.K, device=a_seq.device, dtype=a_seq.dtype)
            return A, B, C, Q

        logits, init_logits = self.markov_regime_posterior(a_seq)  # logits: [B, T, K, K], init_logits: [B,K]
        y0 = gumbel_softmax(init_logits, tau=self.tau, hard=not is_training, dim=-1)  # [B,K]
        log_q0 = torch.log_softmax(init_logits, dim=-1)
        log_p0 = torch.full_like(log_q0, 1.0 / self.K).log()

        y_seq = torch.zeros((batch, T, self.K), device=a_seq.device, dtype=a_seq.dtype)
        log_qseq = torch.zeros((batch, T), device=a_seq.device, dtype=a_seq.dtype)
        log_pseq = torch.zeros((batch, T), device=a_seq.device, dtype=a_seq.dtype)

        y_seq[:, 0] = y0
        log_qseq[:, 0] = (y0 * log_q0).sum(dim=-1)
        log_pseq[:, 0] = (y0 * log_p0).sum(dim=-1)

        y_tprev = y0
        trans_matrix = self.prior.transition_matrix.to(device=a_seq.device, dtype=a_seq.dtype)
        
        for t in range(1, T):
            l_t = torch.matmul(y_tprev.unsqueeze(1), logits[:, t, ...]).squeeze(1)  # [B,K]
            y_t = gumbel_softmax(l_t, tau=self.tau, hard=not is_training, dim=-1)  # [B,K]
            y_seq[:, t] = y_t

            # ELBO terms
            log_q = torch.log_softmax(l_t, dim=-1)
            log_qseq[:, t] = (y_t * log_q).sum(dim=-1)

            trans_probs = torch.matmul(y_tprev.unsqueeze(1), trans_matrix).squeeze(1)
            log_pseq[:, t] = (y_t * torch.log(trans_probs.clamp_min(1e-8))).sum(dim=-1)
            
            y_tprev = y_t

        # Compute weighted sums: A_t = sum_k y_{t,k} * A_k
        A_seq = torch.einsum('btk,kij->btij', y_seq, self.A)  # [B,T,n,n]
        B_seq = torch.einsum('btk,knm->btnm', y_seq, self.B)  # [B,T,n,m]
        Q_seq = torch.einsum('btk,kij->btij', y_seq, self.Q)  # [B,T,n,n]
        C_shared = self.C[0]  # [p,n], emission assumed constant across regimes
        C_seq = C_shared.expand(batch, T, -1, -1)  # [B,T,p,n]
        self.log_qseq = log_qseq
        self.log_pseq = log_pseq
        self.Q_seq = Q_seq
        self.state_seq = y_seq

        return A_seq, B_seq, C_seq, Q_seq

    def elbo_terms(self):
        return self.log_qseq, self.log_pseq


class StickyRegimePrior:
    def __init__(self, K, p_stay=0.9):
        self.K = K
        self.p_stay = p_stay
        self.transition_matrix = torch.ones((K, K)) * ((1 - p_stay) / (K - 1))
        self.transition_matrix.fill_diagonal_(p_stay)

    def reset_state(self):
        self.state_probabilities = Multinomial(probs=torch.ones(self.K) / self.K)

    def compute_step(self, prev_probs):
        new_probs = torch.matmul(prev_probs, self.transition_matrix)
        return new_probs


class MarkovVariationalRegimePosterior(nn.Module):
    def __init__(self, K, input_dim, hidden_size=32):
        super().__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.bigru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_head = nn.Linear(2 * hidden_size, K * K)
        self.init_head = nn.Linear(2 * hidden_size, K)

    def forward(self, a_seq):
        h_seq, _ = self.bigru(a_seq)
        logits = self.linear_head(h_seq)  # [B,T,K*K]
        B, T, _ = logits.shape
        logits = logits.view(B, T, self.K, self.K)
        # Change wrt what we did last night, dedicated head for initial logits
        init_logits = self.init_head(h_seq[:, 0]) 
        return logits, init_logits
