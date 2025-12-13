
import torch
import torch.nn as nn
from torch.distributions import Multinomial
from torch.nn.functional import gumbel_softmax

class DynamicsParameter(nn.Module):
    def __init__(self, A, B, C, Q, prior, hidden_size, ):
        super().__init__()
        self.K = A.size(0)
        n = A.size(1)
        m = B.size(2)
        p = C.size(1)
        self.n, self.m, self.p = n, m, p
        self.tau = 0.5  # Gumbel-softmax temperature

        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.C = nn.Parameter(C.clone())
        self.Q = nn.Parameter(Q.clone())

        self.s_tprev = None  # To be set externally if needed
        self.prior = prior
        self.markov_regime_posterior = None  # To be set externally if needed
        self.hidden_size = hidden_size

    def reset_state(self):
        self.switch_state_probabilities = Multinomial(probs=torch.ones(self.K) / self.K)

    def compute_batch(self, a_seq, is_training=True):
        batch = a_seq.size(0)
        T = a_seq.size(1)

        if self.K == 1:
            A = self.A[0].expand(batch, -1, -1)
            B = self.B[0].expand(batch, -1, -1)
            C = self.C[0].expand(batch, -1, -1)
            Q = self.Q[0].expand(batch, 1, -1, -1)
            return A, B, C, Q

        logits = self.markov_regime_posterior(a_seq)  # logits: [B, T, K, K]
        y0 = gumbel_softmax(logits[:, 0, :], tau=self.tau, hard=not is_training, dim=-1)  # [B,K]
        log_q0 = torch.log_softmax(logits[:, 0, :], dim=-1)
        # self.switch_state_probabilities = torch.softmax(logits, dim=-1)  # TODO apply softmax row-wise, review shapes
        # PI_t: [B, T, K, K]

        # prev_probs = self.switch_state_probabilities.probs.unsqueeze(0).expand(batch, -1)  # [B,K]
        y_seq = torch.zeros((batch, self.K), device=a_seq.device)
        y_tprev = y0
        y_seq[:, 0] = y0
        log_qseq = torch.zeros((batch, self.K), device=a_seq.device)
        log_qseq[:, 0] = torch.matmul(y_tprev.unsqueeze(1), log_q0).squeeze(1)
        log_pseq = torch.zeros((batch, self.K), device=a_seq.device)
        log_p0 = torch.log(torch.ones((batch, self.K), device=a_seq.device)/self.K, dim=-1)
        log_pseq[:, 0] = torch.matmul(y_tprev.unsqueeze(1), log_p0).squeeze(1)
        
        for t in range(1, T):
            l_t = torch.matmul(y_tprev.unsqueeze(1), logits[:, t, ...])
            y_t = gumbel_softmax(l_t, tau=self.tau, hard=not is_training, dim=-1)  # [B,1,K] TODO check dimensions
            y_seq[:, t] = y_t.squeeze(1)  # [B,K]
            # elbo terms
            log_q = torch.log_softmax(l_t, dim=-1) # revisar dim
            log_qseq[:, t] = torch.matmul(y_t.unsqueeze(1), torch.matmul(y_tprev.unsqueeze(1), log_q)).squeeze(1)
            log_p = torch.log(self.prior.transition_matrix, dim=-1)
            log_pseq = torch.matmul(y_t.unsqueeze(1), torch.matmul(y_tprev.unsqueeze(1), log_p)).squeeze(1)
            
            y_tprev = y_t.squeeze(1)

        # Compute weighted sums: A_t = sum_k y_{t,k} * A_k
        A_seq = torch.einsum('btk,kij->btij', y_seq, self.A)  # [B,T,n,n]
        B_seq = torch.einsum('btk,knm->btnm', y_seq, self.B)  # [B,T,n,m]
        C_seq =  self.C.expand(batch, 1, -1, -1) # [B,1,p,n]
        self.log_qseq = log_qseq
        self.log_pseq = log_pseq

        return A_seq, B_seq, C_seq

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


class MarkovVariationalRegimePosterior:
    def __init__(self, K, hidden_size=32):
        self.K = K
        self.hidden_size = hidden_size
        self.bigru = nn.GRU(input_size=K, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_head = nn.Linear(hidden_size, K*K)

    def forward(self, a_seq):
        h_seq, _ = self.bigru(a_seq)
        logits = self.linear_head(h_seq)  # [B,T,K,K]
        B, T, _ = logits.shape
        logits = logits.view(B, T, self.K, self.K)
        return logits