
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvae.model.config import KVAEConfig

class DynamicsParameter(nn.Module):
    def __init__(self, A, B, C, hidden_lstm=32):
        """
        A0: [K, n, n] initial state transition matrix
        B0: [K, n, m] initial control input matrix
        C0: [K, p, n] initial observation matrix
        """
        super().__init__()
        self.K = A.size(0)  
        n = A.size(1)
        m = B.size(2)
        p = C.size(1)
        self.n, self.m, self.p = n, m, p

        self.A = nn.Parameter(A.clone()) 
        self.B = nn.Parameter(B.clone())
        self.C = nn.Parameter(C.clone())

        # No persistent LSTM state kept on the module; callers should
        # manage any state if needed. This avoids storing tensors with
        # autograd history on module attributes.

        if self.K > 1:
            self.lstm = nn.LSTM(
                input_size=self.p, 
                hidden_size=hidden_lstm, 
                num_layers=1, 
                batch_first=True)

            self.mlp = nn.Sequential(
                nn.Linear(hidden_lstm, hidden_lstm), 
                nn.Tanh(),
                )
            
            self.head_w = nn.Linear(hidden_lstm, self.K)

    def reset_state(self):
        # kept for backward compatibility; no-op in stateless implementation
        return

    # Stateless compute_step: accepts an optional `state` and returns
    # (A, B, C, new_state). Callers should pass the previous LSTM state
    # (or `None`) and manage it across time steps.
    def compute_step(self, a_tprev, state=None):
        batch = a_tprev.size(0)

        if self.K == 1:
            A = self.A[0].expand(batch, -1, -1)
            B = self.B[0].expand(batch, -1, -1)
            C = self.C[0].expand(batch, -1, -1)

            return A, B, C, None

        # K > 1: run LSTM for this step using provided state
        a_tprev = a_tprev.unsqueeze(1)
        # `state` should be a tuple (h, c) or None
        h, new_state = self.lstm(a_tprev, state)
        f = self.mlp(h)
        w = torch.softmax(self.head_w(f), dim=-1).squeeze(1)

        A = torch.einsum('bk,kij->bij', w, self.A)
        B = torch.einsum('bk,knm->bnm', w, self.B)
        C = torch.einsum('bk,kpn->bpn', w, self.C)

        return A, B, C, new_state