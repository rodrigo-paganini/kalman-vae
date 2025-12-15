
import torch
import torch.nn as nn

class DynamicsParameter(nn.Module):
    def __init__(self, A, B, C, hidden_lstm=50):
        super().__init__()
        self.use_switching_dynamics = False
        self.K = A.size(0)
        n = A.size(1)
        m = B.size(2)
        p = C.size(1)
        self.n, self.m, self.p = n, m, p

        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.C = nn.Parameter(C.clone())

        self.lstm_state = None
        self.state_seq = None

        if self.K > 1:
            self.lstm = nn.LSTM(
                input_size=self.p,
                hidden_size=hidden_lstm,
                num_layers=1,
                batch_first=True,
            )
            self.head_w = nn.Linear(hidden_lstm, self.K)
            # Bias alpha to mode 0 at initialization
            with torch.no_grad():
                self.head_w.bias.fill_(-10.0)
                self.head_w.bias[0] = 0.0

    def reset_state(self):
        self.lstm_state = None
        self.state_seq = []

    def compute_step(self, a_tprev):
        batch = a_tprev.size(0)

        if self.K == 1:
            A = self.A[0].expand(batch, -1, -1)
            B = self.B[0].expand(batch, -1, -1)
            C = self.C[0].expand(batch, -1, -1)
            w = torch.ones(batch, 1, device=a_tprev.device, dtype=a_tprev.dtype)
            self.state_seq.append(w)
            return A, B, C

        a_tprev = a_tprev.unsqueeze(1)

        h, new_state = self.lstm(a_tprev, self.lstm_state)  # h: [B,1,hidden_lstm]
        self.lstm_state = new_state

        alpha_logits = self.head_w(h.squeeze(1))                  
        w = torch.softmax(alpha_logits, dim=-1)             

        A = torch.einsum('bk,kij->bij', w, self.A)
        B = torch.einsum('bk,knm->bnm', w, self.B)
        C = torch.einsum('bk,kpn->bpn', w, self.C)

        self.state_seq.append(w)
        return A, B, C
