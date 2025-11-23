
import torch
import torch.nn as nn

class DynamicsParameter(nn.Module):
    def __init__(self, A, B, C, hidden_lstm=32):
        """
        A0: [K, n,n] initial state transition matrix
        B0: [K, n,m] initial control input matrix
        C0: [K, p,n] initial observation matrix
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

        self.lstm_state = None 

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
        self.lstm_state = None

    #TODO: why only a and not z?
    def compute_step(self, a_tprev):
        batch = a_tprev.size(0)

        if self.K == 1:
            A = self.A[0].expand(batch, -1, -1)
            B = self.B[0].expand(batch, -1, -1)
            C = self.C[0].expand(batch, -1, -1)

            return A, B, C

        else:
            a_tprev = a_tprev.unsqueeze(1)  

            h, new_state = self.lstm(a_tprev, self.lstm_state)
            f = self.mlp(h)                 
            self.w = torch.softmax(self.head_w(f), dim=-1).squeeze(1)  
            
            self.lstm_state = new_state

            A = torch.einsum('bk,kij->bij', self.w, self.A)
            B = torch.einsum('bk,knm->bnm', self.w, self.B)
            C = torch.einsum('bk,kpn->bpn', self.w, self.C)

            return A, B, C