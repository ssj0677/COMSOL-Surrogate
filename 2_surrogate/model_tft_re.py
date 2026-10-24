
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.nn import functional as F

from surrogate import N_IN, N_OUT


class GRN(nn.Module):
    """Gated Residual Network — 입력을 얼마나 반영할지 sigmoid 게이트로 결정 """

    def __init__(self, dim, ctx=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ctx = nn.Linear(ctx, dim, bias=False) if ctx else None
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.fc2(F.elu(h))
        a, b = self.drop(self.gate(h)).chunk(2, dim=-1)
        return self.norm(x + a * torch.sigmoid(b))


class GateAddNorm(nn.Module):
    """블록(LSTM/어텐션/FFN) 출력을 GLU로 게이팅한 뒤 잔차+LN """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.glu = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, residual):
        a, b = self.drop(self.glu(x)).chunk(2, dim=-1)
        return self.norm(residual + a * torch.sigmoid(b))


class StaticCovariateEncoder(nn.Module):
    """정적 변수(여기서는 초기 Vcell/c_mix)로부터 용도가 다른 4개 컨텍스트를
    만든다 : static covariate encoder."""

    def __init__(self, static_dim, hidden, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(static_dim, hidden)
        self.grn_vsn = GRN(hidden, dropout=dropout)      # c_s: 변수 선택용
        self.grn_enrich = GRN(hidden, dropout=dropout)   # c_e: static enrichment용
        self.grn_h = GRN(hidden, dropout=dropout)        # c_h: LSTM 초기 hidden
        self.grn_c = GRN(hidden, dropout=dropout)        # c_c: LSTM 초기 cell

    def forward(self, static):
        s = self.embed(static)
        return self.grn_vsn(s), self.grn_enrich(s), self.grn_h(s), self.grn_c(s)


class VariableSelectionNetwork(nn.Module):
    """시점마다 입력 변수(I, Q_cum, c_in)별 중요도를 학습해 가중합 """

    def __init__(self, n_vars, dim, ctx=None, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList(
            [GRN(dim, dropout=dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(dim * n_vars, ctx=ctx, dropout=dropout)
        self.weight_fc = nn.Linear(dim * n_vars, n_vars)

    def forward(self, vars_embedded, c=None):
        # vars_embedded: [B, T, n_vars, dim]
        flat = vars_embedded.flatten(-2)                      # [B,T,n_vars*dim]
        w = F.softmax(self.weight_fc(self.weight_grn(flat, c)), dim=-1)
        processed = torch.stack(
            [grn(vars_embedded[..., i, :]) for i, grn in enumerate(self.var_grns)],
            dim=-2)                                            # [B,T,n_vars,dim]
        selected = (processed * w.unsqueeze(-1)).sum(-2)        # [B,T,dim]
        return selected, w


class TFT(nn.Module):

    def __init__(self, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.static_enc = StaticCovariateEncoder(N_OUT, hidden, dropout)
        self.var_embed = nn.ModuleList(
            [nn.Linear(1, hidden) for _ in range(N_IN)])
        self.vsn = VariableSelectionNetwork(N_IN, hidden, ctx=hidden, dropout=dropout)

        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm_gate = GateAddNorm(hidden, dropout)

        self.enrich = GRN(hidden, ctx=hidden, dropout=dropout)

        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.post_attn_gate = GateAddNorm(hidden, dropout)

        self.ffn = GRN(hidden, dropout=dropout)
        self.post_ffn_gate = GateAddNorm(hidden, dropout)

        self.head = nn.Linear(hidden, N_OUT)

    def forward(self, init, exo):
        # 1) 정적 공변량 인코더: 초기조건 → 용도별 컨텍스트 4개
        c_s, c_e, c_h, c_c = self.static_enc(init)              # 각 [B, hidden]

        # 2) 변수별 임베딩 → VSN으로 시점별 변수 선택
        vars_embedded = torch.stack(
            [emb(exo[..., i:i + 1]) for i, emb in enumerate(self.var_embed)],
            dim=-2)                                              # [B,T,N_IN,hidden]
        selected, _ = self.vsn(vars_embedded, c_s[:, None])       # [B,T,hidden]

        # 3) LSTM — 초기 hidden/cell을 정적 컨텍스트로 초기화 (0 초기화 대신)
        lstm_out, _ = self.lstm(selected, (c_h[None], c_c[None]))
        x = self.post_lstm_gate(lstm_out, selected)                # 게이트+잔차

        # 4) static enrichment — 컨텍스트를 다시 한 번 주입
        e = self.enrich(x, c_e[:, None])

        # 5) causal self-attention (장거리 의존성)
        T = e.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=e.device), 1)
        a, _ = self.attn(e, e, e, attn_mask=mask)
        x = self.post_attn_gate(a, e)                               # 게이트+잔차

        # 6) position-wise feed-forward (GRN) + 최종 게이트
        f = self.ffn(x)
        x = self.post_ffn_gate(f, x)

        return self.head(x)


if __name__ == '__main__':
    import surrogate as S
    S.run_single('TFT')
