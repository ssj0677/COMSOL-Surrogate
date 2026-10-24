# -*- coding: utf-8 -*-
"""Autoregressive TFT — 구조는 Direct TFT 와 동일, 예측 방식만 자기회귀.

왜 만들었나
    기존 자기회귀 비교군은 LSTMCell 기반(model_ar.py)이었다. 그것을 Direct TFT
    와 비교하면 **방식(되먹임 vs 직접)과 구조(LSTM vs TFT)가 동시에 다르다.**
    대리모델로 TFT 를 채택했으므로 '왜 자기회귀를 안 썼는가' 는 같은 TFT 로
    답해야 한다.

무엇이 같고 무엇이 다른가
    같음 : StaticCovariateEncoder, VSN, LSTM, GateAddNorm, causal attention, GRN,
           그리고 하이퍼파라미터(S.TUNED['TFT'])
    다름 : 입력 변수가 하나 늘어난다 — '직전 lag 스텝의 출력' 블록
           평가 때 그 자리에 자기 예측을 넣는다

★ 되먹임을 '변수 하나' 로 넣는 이유
    lag=8 이면 되먹임 값이 16개다. 이것을 VSN 의 변수 16개로 넣으면 변수 수가
    3 -> 19 로 늘어 VSN 의 weight_grn 이 (19*64)^2 로 커지고, 파라미터가
    392k -> 6.5M 으로 16배가 된다. 그러면 '같은 구조' 비교가 깨진다.
    TFT 의 VSN 은 '변수' 단위로 고르는 장치이므로, 되먹인 상태 벡터는 하나의
    변수로 보는 것이 구조상으로도 맞다. 이렇게 하면 파라미터가 Direct TFT 와
    거의 같아진다(추가분은 Linear(N_OUT*lag, hidden) 하나뿐).

★ 롤아웃이 정확한 이유 — TFT 의 attention 이 이미 causal 이다
    triu 마스크를 쓰므로 출력 t 는 입력 t 이하만 본다. 따라서 길이 t+1 접두부로
    돌려 마지막 자리를 읽는 것이 전 구간 결과의 t 자리와 같다(수치오차 4e-7 확인).
    비용은 O(T^2) 이지만 평가에서만 든다. 학습은 teacher forcing 이라 Direct 와
    같은 비용이다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn

from model_tft_re import TFT, VariableSelectionNetwork
from surrogate import N_IN, N_OUT


def _lagged(y, lag):
    """y (B,T,N_OUT) -> (B,T,N_OUT*lag). 위치 t 에 [y(t-1),...,y(t-lag)].

    앞쪽이 모자라면 y(0) 으로 채운다. 0 으로 채우면 표준화 공간에서 '평균'이
    아니라 임의의 값이 되어 초반 스텝이 흔들린다.
    """
    B, T, _ = y.shape
    out = []
    for k in range(1, lag + 1):
        pad = y[:, :1].expand(B, min(k, T), N_OUT)
        out.append(torch.cat([pad, y[:, :max(T - k, 0)]], dim=1)[:, :T])
    return torch.cat(out, dim=-1)


class TFTAR(TFT):
    """TFT 와 같은 블록. 입력 변수만 하나(되먹임 블록) 늘렸다."""

    def __init__(self, hidden=64, heads=4, dropout=0.1, lag=8):
        super().__init__(hidden=hidden, heads=heads, dropout=dropout)
        self.lag = lag
        # 되먹임 블록 전용 임베딩. var_embed 는 TFT 것을 그대로 쓰고,
        # VSN 만 변수 하나가 늘어난 크기로 다시 만든다.
        self.fb_embed = nn.Linear(N_OUT * lag, hidden)
        self.vsn = VariableSelectionNetwork(N_IN + 1, hidden, ctx=hidden,
                                            dropout=dropout)

    # TFT.forward 를 그대로 못 쓰는 이유: 임베딩 단계에서 되먹임 블록을
    # 변수 하나로 끼워 넣어야 한다. 그 한 줄만 다르고 나머지는 동일하다.
    def _core(self, init, exo, fb):
        c_s, c_e, c_h, c_c = self.static_enc(init)
        v = [emb(exo[..., i:i + 1]) for i, emb in enumerate(self.var_embed)]
        v.append(self.fb_embed(fb))                 # <- 되먹임 블록 = 변수 하나
        vars_embedded = torch.stack(v, dim=-2)      # [B,T,N_IN+1,hidden]
        selected, _ = self.vsn(vars_embedded, c_s[:, None])
        lstm_out, _ = self.lstm(selected, (c_h[None], c_c[None]))
        x = self.post_lstm_gate(lstm_out, selected)
        e = self.enrich(x, c_e[:, None])
        T = e.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=e.device), 1)
        a, _ = self.attn(e, e, e, attn_mask=mask)
        x = self.post_attn_gate(a, e)
        f = self.ffn(x)
        x = self.post_ffn_gate(f, x)
        return self.head(x)

    def forward_tf(self, y_true, exo):
        """teacher forcing — 학습용. 전 구간 한 번에."""
        return self._core(y_true[:, 0], exo, _lagged(y_true, self.lag))

    def forward(self, init, exo):
        """closed-loop — 평가용. 자기 예측을 되먹이며 한 스텝씩."""
        B, T, _ = exo.shape
        y = init[:, None].clone()                   # t=0 은 참값
        for t in range(1, T):
            pad = y[:, -1:].expand(B, 1, N_OUT)     # 아직 없는 t 자리를 임시로
            yy = torch.cat([y, pad], dim=1)
            nxt = self._core(init, exo[:, :t + 1],
                             _lagged(yy, self.lag))[:, -1:]
            y = torch.cat([y, nxt], dim=1)
        return y
