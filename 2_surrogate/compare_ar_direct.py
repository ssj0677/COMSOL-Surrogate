# -*- coding: utf-8 -*-
"""자기회귀 TFT vs 직접예측 TFT — 표(CSV)를 만든다.  Fig. 7 과 같은 값.

왜 이 비교인가
    "왜 자기회귀를 안 썼는가" 는 같은 구조로 답해야 한다. 예전 비교군은
    LSTMCell 기반이라 **방식(되먹임 vs 직접)과 구조(LSTM vs TFT)가 동시에**
    달랐다. model_tft_ar.TFTAR 은 Direct TFT 와 같은 블록·같은 하이퍼파라미터
    (S.TUNED['TFT'])를 쓰고 입력 변수만 하나(되먹임 블록) 늘렸다.

세 행의 뜻
    Direct TFT                 전체 궤적을 한 번에 예측 — 채택한 방식
    Autoregressive (closed-loop)  자기 예측을 되먹임 — 실제로 쓸 수 있는
                               유일한 자기회귀 모드
    Autoregressive (one-step)  참값을 되먹임(teacher forcing). 실제로는 못 쓴다.
                               closed-loop 과 **같은 가중치**이므로
                               '학습이 부족한가 되먹임이 문제인가' 를 가른다.
                               t=0 은 뺀다 — 학습 손실이 t=0 을 빼므로 그 자리는
                               학습된 적이 없다(오차 ~183 mV).

지표는 model_table_3seed.csv 와 같게 (RMSE / MAE / MAPE, 전압 mV·농도 mM).
n=3 이라 표준편차와 함께 **최소~최대**도 남긴다 — 자기회귀는 seed 편차가
평균보다 커서(sd 17.7 > 평균 15.2) 범위 쪽이 읽기 쉽다.

가중치
    Direct  figures_TFT_s{seed}.pt   (train_seq_seeds.py, 현재 TUNED)
    AR      tftar_lag8_s{seed}.pt    (run_tft_ar.py, 같은 TUNED + lag 8)

    py 2_surrogate/compare_ar_direct.py
"""
import os
import sys
import warnings

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '2_surrogate'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

import surrogate as S
from model_tft_ar import TFTAR

OUT_DIR = os.path.join(S.BASE, '_S_result')
CKPT = os.path.join(OUT_DIR, 'ar_models')
SEEDS = [0, 1, 2]
EPOCHS = 150
L_DIR = 'Direct TFT'
L_AR = 'Autoregressive TFT (closed-loop)'
L_1S = 'Autoregressive TFT (one-step)'
ORDER = [L_DIR, L_AR, L_1S]


def load_direct(seed):
    m = S.build('TFT', epochs=EPOCHS, seed=seed)
    m.net = S.TORCH_MODELS['TFT'](**m.kwargs).to(S.DEVICE)
    m.net.load_state_dict(torch.load(
        os.path.join(CKPT, f'figures_TFT_s{seed}.pt'), map_location=S.DEVICE))
    m.net.eval()
    return m


def load_ar(seed):
    kw = {k: v for k, v in S.TUNED['TFT'].items() if k != 'lr'}
    lag = S.TUNED['AR']['lag']
    net = TFTAR(lag=lag, **kw).to(S.DEVICE)
    net.load_state_dict(torch.load(
        os.path.join(CKPT, f'tftar_lag{lag}_s{seed}.pt'), map_location=S.DEVICE))
    net.eval()
    return net


def stats(E, O, EC, OC):
    e, o = np.concatenate(E), np.concatenate(O)
    ec, oc = np.concatenate(EC), np.concatenate(OC)
    return dict(V_RMSE_mV=np.sqrt((e ** 2).mean()) * 1000,
                V_MAE_mV=np.abs(e).mean() * 1000,
                V_MAPE_pct=np.mean(np.abs(e / o)) * 100,
                c_RMSE_mM=np.sqrt((ec ** 2).mean()),
                c_MAE_mM=np.abs(ec).mean(),
                c_MAPE_pct=np.mean(np.abs(ec / oc)) * 100)


def main():
    pool = S.load(list(S.USE))
    tr, va, te = S.split(pool)
    sc = S.Scaler().fit(tr)
    rows = []
    for seed in SEEDS:
        m = load_direct(seed)
        E, O, EC, OC = [], [], [], []
        for r in te.values():
            p = S.to_c_mix(sc.back(m.predict(sc.forward(r))), r)
            E.append(p[:, 0] - r['Vcell'].to_numpy(float))
            O.append(r['Vcell'].to_numpy(float))
            EC.append(p[:, 1] - r['c_mix'].to_numpy(float))
            OC.append(r['c_mix'].to_numpy(float))
        d = stats(E, O, EC, OC)
        d.update(model=L_DIR, seed=seed, amp_V=np.nan, amp_c=np.nan,
                 params_k=sum(q.numel() for q in m.net.parameters()) / 1e3)
        rows.append(d)

        net = load_ar(seed)
        E, O, EC, OC = [], [], [], []
        E1, O1, EC1, OC1 = [], [], [], []
        with torch.no_grad():
            for r in te.values():
                t = S._tensor(sc.forward(r))
                y, exo = t[None, :, :S.N_OUT], t[None, :, S.N_OUT:]
                o, oc = r['Vcell'].to_numpy(float), r['c_mix'].to_numpy(float)
                p = S.to_c_mix(sc.back(net(y[:, 0], exo)[0].cpu().numpy()), r)
                ptf = S.to_c_mix(
                    sc.back(net.forward_tf(y, exo)[0].cpu().numpy()), r)
                E.append(p[:, 0] - o)
                O.append(o)
                EC.append(p[:, 1] - oc)
                OC.append(oc)
                E1.append(ptf[1:, 0] - o[1:])
                O1.append(o[1:])
                EC1.append(ptf[1:, 1] - oc[1:])
                OC1.append(oc[1:])
        npar = sum(q.numel() for q in net.parameters()) / 1e3
        d, d1 = stats(E, O, EC, OC), stats(E1, O1, EC1, OC1)
        d.update(model=L_AR, seed=seed, params_k=npar,
                 amp_V=d['V_RMSE_mV'] / d1['V_RMSE_mV'],
                 amp_c=d['c_RMSE_mM'] / d1['c_RMSE_mM'])
        d1.update(model=L_1S, seed=seed, params_k=npar,
                  amp_V=np.nan, amp_c=np.nan)
        rows += [d, d1]
        print(f'  seed {seed}:  Direct {rows[-3]["V_RMSE_mV"]:6.2f}   '
              f'AR closed {d["V_RMSE_mV"]:7.2f}   one-step '
              f'{d1["V_RMSE_mV"]:5.2f} mV   증폭 {d["amp_V"]:5.1f}배',
              flush=True)

    raw = pd.DataFrame(rows)
    cols = ['V_RMSE_mV', 'V_MAE_mV', 'V_MAPE_pct',
            'c_RMSE_mM', 'c_MAE_mM', 'c_MAPE_pct', 'amp_V', 'amp_c']
    g = raw.groupby('model')[cols]
    mu, sd = g.mean().loc[ORDER], g.std().loc[ORDER]
    lo, hi = g.min().loc[ORDER], g.max().loc[ORDER]
    par = raw.groupby('model').params_k.first().loc[ORDER]

    print(f'\n테스트 {len(te)}조건, seed {SEEDS}   평균 [최소-최대]\n')
    hdr = ['V RMSE (mV)', 'V MAE (mV)', 'V MAPE (%)',
           'c RMSE (mM)', 'c MAE (mM)', 'c MAPE (%)']
    print(f'{"":34}' + ''.join(f'{h:>23}' for h in hdr) + f'{"params(k)":>11}')
    for n in ORDER:
        line = ''.join(
            f'{mu.loc[n, c]:.2f} [{lo.loc[n, c]:.2f}-{hi.loc[n, c]:.2f}]'
            .rjust(23) for c in cols[:6])
        print(f'{n:34}{line}{par[n]:>11.0f}')
    print(f'\n되먹임 증폭배수 (closed-loop / one-step, 같은 가중치)')
    for lab, c in (('전압', 'amp_V'), ('농도', 'amp_c')):
        v = raw[raw.model == L_AR][c]
        print(f'  {lab} {v.mean():.1f}배   seed 별 {v.round(1).tolist()}')

    out = pd.DataFrame(index=mu.index)
    for c in cols:
        out[c + '_mean'] = mu[c]
        out[c + '_sd'] = sd[c]
        out[c + '_min'] = lo[c]
        out[c + '_max'] = hi[c]
    out['params_k'] = par
    for p in (os.path.join(OUT_DIR, 'ar_vs_direct.csv'),
              os.path.join(OUT_DIR, 'ar_vs_direct_new.csv')):
        try:
            out.to_csv(p, encoding='utf-8-sig')
            raw.to_csv(p.replace('.csv', '_byseed.csv'), index=False,
                       encoding='utf-8-sig')
            print(f'\nsaved: {os.path.basename(p)}  +  '
                  f'{os.path.basename(p).replace(".csv", "_byseed.csv")}')
            return
        except PermissionError:
            continue
    print('\n[!!] CSV 저장 실패 — 엑셀에서 닫고 다시 돌릴 것')


if __name__ == '__main__':
    main()
