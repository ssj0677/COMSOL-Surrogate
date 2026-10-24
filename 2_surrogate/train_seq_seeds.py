# -*- coding: utf-8 -*-
"""시퀀스 모델 3종을 seed 별로 학습해 가중치를 남긴다.

왜 필요한가
    지금 저장된 가중치는 figures_{모델}.pt 하나뿐이고 전부 seed 0 이다.
    그래서 plot_seq_metrics 의 (a) '궤적을 따라간 오차' 가 seed 하나에만
    걸려 있다 — 그 그림의 핵심 발견(Transformer 가 뒤로 갈수록 무너진다)이
    우연인지 아닌지 확인할 방법이 없었다.

    compare_all.py 가 seed 3개를 돌리긴 하지만 가중치를 저장하지 않아
    지표(RMSE/MAE)만 남고 시점별 예측을 다시 못 만든다. 그래서 따로 둔다.

저장 이름
    figures_{모델}_s{seed}.pt      ← 이 스크립트
    figures_{모델}.pt              ← 기존 seed 0 (건드리지 않는다)

    py 2_surrogate/train_seq_seeds.py [seed ...]      기본 1 2
"""
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '2_surrogate'))

import torch

import surrogate as S

OUT_DIR = os.path.join(S.BASE, '_S_result')
CKPT = os.path.join(OUT_DIR, 'ar_models')
# MLP 도 넣는다 — 5모델 표를 한 실행으로 통일하기 위해서다.
# XGBoost 는 torch 모델이 아니고 4초면 학습되므로 저장하지 않고
# 필요할 때 그 자리에서 학습한다.
SEQ = ['MLP', 'LSTM', 'Transformer', 'TFT']
EPOCHS = 150


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2]
    os.makedirs(CKPT, exist_ok=True)
    pool = S.load(list(S.USE))
    train, val, test = S.split(pool)
    scaler = S.Scaler().fit(train)
    print(f'\nseed {seeds}  x  {SEQ}   ({EPOCHS} 에폭)\n')

    for seed in seeds:
        for name in SEQ:
            p = os.path.join(CKPT, f'figures_{name}_s{seed}.pt')
            if os.path.exists(p):
                print(f'  [{name} s{seed}] 이미 있음 — 건너뜀')
                continue
            t0 = time.time()
            m = S.build(name, epochs=EPOCHS, seed=seed).fit(train, val, scaler)
            torch.save(m.net.state_dict(), p)
            print(f'  [{name} s{seed}] {time.time() - t0:.0f}s  -> '
                  f'{os.path.basename(p)}', flush=True)
    print('\n끝. plot_seq_metrics.py 를 다시 돌릴 것.')


if __name__ == '__main__':
    main()
