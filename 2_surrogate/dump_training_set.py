# -*- coding: utf-8 -*-
"""추출된 539조건 전부를 그대로 뽑는다 — 직접 확인할 수 있게.
   -> _S_result/dataset_conditions.csv   조건 한 줄씩 (539행)
      _S_result/dataset_phases.csv       충전/방전 구간 한 줄씩 (3212행)

    used 열이 학습 사용 여부, n_cycle_used 가 몇 사이클까지 썼는지다.
    (학습에 들어간 것만 보려면 used == True 로 거르면 된다.)

왜 (사용자 요청 260928)
    표에 쓴 539조건은 '추출된 전부'이고, 학습에 들어간 것은 그 일부다.
    규칙을 글로 설명하는 대신 load() 가 실제로 남긴 것을 파일로 낸다.

★ 규칙을 베끼지 않는다
    surrogate.load() 를 그대로 불러 살아남은 것을 받는다. 필터를 흉내내면
    앞단 정리(재격자 동기화, 길이 하한, 결측 처리)를 놓쳐 숫자가 어긋난다.

    py 2_surrogate/dump_training_set.py
"""
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '2_surrogate'))

import numpy as np
import pandas as pd
import surrogate as S

OUT = os.path.join(S.BASE, '_S_result')


def main():
    # 학습에 살아남은 것 — 규칙을 베끼지 않고 load() 를 진실로 삼는다
    kept = S.load(verbose=False)
    used = {k: int(v['cycle'].max()) for k, v in kept.items()}
    print('기준: surrogate.TRUNC_V_MAX = %.2f V — 충전 종료전압이 이보다 낮은'
          ' 사이클부터 뒤를 버린다' % S.TRUNC_V_MAX)

    crow, prow = [], []
    for key in S.USE:
        d = S.DATASETS[key]
        base = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            if not f.endswith('.csv'):
                continue
            run = pd.read_csv(os.path.join(d, f))
            if not {'phase', 'cycle', 'Vcell'} <= set(run.columns):
                continue
            num = f.replace('cond_', '').replace('.csv', '')
            name = '%s#%s' % (base, num)
            ncy_used = used.get(name, 0)
            c_in = float(run['c_in'].iloc[0])
            ch = run[run.phase == 'charge']
            di = run[run.phase == 'discharge']
            r = dict(key=name, dataset=key, cond=f.replace('.csv', ''),
                     c_in=c_in, used=name in used, n_cycle_used=ncy_used,
                     n_cycle=int(run['cycle'].max()),
                     n_charge=int(ch.cycle.nunique()),
                     n_discharge=int(di.cycle.nunique()),
                     n_row=len(run))
            # 사이클별 전류와 충전 종료전압 — 왜 잘렸는지 여기서 바로 보인다
            for cy in (1, 2, 3):
                g = ch[ch.cycle == cy]
                h = di[di.cycle == cy]
                r['I_chg%d' % cy] = (abs(float(g['I_applied'].iloc[0]))
                                     if len(g) else np.nan)
                r['I_dis%d' % cy] = (abs(float(h['I_applied'].iloc[0]))
                                     if len(h) else np.nan)
                r['Vend_chg%d' % cy] = (float(g['Vcell'].iloc[-1])
                                        if len(g) else np.nan)
            crow.append(r)

            for (cy, ph), g in run.groupby(['cycle', 'phase'], sort=True):
                prow.append(dict(key=name, dataset=key,
                                 cond=f.replace('.csv', ''), c_in=c_in,
                                 cycle=int(cy), phase=str(ph),
                                 I_mA=abs(float(g['I_applied'].iloc[0])),
                                 n_row=len(g),
                                 V_end=float(g['Vcell'].iloc[-1]),
                                 used=(name in used and int(cy) <= ncy_used)))

    C = pd.DataFrame(crow).sort_values(['dataset', 'cond'])
    P = pd.DataFrame(prow).sort_values(['dataset', 'cond', 'cycle', 'phase'])
    pc = os.path.join(OUT, 'dataset_conditions.csv')
    pp = os.path.join(OUT, 'dataset_phases.csv')
    C.to_csv(pc, index=False, encoding='utf-8-sig')
    P.to_csv(pp, index=False, encoding='utf-8-sig')

    print('\n추출 조건 %d개   (학습 사용 %d / 제외 %d)'
          % (len(C), int(C.used.sum()), int((~C.used).sum())))
    print('\n데이터셋별  추출 / 사용')
    for k, g in C.groupby('dataset'):
        print('  %-8s %3d / %3d' % (k, len(g), int(g.used.sum())))
    print('\n조건당 사용 사이클 수 (0 = 학습 제외)')
    print(C.n_cycle_used.value_counts().sort_index().to_string())
    for lab, m in (('전체', P.index == P.index), ('학습 사용분', P.used)):
        n_ch = int(((P.phase == 'charge') & m).sum())
        n_di = int(((P.phase == 'discharge') & m).sum())
        print('%-10s 충전 %4d   방전 %4d   합 %4d' % (lab, n_ch, n_di, n_ch + n_di))
    print('\nsaved:', pc)
    print('saved:', pp)


if __name__ == '__main__':
    main()
