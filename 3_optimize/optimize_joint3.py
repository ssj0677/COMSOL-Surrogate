# -*- coding: utf-8 -*-
"""충방전 6전류 동시(joint) 3목적 최적화 — 전류비 제약 없음.

    -> _O_result/pareto_joint3.csv     농도별 비지배해 (POP 120)
       _O_result/protocol_joint3.csv   농도별 utopia 해 (min-max 정규화 후 이상점 최근접)

왜 다시 돌리나
    2단계 순차 최적화(f2su)는 충전을 먼저 골라 고정하므로 탐색 공간을 제한한다는
    지적을 받았다. 6전류를 한 번에 최적화해 그 제약을 없앤다.
    9/13 판(optimize_nsga2_3obj.py, COMSOL f10uto3)과 목적·POP·GEN 은 같고,
    방전 <= 2.24 x 충전 제약(rmax)만 뺐다. 그 제약은 학습 데이터에 전류비가 1 근처뿐이던
    시절에 넣은 것이고, 지금은 ratio/ratio2 세트가 높은 비 영역을 덮는다.
    ★ 결과 해가 학습 범위 안에 있는지(전류비)를 반드시 확인한다 — 끝에 출력한다.
    ★ NSGA-II 는 휴리스틱이다. '전역 최적' 이 아니라 '동시 최적화' 라고 쓴다.

    물리 타당성 제약  eta = 회수E / 충전E <= ETA_MAX (0.80)
        첫 실행(제약 없음, *_invalid.csv)에서 650 M 전선 120개 중 38개가 eta 0.8~1.73
        (회수 > 충전, 순에너지 음수) 였고 utopia 도 그중 하나였다. 대리모델이 학습 범위
        밖 조합에서 방전 전압을 틀리게 낸 것을 최적화가 '최고 에너지 해' 로 골랐다.
        COMSOL 459조건의 eta 는 최대 0.790 (99% 0.781) 이라 0.80 은 관측 상한 바로 위다.
        다른 16농도는 제약 없이도 위반이 0개였다. 규칙은 모든 농도에 똑같이 건다.

    목적 (모두 최소화 형태)
        f1 = 총 운전시간 [h]        충전 3 + 방전 3
        f2 = -탈염량 [mol/m^3]
        f3 = 순에너지 [mWh]         충전 3 구간 ∫VIdt - 방전 3 구간 ∫|VI|dt

    ※ 에너지는 i0_orr 미보정 — 상대 비교로만 읽을 것.

    python 3_optimize/optimize_joint3.py [workers=17] [pop=120] [gen=50]
"""
import contextlib
import io
import os
import sys
import time
import warnings
from multiprocessing import get_context

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '3_optimize'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

POP, GEN, WORKERS = 120, 50, 17
ETA_MAX = 0.80
for a in sys.argv[1:]:
    if a.startswith('workers='):
        WORKERS = int(a.split('=')[1])
    elif a.startswith('pop='):
        POP = int(a.split('=')[1])
    elif a.startswith('gen='):
        GEN = int(a.split('=')[1])

_G = {}


def _init():
    """워커마다 대리모델을 한 번만 올린다.

    환경변수 SURR_VARIANT (예: B) 가 있으면 2_surrogate/retrain_expt.py 의 재학습 모델을 쓴다
    (260917: B = 격자 20 분 재학습. 60 분 격자 모델은 전류 가변 운전의 탈염량을 +2.6 % 과대예측했다).
    그때 metrics 는 격자 반올림 없이 충전 종료 4.2 V 교차를 보간한다.
    """
    import torch
    torch.set_num_threads(1)
    v = os.environ.get('SURR_VARIANT')
    if v:
        os.environ['RETRAIN_VARIANT'] = v
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '2_surrogate'))
        with contextlib.redirect_stdout(io.StringIO()):
            import retrain_expt as RX
            m, scaler, v0 = RX.train_model()
        _G.update(O=RX.O, net=m.net, scaler=scaler, v0=v0, RX=RX, m=m)
        return
    import optimize_nsga2 as O
    with contextlib.redirect_stdout(io.StringIO()):
        m, scaler, v0 = O.get_surrogate()
    _G.update(O=O, net=m.net, scaler=scaler, v0=v0)


def metrics(x, c_in):
    """전류 6개 -> 시간·탈염량·에너지. optimize_2stage.metrics 와 같은 정의. 실패면 None."""
    O = _G['O']
    if 'RX' in _G:                                   # 재학습 모델: 보간 지표
        r = _G['RX'].closed_loop(_G['m'], _G['scaler'], _G['v0'], np.asarray(x, float), c_in)
        if r is None or not all(np.isfinite(r)):
            return None
        h, rem, en = r
        E_chg, E_dis = _G['RX'].LAST_E if hasattr(_G['RX'], 'LAST_E') else (np.nan, np.nan)
        return dict(hours=h, removed=rem, t_chg=np.nan, t_dis=np.nan, E_chg=E_chg, E_dis=E_dis,
                    E_net=en, eta=E_dis / E_chg if E_chg and np.isfinite(E_chg) else np.nan)
    d = O.rollout(_G['net'], _G['scaler'], _G['v0'], x, c_in, full=True)
    if not d['ok'] or not np.isfinite(d['hours']) or not np.isfinite(d['removed']):
        return None
    st = d['steps']
    e = np.r_[0, np.cumsum(st)]
    pw = np.abs(d['V'] * d['I']) * O.DT_H
    E = [float(pw[e[p]:e[p + 1]].sum()) for p in range(len(st))]
    if not np.all(np.isfinite(E)):
        return None
    E_chg, E_dis = sum(E[0::2]), sum(E[1::2])
    return dict(hours=d['hours'], removed=d['removed'],
                t_chg=sum(st[0::2]) * O.DT_H, t_dis=sum(st[1::2]) * O.DT_H,
                E_chg=E_chg, E_dis=E_dis, E_net=E_chg - E_dis,
                eta=E_dis / E_chg if E_chg > 0 else np.nan)


def pick(cols, maximize):
    """min-max 정규화 후 이상점 최근접 (optimize_2stage.pick 과 같은 규칙)."""
    s = 0.0
    for v, mx in zip(cols, maximize):
        v = np.asarray(v, float)
        n = (v - v.min()) / (np.ptp(v) or 1.0)
        s = s + ((1 - n) if mx else n) ** 2
    return int(np.argmin(s))


def run_cin(c_in):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    O = _G['O']
    lo, hi = O.I_BOUNDS
    t0 = time.time()

    class Joint(Problem):
        def __init__(self):
            super().__init__(n_var=6, n_obj=3, n_ieq_constr=1,
                             xl=np.full(6, lo), xu=np.full(6, hi))

        def _evaluate(self, X, out, *a, **k):
            f = np.full((len(X), 3), 1e6)
            g = np.ones((len(X), 1))
            for i, x in enumerate(X):
                r = metrics(x, c_in)
                if r is not None and np.isfinite(r['eta']):
                    f[i] = [r['hours'], -r['removed'], r['E_net']]
                    g[i, 0] = r['eta'] - ETA_MAX
            out['F'] = f
            out['G'] = g

    algo = NSGA2(pop_size=POP, sampling=LHS(), crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(Joint(), algo, get_termination('n_gen', GEN), seed=O.SEED,
                   verbose=False)
    rows = []
    for x, fv in zip(res.X, res.F):
        if fv[0] >= 1e5:
            continue
        r = metrics(x, c_in)
        if r is None or not r['eta'] <= ETA_MAX:
            continue
        rows.append(dict(c_in=c_in, I_chg1=x[0], I_dis1=x[1], I_chg2=x[2],
                         I_dis2=x[3], I_chg3=x[4], I_dis3=x[5], **r))
    f = pd.DataFrame(rows)
    k = pick([f.hours, f.removed, f.E_net], [False, True, False])
    f['utopia'] = False
    f.loc[f.index[k], 'utopia'] = True
    return f, time.time() - t0


def main():
    import optimize_nsga2 as O
    levels = list(O.C_IN_LEVELS)
    print('동시 3목적 최적화  농도 %d개  워커 %d  POP %d x GEN %d  전류 %.1f~%.1f mA  '
          '전류비 제약 없음  eta <= %.2f' % (len(levels), WORKERS, POP, GEN, *O.I_BOUNDS, ETA_MAX),
          flush=True)
    print('  목적 = (총 운전시간 ↓, 탈염량 ↑, 순에너지 ↓)', flush=True)
    print('  ※ 에너지 축은 i0_orr 미보정 — 상대 비교로만 읽을 것', flush=True)
    t00 = time.time()
    F = []
    with get_context('spawn').Pool(WORKERS, initializer=_init) as pool:
        for f, sec in pool.imap_unordered(run_cin, levels):
            F.append(f)
            u = f[f.utopia].iloc[0]
            print('  c_in %6.1f  비지배 %3d점  utopia 충전 %.2f/%.2f/%.2f  방전 %.2f/%.2f/%.2f mA  '
                  '(%.0fs)' % (u.c_in, len(f), u.I_chg1, u.I_chg2, u.I_chg3,
                               u.I_dis1, u.I_dis2, u.I_dis3, sec), flush=True)
    fr = pd.concat(F, ignore_index=True).sort_values(['c_in', 'hours'])
    pr = fr[fr.utopia].drop(columns='utopia').sort_values('c_in').reset_index(drop=True)
    fr.to_csv(os.path.join(O.OUT_DIR, 'pareto_joint3.csv'), index=False, encoding='utf-8-sig')
    pr.to_csv(os.path.join(O.OUT_DIR, 'protocol_joint3.csv'), index=False, encoding='utf-8-sig')

    # 학습 범위 확인 — 사이클별 방전/충전 비
    ch = pr[['I_chg1', 'I_chg2', 'I_chg3']].to_numpy()
    di = pr[['I_dis1', 'I_dis2', 'I_dis3']].to_numpy()
    rt = (di / ch).ravel()
    print('\n총 %.1f분   비지배해 %d개' % ((time.time() - t00) / 60, len(fr)))
    print('  utopia 전류비(방전/충전, 51개): 중앙 %.2f, 최대 %.2f, 2.24 초과 %d개'
          % (np.median(rt), rt.max(), (rt > 2.24).sum()))
    print('  utopia 전류가 경계(0.5 / 1.5 mA)에 붙은 개수: %d / 102'
          % (np.isclose(np.r_[ch.ravel(), di.ravel()], O.I_BOUNDS[0], atol=.005).sum()
             + np.isclose(np.r_[ch.ravel(), di.ravel()], O.I_BOUNDS[1], atol=.005).sum()))
    print('  eta 범위 (전선 전체): %.3f ~ %.3f' % (fr.eta.min(), fr.eta.max()))
    print('  saved: pareto_joint3.csv, protocol_joint3.csv')


if __name__ == '__main__':
    main()
