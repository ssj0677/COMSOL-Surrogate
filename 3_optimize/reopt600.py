# -*- coding: utf-8 -*-
"""600 mM 소규모 재최적화 — 방전 조건을 바로잡은 평가 체계로 '개선 가능성' 을 탐색한다.

    왜 (260917 밤)
        기존 COMSOL 658운전 중 525 에서 방전이 조기 종료됐고 (1_extract/comsol_setup.py DIS_STRICT_STEPS 주석),
        보정식의 1사이클 방전전하비 0.95 는 그 인공물을 따라간 것이었다. 기존 j2e5 는 오류가 섞인 체계에서 고른 점이다.
        -> 설계 방전 0.98 을 그대로 쓰고, 방전이 정상 종료한 COMSOL 운전만으로 학습한 종료 판정·최종 농도 보정
           (term_corr.py, TERMCORR_CLEAN=1 -> termcorr_model_clean.pkl 'all') 을 rollout 안에 넣어 다시 찾는다.

    평가 (한 rollout 에서 세 값)
        T = 총 운전시간 [h],  N = 탈염량 (3사이클 끝, 농도 보정),  E_net = 충전E - 회수E [mWh]
        제약 eta = 회수E/충전E <= 0.80 (optimize_joint3 와 같음)

    단계
        base   충·방전 분리 정전류 격자 (충전 0.55-1.20 x 0.01, 방전 0.5-1.5 x 0.1) + 방전=충전
               -> _O_result/reopt600_sepcc.csv
        opt    6전류 3목적 NSGA-II (POP 120, GEN 50), −5 % 같은 에너지 필수 제약 없음
               -> _O_result/reopt600_pareto.csv
        cmp    전선 각 점을 같은 탈염량의 '분리 정전류 전선' 과 비교
               gE = 같은 시간에서의 에너지 차 %, gT = 같은 에너지에서의 시간 차 %  (음수 = 스케줄이 유리)
               -> reopt600_pareto.csv 에 열 추가
        후보 선택과 COMSOL 평가는 결과를 보고 따로 정한다 (시간 우선 · 에너지 우선 · 절충).
        eq5    같은 3목적 + 제약 gEq_pred <= -5 % (같은 탈염량에서 방전 = 충전 정전류 대비 순에너지; 원래 연구 질문)
               -> _O_result/*_pareto_gEq5.csv.  대표점은 그 전선에서 이상점 최근접 (사용자 260918).
        opt5   같은 3목적 + 제약 gE_pred <= -5 % (분리 정전류 전선 대비, 보간 범위 밖 = 위반)   (260918 사용자 결정)
               -> _O_result/reopt600_pareto_gE5.csv.  5 % 는 후보 선택 설계 조건이지 오차 보증이 아니다.
               모든 충전 전하 <= Q_MAX (76 mAh; COMSOL 러너 상한 80 mAh 에 5 % 여유) — 대리모델 기준선 전선도 같은 조건.
               COMSOL 판정: gE <= -5 재현 / -5 < gE < 0 절감만 확인 / gE >= 0 재현 안 됨. 기준선·제약은 결과 보고 바꾸지 않는다.

    SURR_VARIANT=B TERMCORR_CLEAN=1 py 3_optimize/reopt600.py [base] [opt] [cmp] [workers=6]
"""
import contextlib
import io
import os
import pickle
import sys
import time
import warnings
from multiprocessing import get_context

os.environ.setdefault('SURR_VARIANT', 'B')
os.environ.setdefault('TERMCORR_CLEAN', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(_ROOT, '_O_result')
C_IN = float(os.environ.get('REOPT_CIN', '600'))
# 파일 이름: 600 mM 은 기존 이름 그대로 (reopt600_*), 다른 농도는 reopt_c<농도>_*  (260918 농도별 확장)
PFX = 'reopt600' if C_IN == 600.0 else 'reopt_c%g' % C_IN
ETA_MAX = 0.80
# COMSOL 러너의 충전 tlist 상한은 80 mAh (run_extraction CAP_CEIL_MAH). 대리모델 rollout 에는 없어서 충전 0.64 mA 정전류가
# 91 mAh 까지 가고 후보도 2사이클 80-87 mAh 로 갔다 (260918). 4.2 V 종료 운전끼리만 비교하도록 모든 충전 <= Q_MAX.
Q_MAX = 76.0
POP, GEN, WORKERS = 120, 50, 6
SEED = int(os.environ.get("REOPT_SEED", "1"))     # 재실행용 (같은 시드면 결과가 같다)
ARGS = [a for a in sys.argv[1:] if '=' not in a] or ['base', 'opt', 'cmp']
for a in sys.argv[1:]:
    if a.startswith('workers='):
        WORKERS = int(a.split('=')[1])
    elif a.startswith('gen='):
        GEN = int(a.split('=')[1])
_G = {}


def _init():
    import torch
    torch.set_num_threads(1)
    sys.argv = ['term_corr.py', 'none']
    with contextlib.redirect_stdout(io.StringIO()):
        import term_corr as T
    with open(os.path.join(OUT, 'termcorr_model_clean.pkl'), 'rb') as f:
        M = pickle.load(f)
    _G.update(T=T, m=M['models']['all'], cm=M['cmodels']['all'])
    p = os.path.join(OUT, '%s_sepcc.csv' % PFX)
    if os.path.exists(p):
        B = pd.read_csv(p, encoding='utf-8-sig')
        _G['fams'] = [(g.removed.to_numpy(float), g.hours.to_numpy(float), g.E_net.to_numpy(float))
                      for _, g in B[~B['eq'] & (B.get('Q_max', 0) <= Q_MAX)].sort_values('removed').groupby('I_dis')]
        # 방전 = 충전 정전류 (관행) 곡선 — 원래 연구 질문의 기준선 (260918 사용자 확인)
        e = B[B['eq'] & (B.get('Q_max', 0) <= Q_MAX)].sort_values('removed')
        _G['eqcc'] = (e.removed.to_numpy(float), e.E_net.to_numpy(float), e.hours.to_numpy(float))


def gE_of(N, T, E):
    """분리 정전류 전선 대비 같은 N·T 에서의 에너지 차 %. 보간 범위 밖이면 None (외삽 없음)."""
    pts = []
    for rr, hh, ee in _G['fams']:
        if len(rr) >= 2 and rr[0] <= N <= rr[-1]:
            pts.append((float(np.interp(N, rr, hh)), float(np.interp(N, rr, ee))))
    pts.sort()
    front, best = [], np.inf
    for h, e in pts:
        if e < best:
            front.append((h, e))
            best = e
    if len(front) < 2:
        return None
    h, e = np.array([p[0] for p in front]), np.array([p[1] for p in front])
    if not h[0] <= T <= h[-1]:
        return None
    return 100 * (E / np.interp(T, h, e) - 1)


def gEq_of(N, E):
    """같은 탈염량에서 방전 = 충전 정전류 대비 순에너지 차 % (시간은 맞추지 않는다 — j2e5 와 같은 정의)."""
    rr, ee, _ = _G['eqcc']
    if len(rr) < 2 or not rr[0] <= N <= rr[-1]:
        return None
    return 100 * (E / np.interp(N, rr, ee) - 1)


def gTq_of(N):
    """같은 탈염량에서 방전 = 충전 정전류의 운전시간 [h]. 보간 범위 밖이면 None."""
    rr, _ee, hh = _G['eqcc']
    if len(rr) < 2 or not rr[0] <= N <= rr[-1]:
        return None
    return float(np.interp(N, rr, hh))


def evaluate(x):
    if not _G:
        _init()
    x = np.asarray(x, float)
    try:
        r = _G['T'].rollout(x, C_IN, model=_G['m'], cmodel=_G['cm'])
    except Exception:
        r = None
    if r is None:
        return None
    E = np.asarray(r['E'], float)
    e_chg, e_dis = E[0::2].sum(), -E[1::2].sum()
    out = dict(hours=r['hours'], removed=r['removed'], E_net=r['E_net'], E_chg=e_chg, E_dis=e_dis,
               eta=e_dis / e_chg if e_chg > 0 else np.nan,
               Q_max=float(max(x[p] * r['hrs'][p] for p in (0, 2, 4))))
    if 'fams' in _G:
        g = gE_of(out['removed'], out['hours'], out['E_net'])
        out['gE_pred'] = np.nan if g is None else g
        g2 = gEq_of(out['removed'], out['E_net'])
        out['gEq_pred'] = np.nan if g2 is None else g2
        # 같은 탈염량 관행 정전류 대비 운전시간 차 % (260920: 'better than CC' 제약용)
        t0 = gTq_of(out['removed'])
        out['gTq_pred'] = np.nan if t0 is None else 100 * (out['hours'] / t0 - 1)
    return out


def pool():
    return get_context('spawn').Pool(WORKERS, initializer=_init)


# ─────────────────────────────────────────── base
def base():
    xs, lab = [], []
    for ic in np.round(np.arange(0.55, 1.2001, 0.01), 3):
        for idis in list(np.round(np.arange(0.5, 1.5001, 0.1), 2)) + ['eq']:
            d = ic if idis == 'eq' else idis
            xs.append([ic, d] * 3)
            lab.append((ic, d, idis == 'eq'))
    t0 = time.time()
    with pool() as p:
        res = p.map(evaluate, xs, chunksize=4)
    rows = [dict(I_ch=a, I_dis=b, eq=e, **r) for (a, b, e), r in zip(lab, res) if r is not None]
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(OUT, '%s_sepcc.csv' % PFX), index=False, encoding='utf-8-sig')
    print('base %d/%d 성공 (%.0f s)' % (len(B), len(xs), time.time() - t0), flush=True)


# ─────────────────────────────────────────── opt
def opt(gE_max=None, ref='sep'):
    """gE_max 가 있으면 에너지 제약을 더한다. ref='sep' 이면 분리 정전류 전선 대비 gE_pred,
    ref='eq' 이면 방전 = 충전 정전류 대비 gEq_pred (원래 연구 질문의 기준, 260918). 보간 범위 밖이면 위반."""
    gkey = 'gE_pred' if ref == 'sep' else 'gEq_pred'
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    P = pool()
    t0 = time.time()
    gen = [0]

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=6, n_obj=3, n_ieq_constr=2 if gE_max is None else 3,
                             xl=np.full(6, 0.5), xu=np.full(6, 1.5))

        def _evaluate(self, X, out, *a, **k):
            res = P.map(evaluate, list(X), chunksize=4)
            f = np.full((len(X), 3), 1e6)
            g = np.ones((len(X), self.n_ieq_constr)) * 100
            for i, r in enumerate(res):
                if r is not None and np.isfinite(r['eta']):
                    f[i] = [r['hours'], -r['removed'], r['E_net']]
                    g[i, 0] = r['eta'] - ETA_MAX
                    g[i, 1] = r['Q_max'] - Q_MAX
                    if gE_max is not None:
                        g[i, 2] = (r[gkey] - gE_max) if np.isfinite(r[gkey]) else 100.0
            out['F'], out['G'] = f, g
            gen[0] += 1
            if gen[0] % 5 == 0:
                print('  gen %d  (%.0f s)' % (gen[0], time.time() - t0), flush=True)

    algo = NSGA2(pop_size=POP, sampling=LHS(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                 eliminate_duplicates=True)
    res = minimize(Prob(), algo, get_termination("n_gen", GEN), seed=SEED, verbose=False)
    X = [x for x, fv in zip(res.X, res.F) if fv[0] < 1e5]
    R = P.map(evaluate, X)
    P.close()
    rows = []
    for x, r in zip(X, R):
        if r is None or not r['eta'] <= ETA_MAX or not r['Q_max'] <= Q_MAX:
            continue
        if gE_max is not None and not (np.isfinite(r[gkey]) and r[gkey] <= gE_max):
            continue
        rows.append(dict(I_chg1=x[0], I_dis1=x[1], I_chg2=x[2], I_dis2=x[3], I_chg3=x[4], I_dis3=x[5], **r))
    F = pd.DataFrame(rows)
    name = ('%s_pareto.csv' % PFX) if gE_max is None else (
        '%s_pareto_%s%d.csv' % (PFX, 'gE' if ref == 'sep' else 'gEq', abs(gE_max)))
    F.to_csv(os.path.join(OUT, name), index=False, encoding='utf-8-sig')
    print('opt 전선 %d점 (%.0f s)' % (len(F), time.time() - t0), flush=True)


# ─────────────────────────────────────────── btc (better than constant current)
# 3목적을 유지한 채 '같은 탈염량의 관행 정전류(충전 = 방전) 보다 시간·에너지가 모두 낫다' 를 제약으로 건다
# (사용자 결정 260920). 2단계 (3목적 -> 등탈염량 2목적) 구성을 없애고 한 번의 최적화로 끝낸다.
#   목적   T 최소, N 최대, E_net 최소
#   제약   eta <= 0.80,  모든 충전 <= Q_MAX,  gTq_pred <= BTC_MARGIN,  gEq_pred <= BTC_MARGIN
#   gTq_pred, gEq_pred 는 같은 탈염량에서 보간한 정전류 대비 % (보간 범위 밖이면 위반으로 본다)
BTC_MARGIN = float(os.environ.get('REOPT_BTC', '0'))     # 0 = 정전류보다 나쁘지 않을 것


def btc():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    # rollout 시간 상한 — dT <= 0 제약상 정전류보다 긴 운전은 어차피 탈락한다. 그 꼬리를 일찍 버려
    # 평가 비용을 줄인다 (260920). 워커는 spawn 이라 환경변수로 넘긴다.
    B = pd.read_csv(os.path.join(OUT, '%s_sepcc.csv' % PFX), encoding='utf-8-sig')
    e = B[B['eq'] & (B.get('Q_max', 0) <= Q_MAX)]
    os.environ['ROLLOUT_TMAX'] = '%.1f' % (1.05 * float(e.hours.max()))
    print('btc: 정전류 최대 %.0f h -> rollout 상한 %s h' % (e.hours.max(), os.environ['ROLLOUT_TMAX']), flush=True)

    P = pool()
    t0 = time.time()
    gen = [0]

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=6, n_obj=3, n_ieq_constr=4, xl=np.full(6, 0.5), xu=np.full(6, 1.5))

        def _evaluate(self, X, out, *a, **k):
            res = P.map(evaluate, list(X), chunksize=4)
            f = np.full((len(X), 3), 1e6)
            g = np.ones((len(X), 4)) * 100
            for i, r in enumerate(res):
                if r is not None and np.isfinite(r['eta']):
                    f[i] = [r['hours'], -r['removed'], r['E_net']]
                    g[i, 0] = r['eta'] - ETA_MAX
                    g[i, 1] = r['Q_max'] - Q_MAX
                    g[i, 2] = (r['gTq_pred'] - BTC_MARGIN) if np.isfinite(r.get('gTq_pred', np.nan)) else 100.0
                    g[i, 3] = (r['gEq_pred'] - BTC_MARGIN) if np.isfinite(r.get('gEq_pred', np.nan)) else 100.0
            out['F'], out['G'] = f, g
            gen[0] += 1
            if gen[0] % 5 == 0:
                print('  gen %d  (%.0f s)' % (gen[0], time.time() - t0), flush=True)

    algo = NSGA2(pop_size=POP, sampling=LHS(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                 eliminate_duplicates=True)
    res = minimize(Prob(), algo, get_termination("n_gen", GEN), seed=SEED, verbose=False)
    X = [x for x, fv in zip(res.X, res.F) if fv[0] < 1e5]
    R = P.map(evaluate, X)
    P.close()
    rows = []
    for x, r in zip(X, R):
        if r is None or not r['eta'] <= ETA_MAX or not r['Q_max'] <= Q_MAX:
            continue
        if not (np.isfinite(r['gTq_pred']) and np.isfinite(r['gEq_pred'])):
            continue
        if r['gTq_pred'] > BTC_MARGIN or r['gEq_pred'] > BTC_MARGIN:
            continue
        rows.append(dict(I_chg1=x[0], I_dis1=x[1], I_chg2=x[2], I_dis2=x[3], I_chg3=x[4], I_dis3=x[5], **r))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, '%s_btc.csv' % PFX), index=False, encoding='utf-8-sig')
    if len(F):
        print('btc 전선 %d점 (%.0f s)  dT %.1f..%.1f %%, dE %.1f..%.1f %%'
              % (len(F), time.time() - t0, F.gTq_pred.min(), F.gTq_pred.max(),
                 F.gEq_pred.min(), F.gEq_pred.max()), flush=True)
    else:
        print('btc 전선 0점 (%.0f s) — 제약을 만족하는 해가 없다' % (time.time() - t0), flush=True)


# ─────────────────────────────────────────── cmp
def cc_front_at(B, R):
    """분리 정전류: 방전 계열마다 탈염량 R 에서 (시간, 에너지) 보간 -> 비지배 점을 시간순으로."""
    pts = []
    for d, g in B[~B['eq']].groupby('I_dis'):
        g = g.sort_values('removed')
        if len(g) < 2 or not g.removed.min() <= R <= g.removed.max():
            continue
        pts.append((float(np.interp(R, g.removed, g.hours)), float(np.interp(R, g.removed, g.E_net)), d))
    pts.sort()
    front, best = [], np.inf
    for h, e, d in pts:
        if e < best:
            front.append((h, e, d))
            best = e
    return front


def cmp():
    B = pd.read_csv(os.path.join(OUT, 'reopt600_sepcc.csv'), encoding='utf-8-sig')
    F = pd.read_csv(os.path.join(OUT, 'reopt600_pareto.csv'), encoding='utf-8-sig')
    eq = B[B['eq']].sort_values('removed')
    out = []
    for _, r in F.iterrows():
        fr = cc_front_at(B, r.removed)
        d = dict(gE=np.nan, gT=np.nan, dt_eq=np.nan, de_eq=np.nan, cc_dis_near=np.nan)
        if eq.removed.min() <= r.removed <= eq.removed.max():
            d['dt_eq'] = 100 * (r.hours / np.interp(r.removed, eq.removed, eq.hours) - 1)
            d['de_eq'] = 100 * (r.E_net / np.interp(r.removed, eq.removed, eq.E_net) - 1)
        if len(fr) >= 2:
            h = np.array([p[0] for p in fr])
            e = np.array([p[1] for p in fr])
            if h.min() <= r.hours <= h.max():
                d['gE'] = 100 * (r.E_net / np.interp(r.hours, h, e) - 1)
            if e.min() <= r.E_net <= e.max():
                d['gT'] = 100 * (r.hours / np.interp(r.E_net, e[::-1], h[::-1]) - 1)
            d['cc_dis_near'] = fr[int(np.argmin(np.abs(h - r.hours)))][2]
        out.append(d)
    F = pd.concat([F.drop(columns=[c for c in ('gE', 'gT', 'dt_eq', 'de_eq', 'cc_dis_near') if c in F]),
                   pd.DataFrame(out)], axis=1)
    F.to_csv(os.path.join(OUT, 'reopt600_pareto.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    cols = ['I_chg1', 'I_dis1', 'I_chg2', 'I_dis2', 'I_chg3', 'I_dis3', 'hours', 'removed', 'E_net', 'eta',
            'dt_eq', 'de_eq', 'gE', 'gT', 'cc_dis_near']
    print('전선 %d점, 분리 정전류 전선과 비교 가능 gE %d / gT %d' % (len(F), F.gE.notna().sum(), F.gT.notna().sum()))
    print('gE (같은 시간·탈염량 에너지 차 %%): %s' % F.gE.describe().round(2).to_dict())
    print('gT (같은 에너지·탈염량 시간 차 %%): %s' % F.gT.describe().round(2).to_dict())
    print(F.sort_values('gE').head(15)[cols].round(3).to_string(index=False))


# ─────────────────────────────────────────── slice (같은 탈염량 면)
# 3목적 전선을 탈염량 밴드로 자르면 농도당 4-15점뿐이고, 그 점들의 탈염량이 서로 달라
# (시간, 에너지) 평면에서 비지배 곡선이 되지 않는다 (260918). 그래서 탈염량을 제약으로 고정하고
# 2목적 (T, E) 으로 다시 최적화해 '같은 탈염량 절충 곡선' 을 따로 만든다.
N_TOL = float(os.environ.get('REOPT_NTOL', '0.01'))      # |N - N*| <= N_TOL * N*


def n_star():
    """기준 탈염량 N*. 환경변수 REOPT_NSTAR 가 있으면 그 값, 없으면 3목적 전선의 이상점 최근접 해."""
    if os.environ.get('REOPT_NSTAR'):
        return float(os.environ['REOPT_NSTAR'])
    F = pd.read_csv(os.path.join(OUT, '%s_pareto.csv' % PFX), encoding='utf-8-sig')
    F.columns = [c.strip().lstrip('﻿') for c in F.columns]
    if 'Q_max' in F:
        F = F[F.Q_max <= Q_MAX]
    F = F.reset_index(drop=True)

    def n(v):
        v = np.asarray(v, float)
        return (v - v.min()) / (np.ptp(v) or 1.0)
    d = np.sqrt(n(F.hours) ** 2 + (1 - n(F.removed)) ** 2 + n(F.E_net) ** 2)
    return float(F.loc[int(np.argmin(d)), 'removed'])


def slice_():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    N0 = n_star()
    print('slice: N* = %.2f mM, 허용 +-%.1f %%' % (N0, N_TOL * 100), flush=True)
    P = pool()
    t0 = time.time()
    gen = [0]

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=6, n_obj=2, n_ieq_constr=3, xl=np.full(6, 0.5), xu=np.full(6, 1.5))

        def _evaluate(self, X, out, *a, **k):
            res = P.map(evaluate, list(X), chunksize=4)
            f = np.full((len(X), 2), 1e6)
            g = np.ones((len(X), 3)) * 100
            for i, r in enumerate(res):
                if r is not None and np.isfinite(r['eta']):
                    f[i] = [r['hours'], r['E_net']]
                    g[i, 0] = r['eta'] - ETA_MAX
                    g[i, 1] = r['Q_max'] - Q_MAX
                    g[i, 2] = abs(r['removed'] - N0) - N_TOL * N0
            out['F'], out['G'] = f, g
            gen[0] += 1
            if gen[0] % 5 == 0:
                print('  gen %d  (%.0f s)' % (gen[0], time.time() - t0), flush=True)

    algo = NSGA2(pop_size=POP, sampling=LHS(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                 eliminate_duplicates=True)
    res = minimize(Prob(), algo, get_termination("n_gen", GEN), seed=SEED, verbose=False)
    X = [x for x, fv in zip(res.X, res.F) if fv[0] < 1e5]
    R = P.map(evaluate, X)
    P.close()
    rows = []
    for x, r in zip(X, R):
        if r is None or not r['eta'] <= ETA_MAX or not r['Q_max'] <= Q_MAX:
            continue
        if abs(r['removed'] - N0) > N_TOL * N0:
            continue
        rows.append(dict(I_chg1=x[0], I_dis1=x[1], I_chg2=x[2], I_dis2=x[3], I_chg3=x[4], I_dis3=x[5],
                         N_star=N0, **r))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, '%s_slice.csv' % PFX), index=False, encoding='utf-8-sig')
    print('slice 전선 %d점 (%.0f s)' % (len(F), time.time() - t0), flush=True)


if __name__ == '__main__':
    for step in ARGS:
        {'base': base, 'opt': opt, 'cmp': cmp, 'opt5': lambda: opt(gE_max=-5.0),
         'eq5': lambda: opt(gE_max=-5.0, ref='eq'), 'slice': slice_, 'btc': btc}[step]()
