# -*- coding: utf-8 -*-
"""대리모델 재학습 실험 — 최적화 해에서 COMSOL 과 어긋나는 원인이 '데이터' 인지 '시간 해상도' 인지 가른다.

    variant A   격자 60 분 + 학습셋에 COMSOL 최적화 · 검증 운전 추가
    variant B   격자 20 분 + 기존 학습셋 (lhs6, vstart, ratio, ratio2)
    variant AB  격자 20 분 + 추가 데이터
    base        기존 캐시 (surrogate_TFT_c_mix.pt, 60 분, 기존 학습셋)

    홀드아웃 (어느 모델도 학습하지 않음): j2p_*, j2e5_*, j2e7_*, j2c_*  — 가장 최근 최적화 선택 해
    평가 지표는 parity plot(곡선)이 아니라 최적화가 실제로 쓰는 폐루프 적분 지표:
        총 운전시간, 탈염량, 순에너지  (충전 종료 4.2 V 교차를 격자 안에서 선형보간, 방전은 연속 시간)
        그리고 같은 농도 정전류 대비 % (대리모델은 자기 정전류 롤아웃, COMSOL 은 COMSOL 정전류) 의 오차.

    왜 (260917 진단, memory sbd-joint-optimization)
        대리모델은 전류가 바뀌는 운전에서 탈염량을 +2~3 % 과대예측(정전류 −0.9 %)하고,
        같은 탈염량 비교에서 최적화 해가 시간 −5 %p, 에너지 −3~−4 %p 가짜 이득을 얻었다.

    -> _O_result/surrogate_TFT_c_mix_<variant>.pt
       _O_result/retrain_expt_<variant>.csv    운전별 COMSOL / 대리모델 지표

    python 2_surrogate/retrain_expt.py A|B|AB|base
"""
import glob
import os
import sys
import time
import warnings

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
BASE = _ROOT
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, '2_surrogate'))
sys.path.insert(0, os.path.join(BASE, '3_optimize'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 최적화 워커가 import 할 때는 sys.argv 가 최적화 스크립트 것이라 환경변수로 받는다
VARIANT = os.environ.get('RETRAIN_VARIANT') or (sys.argv[1] if len(sys.argv) > 1 else 'A')
DT = {'A': 60.0, 'B': 20.0, 'AB': 20.0, 'base': 60.0}[VARIANT]
EXTRA = VARIANT in ('A', 'AB')

import surrogate as S
S.DT_MIN = DT                                   # optimize_nsga2 가 import 시 DT_H 를 여기서 읽는다
import optimize_nsga2 as O
import torch
import export_cycle_table as E

OUT = os.path.join(BASE, '_O_result')
HOLDOUT = ('j2p', 'j2e5', 'j2e7', 'j2c')
CC_TAGS = ['f10base', 'f10base2', 'f2scc_a', 'f2scc_b', 'f2scc_c']
ORDER = ['I_chg1', 'I_dis1', 'I_chg2', 'I_dis2', 'I_chg3', 'I_dis3']
EXTRA_SKIP = ('orrtest',)                        # i0 를 바꾼 민감도 판은 물리가 달라 뺀다


def fam(tag):
    import re
    return re.sub(r'_[abcr]$', '', tag)


def extra_sets():
    keys = []
    for tag in E.COMSOL:
        if fam(tag) in HOLDOUT or tag in EXTRA_SKIP:
            continue
        d = os.path.join(BASE, '_E_seawater_%s_v49' % tag)
        if os.path.isdir(d):
            S.DATASETS[tag] = d
            keys.append(tag)
    return keys


def train_model():
    use = list(S.USE) + (extra_sets() if EXTRA else [])
    pool = S.load(use, dt_min=DT)
    train, val, test = S.split(pool)
    scaler = S.Scaler().fit(train)
    rows = []
    for r in pool.values():
        g = r[r['phase'] == 'charge']
        g = g[g['cycle'] == g['cycle'].min()]
        if len(g):
            rows.append((float(r['c_in'].iloc[0]), float(abs(g['I'].iloc[0])), float(g['Vcell'].iloc[0])))
    a = np.array(rows)
    A = np.c_[np.ones(len(a)), a[:, 0], a[:, 1]]
    coef, *_ = np.linalg.lstsq(A, a[:, 2], rcond=None)
    m = S.build(O.MODEL_NAME, epochs=O.EPOCHS, seed=O.SURR_SEED)
    cache = os.path.join(OUT, 'surrogate_%s_%s_%s.pt' % (O.MODEL_NAME, S.CONC_TARGET, VARIANT))
    if VARIANT == 'base':
        cache = O.CACHE
    if os.path.exists(cache):
        print('[캐시] %s' % os.path.basename(cache), flush=True)
        m.net = S.TORCH_MODELS[O.MODEL_NAME](**m.kwargs).to(S.DEVICE)
        m.net.load_state_dict(torch.load(cache, map_location=S.DEVICE))
        m.net.eval()
    else:
        print('[학습] %s  %d조건 (val %d), 격자 %.0f분, %d에폭' % (VARIANT, len(train), len(val), DT, O.EPOCHS), flush=True)
        t0 = time.time()
        m.fit(train, val, scaler)
        torch.save(m.net.state_dict(), cache)
        print('  %.0f초, 저장 %s' % (time.time() - t0, cache), flush=True)
    return m, scaler, coef


def closed_loop(m, scaler, coef, x, c_in):
    d = O.rollout(m.net, scaler, coef, np.asarray(x, float), c_in, full=True)
    if not d['ok']:
        return None
    st, V, cm = d['steps'], d['V'], d['c_mix']
    # rollout 이 ROLLOUT_ITER 안에 수렴하지 못하면 steps 는 마지막 재추정값이라 V 보다 길다 -> 실패로 본다
    # (260917 모델 B 최적화가 이 경우 np.interp 길이 오류로 멈췄다)
    if len(V) < sum(st) or min(st) < 1:
        return None
    DTH = O.DT_H
    e = np.r_[0, np.cumsum(st)]
    hours = E_chg = E_dis = 0.0
    removed = np.nan
    q = 0.0
    for p in range(len(st)):
        a, k, I = e[p], st[p], x[p]
        Vp = V[a:a + k]
        tt = np.arange(k) * DTH
        if p % 2 == 0:
            if k >= 2 and Vp[-1] >= O.V_MAX and Vp[-2] < O.V_MAX:
                tc = (k - 2 + (O.V_MAX - Vp[-2]) / (Vp[-1] - Vp[-2])) * DTH
            else:
                tc = k * DTH
            tg = np.r_[tt[tt < tc], tc]
            E_chg += I * np.trapezoid(np.interp(tg, tt, Vp), tg)
            hours += tc
            q = I * tc
            if p == len(st) - 2:
                removed = c_in - float(np.interp(tc, tt, cm[a:a + k]))
        else:
            td = O.DIS_FRAC * q / I
            tg = np.r_[tt[tt < td], td]
            E_dis += I * np.trapezoid(np.interp(tg, tt, Vp), tg)
            hours += td
    global LAST_E
    LAST_E = (E_chg, E_dis)
    return hours, removed, E_chg - E_dis


def comsol_pairs():
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        t = E.comsol_rows()
    g = t.groupby(['protocol', 'c_in', 'cond_id']).apply(lambda z: pd.Series(dict(
        n=len(z), hours=z.hours.sum(), removed=z.removed_total.iloc[0], E_net=z.E_mWh.sum()))).reset_index()
    g = g[(g.n == 6) & (g.hours > 60)]
    cur = []
    for p in glob.glob(os.path.join(BASE, 'cond_cycle_*_seawater.csv')):
        tag = os.path.basename(p)[11:-13]
        c = pd.read_csv(p, encoding='utf-8-sig')
        c.columns = [k.lstrip('\ufeff') for k in c.columns]
        if 'I_chg1' not in c:
            if 'I[mA]' not in c:
                continue
            for n in ORDER:
                c[n] = c['I[mA]']
        c['protocol'] = tag
        cur.append(c[['protocol', 'cond_id'] + ORDER])
    D = g.merge(pd.concat(cur), on=['protocol', 'cond_id'])
    D['fam'] = D.protocol.map(fam)
    return D[D.fam.isin(HOLDOUT) | D.protocol.isin(CC_TAGS)].reset_index(drop=True)


def main():
    m, scaler, coef = train_model()
    D = comsol_pairs()
    out = []
    for _, r in D.iterrows():
        s = closed_loop(m, scaler, coef, r[ORDER].to_numpy(float), float(r.c_in))
        out.append(s if s is not None else (np.nan, np.nan, np.nan))
    D['hours_s'], D['removed_s'], D['E_net_s'] = np.array(out).T
    D.to_csv(os.path.join(OUT, 'retrain_expt_%s.csv' % VARIANT), index=False, encoding='utf-8-sig')

    # 정전류 대비 % 오차
    C = D[D.protocol.isin(CC_TAGS)]
    H = D[D.fam.isin(HOLDOUT)].copy()
    rel = []
    print('롤아웃 실패: 정전류 %d/%d, 홀드아웃 %d/%d'
          % (C.hours_s.isna().sum(), len(C), H.hours_s.isna().sum(), len(H)))
    for _, r in H.iterrows():
        b = C[np.isclose(C.c_in, r.c_in)].dropna()
        # 새 모델 롤아웃이 실패해 기준 곡선이 비는 농도는 비교하지 않는다 (개수는 위에 출력)
        if len(b) < 2 or not np.isfinite(r.hours_s):
            rel.append((np.nan,) * 4)
            continue
        bc, bs = b.sort_values('removed'), b.sort_values('removed_s')
        tc = 100 * (r.hours / np.interp(r.removed, bc.removed, bc.hours) - 1)
        ec = 100 * (r.E_net / np.interp(r.removed, bc.removed, bc.E_net) - 1)
        ts = 100 * (r.hours_s / np.interp(r.removed_s, bs.removed_s, bs.hours_s) - 1)
        es = 100 * (r.E_net_s / np.interp(r.removed_s, bs.removed_s, bs.E_net_s) - 1)
        rel.append((tc, ec, ts, es))
    R = np.array(rel)
    H['t_c'], H['e_c'], H['t_s'], H['e_s'] = R.T
    for k in ('hours', 'removed', 'E_net'):
        H['err_' + k] = 100 * (H[k + '_s'] / H[k] - 1)
        C = C.assign(**{'err_' + k: 100 * (C[k + '_s'] / C[k] - 1)})
    print('\n==== %s (격자 %.0f분, 추가데이터 %s) ====' % (VARIANT, DT, EXTRA))
    print('절대 오차 (대리모델/COMSOL-1, %%) 평균 [sd]')
    for lab, q in (('정전류', C), ('홀드아웃 최적화 해', H)):
        print('  %-18s n=%3d  시간 %+.1f [%.1f]  탈염 %+.1f [%.1f]  순에너지 %+.1f [%.1f]'
              % (lab, len(q), q.err_hours.mean(), q.err_hours.std(), q.err_removed.mean(), q.err_removed.std(),
                 q.err_E_net.mean(), q.err_E_net.std()))
    dt, de = H.t_s - H.t_c, H.e_s - H.e_c
    print('정전류 대비 %% 오차 (대리모델 - COMSOL), 홀드아웃 n=%d:  시간 %+.2f [sd %.2f]   에너지 %+.2f [sd %.2f]'
          % (dt.notna().sum(), dt.mean(), dt.std(), de.mean(), de.std()))
    print('  계열별 (시간, 에너지):')
    for f_, q in H.groupby('fam'):
        print('    %-5s n=%2d  COMSOL %+.1f / %+.1f   대리모델 %+.1f / %+.1f'
              % (f_, len(q), q.t_c.mean(), q.e_c.mean(), q.t_s.mean(), q.e_s.mean()))


if __name__ == '__main__':
    main()
