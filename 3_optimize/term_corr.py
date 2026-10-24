# -*- coding: utf-8 -*-
"""충전 종료 판정 보정 (대리모델 B, 20 분 격자) — rollout 안에서 보정해 뒤 위상까지 다시 예측한다.

    왜 (260917, stepwise_j2e5.py)
        COMSOL 위상 길이를 넣으면 위상별 에너지 오차 < 1 %. 그러나 2·3사이클 충전 끝 전압을 0.07-0.13 V 낮게 예측해
        폐루프 충전이 +11~18 % 길어진다. 같은 시간의 탈염량은 과소예측이라 두 오차가 상쇄된다.
        -> 종료 시간만 맞추면 가려진 탈염 오차가 드러날 수 있으므로 보정 후 전체 rollout 으로 판정한다.

    단계
      1 build   COMSOL 전 운전(d600 제외)을 COMSOL 위상 길이로 개루프 예측 -> 충전 위상마다
                delta = 4.2 V - (COMSOL 종료 시각의 예측 전압), 같은 시각 예측 농도 오차
      2 fit     delta ~ 사전에 알 수 있는 변수 (사이클, 현재 충전 전류, 직전 방전 전류·시간, 직전 충전 전류,
                위상 시작 순전하, c_in). 운전 단위 GroupKFold 5.
      3 eval    rollout 의 종료 조건을 V >= 4.2 - delta_hat 로 바꿔 전체 폐루프 재평가 (out-of-fold 모델).
                보정 전후: 위상별 충전 시간, 탈염량, 순에너지, 그리고 정전류 / 가변 전류 그룹 간 오차 차이 (가짜 이득).
      d600 은 학습에 쓰지 않는다 (독립 검증, 4_validate/analyze_d600.py 이후 eval_d600).

    -> _O_result/termcorr_phases.csv, termcorr_cv.csv, termcorr_runs.csv, termcorr_model.pkl
    SURR_VARIANT=B py 3_optimize/term_corr.py [build] [fit] [eval] [d600]
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SURR_VARIANT', 'B')
import optimize_joint3 as J  # noqa: E402

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
BASE = _ROOT
OUT = os.path.join(BASE, '_O_result')
FEAT = ['cyc2', 'cyc3', 'I_ch', 'I_ch_prev', 'I_dis_prev', 'h_dis_prev', 'Q_start', 'c_in']
FEAT_C = FEAT + ['h_ch']          # 농도 보정은 그 충전 위상 길이도 쓴다 (폐루프에서는 보정된 길이)
# (폐기 260917 23:00) COMSOL 방전 전하비 사이클별 중앙값 (0.9503, 0.9789, 0.9810) 은 방전 조기 종료 인공물이었다
#   (1_extract/comsol_setup.py DIS_STRICT_STEPS 주석). 정상 종료 운전에서는 에너지를 +2.8 % 과대예측한다.
#   rollout 은 설계값 0.98 (O.DIS_FRAC) 을 그대로 쓴다.
# TERMCORR_CLEAN=1 : 방전이 정상 종료한 운전(세 사이클 전하비 >= CLEAN_MIN)만 학습·평가, 출력 파일에 _clean
CLEAN = os.environ.get('TERMCORR_CLEAN') == '1'
CLEAN_MIN = 0.972
SUF = '_clean' if CLEAN else ''
STEPS = sys.argv[1:] or ['build', 'fit', 'eval']

J._init()
O, net, scaler, v0 = J._G['O'], J._G['net'], J._G['scaler'], J._G['v0']
S = O.S
DT = O.DT_H
VMAX = O.V_MAX


def predict(currents, steps, c_in):
    I_seq = np.concatenate([np.full(k, (1.0 if p % 2 == 0 else -1.0) * currents[p]) for p, k in enumerate(steps)])
    T = len(I_seq)
    Q = np.r_[0.0, np.cumsum(I_seq[:-1])] * DT
    c_far = c_in - O.C_PER_MAH * Q
    raw = np.c_[np.zeros(T), np.zeros(T), I_seq, Q, np.full(T, c_in)]
    raw[0, 0] = float(v0[0] + v0[1] * c_in + v0[2] * currents[0])
    raw[0, 1] = c_in if S.CONC_TARGET == 'c_mix' else 0.0
    x = (raw - scaler.mean) / scaler.std
    with torch.no_grad():
        init = torch.as_tensor(x[:1, :S.N_OUT], dtype=torch.float32, device=S.DEVICE)
        exo = torch.as_tensor(x[None, :, S.N_OUT:], dtype=torch.float32, device=S.DEVICE)
        y = scaler.back(net(init, exo)[0].cpu().numpy())
    return y[:, 0], y[:, 1] + (c_far if S.CONC_TARGET == 'c_res' else 0.0)


def feats(p, currents, hrs, c_in):
    """충전 위상 p (0, 2, 4) 의 사전 변수. hrs = 앞 위상들의 실제 길이 [h]."""
    cyc = p // 2 + 1
    q = sum((1 if i % 2 == 0 else -1) * currents[i] * hrs[i] for i in range(p))
    return dict(cyc2=float(cyc == 2), cyc3=float(cyc == 3), I_ch=currents[p],
                I_ch_prev=currents[p - 2] if p >= 2 else 0.0, I_dis_prev=currents[p - 1] if p >= 1 else 0.0,
                h_dis_prev=hrs[p - 1] if p >= 1 else 0.0, Q_start=q, c_in=c_in)


def comsol_runs(only=None, exclude=('d600',)):
    t = pd.read_csv(os.path.join(OUT, 'cycle_table_comsol.csv'), encoding='utf-8-sig')
    t = t.assign(o=t.phase.eq('discharge').astype(int)).sort_values(['protocol', 'c_in', 'cond_id', 'cycle', 'o'])
    runs = []
    for (pr, c, cid), z in t.groupby(['protocol', 'c_in', 'cond_id'], sort=False):
        if len(z) != 6 or z.hours.sum() < 60:
            continue
        if only and not pr.startswith(only):
            continue
        if not only and any(pr.startswith(e) for e in exclude):
            continue
        ch = z[z.phase == 'charge']
        frac = z[z.phase == 'discharge'].Q_mAh.to_numpy(float) / ch.Q_mAh.to_numpy(float)
        if CLEAN and not only and not (frac >= CLEAN_MIN).all():
            continue
        runs.append(dict(run='%s|%g|%d' % (pr, c, cid), protocol=pr, c_in=float(c), cur=z.I_mA.to_numpy(float),
                         hrs=z.hours.to_numpy(float), E=z.E_mWh.to_numpy(float), cend=ch.c_mix_end.to_numpy(float),
                         removed=float(z.removed_total.iloc[0]), E_net=float(z.E_mWh.sum()),
                         const=bool(np.ptp(z.I_mA) < 2e-3)))
    return runs


# ─────────────────────────────────────────── 1 build
def build():
    rows = []
    runs = comsol_runs()
    t0 = time.time()
    for n, r in enumerate(runs):
        ends = np.cumsum(r['hrs'])
        bnd = np.r_[0, np.maximum.accumulate(np.maximum(np.round(ends / DT).astype(int), 1))]
        steps = list(np.maximum(np.diff(bnd), 2))
        bnd = np.r_[0, np.cumsum(steps)]
        V, cm = predict(r['cur'], steps, r['c_in'])
        for p in (0, 2, 4):
            a, b = bnd[p], bnd[p + 1]
            tt = np.arange(a, b) * DT - (bnd[p] * DT)             # 위상 안 시간 (격자)
            te = r['hrs'][p]                                     # COMSOL 위상 길이
            Vp, cp = V[a:b], cm[a:b]
            if te <= tt[-1]:
                Ve, ce = np.interp(te, tt, Vp), np.interp(te, tt, cp)
            else:                                                # 마지막 두 점으로 외삽
                s = (Vp[-1] - Vp[-2]) / DT
                Ve = Vp[-1] + s * (te - tt[-1])
                ce = cp[-1] + (cp[-1] - cp[-2]) / DT * (te - tt[-1])
            d = dict(run=r['run'], protocol=r['protocol'], const=r['const'], p=p, h_comsol=te,
                     delta=VMAX - Ve, c_err=ce - r['cend'][p // 2], V_pred_end=Ve)
            d.update(feats(p, r['cur'], r['hrs'], r['c_in']))
            d['h_ch'] = te
            rows.append(d)
        if n % 100 == 0:
            print('  build %d/%d  (%.0fs)' % (n, len(runs), time.time() - t0), flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, 'termcorr_phases%s.csv' % SUF), index=False, encoding='utf-8-sig')
    print('\n위상 %d (운전 %d)' % (len(P), P.run.nunique()))
    print(P.groupby(['p', 'const'])[['delta', 'c_err']].agg(['mean', 'std']).round(3).to_string())


# ─────────────────────────────────────────── 2 fit
def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20)


def fit():
    from sklearn.model_selection import GroupKFold
    P = pd.read_csv(os.path.join(OUT, 'termcorr_phases%s.csv' % SUF), encoding='utf-8-sig')
    runs = P.run.unique()
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(runs), np.arange(len(runs)) % 5))
    P['fold'] = P.run.map(fold_of)
    P['delta_cv'] = np.nan
    P['delta_cyc'] = np.nan
    for k in range(5):
        tr, te = P.fold != k, P.fold == k
        m = make_model().fit(P.loc[tr, FEAT], P.loc[tr, 'delta'])
        P.loc[te, 'delta_cv'] = m.predict(P.loc[te, FEAT])
        P.loc[te, 'delta_cyc'] = P.loc[te, 'p'].map(P[tr].groupby('p').delta.mean())
    for k, lab in (('delta_cv', '변수 모델'), ('delta_cyc', '사이클 평균')):
        e = P[k] - P.delta
        print('%-10s RMSE %.4f V  (보정 없음 %.4f)' % (lab, np.sqrt((e ** 2).mean()), np.sqrt((P.delta ** 2).mean())))
        print(P.assign(e=e).groupby(['p', 'const']).e.agg(lambda s: np.sqrt((s ** 2).mean())).round(4).to_string())
    P.to_csv(os.path.join(OUT, 'termcorr_cv%s.csv' % SUF), index=False, encoding='utf-8-sig')
    models = {k: make_model().fit(P.loc[P.fold != k, FEAT], P.loc[P.fold != k, 'delta']) for k in range(5)}
    models['all'] = make_model().fit(P[FEAT], P.delta)
    # 3사이클 끝 농도 오차. 대리모델은 자기 농도 예측을 입력으로 쓰지 않으므로 사후 보정해도 뒤 위상 예측은 바뀌지 않는다.
    Q = P[P.p == 4].copy()
    Q['c_cv'] = np.nan
    cmod = {}
    for k in range(5):
        tr, te = Q.fold != k, Q.fold == k
        cmod[k] = make_model().fit(Q.loc[tr, FEAT_C], Q.loc[tr, 'c_err'])
        Q.loc[te, 'c_cv'] = cmod[k].predict(Q.loc[te, FEAT_C])
    cmod['all'] = make_model().fit(Q[FEAT_C], Q.c_err)
    print('3사이클 끝 농도 RMSE %.2f mM (보정 없음 %.2f)' % (np.sqrt(((Q.c_cv - Q.c_err) ** 2).mean()),
                                                         np.sqrt((Q.c_err ** 2).mean())))
    with open(os.path.join(OUT, 'termcorr_model%s.pkl' % SUF), 'wb') as f:
        pickle.dump(dict(models=models, cmodels=cmod, fold_of=fold_of, FEAT=FEAT), f)


# ─────────────────────────────────────────── 3 폐루프 (보정 선택)
def rollout(cur, c_in, model=None, cmodel=None, dis_frac=None):
    n_ph = 6
    steps = [O.MAX_STEPS] * n_ph
    hrs = [0.0] * n_ph
    for _ in range(O.ROLLOUT_ITER + 3):
        V, cm = predict(cur, steps, c_in)
        e = np.r_[0, np.cumsum(steps)]
        new, h_new, bad = [], [], False
        for p in range(n_ph):
            Vp = V[e[p]:e[p + 1]]
            if p % 2 == 0:
                d = 0.0
                if model is not None:
                    d = float(model.predict(pd.DataFrame([feats(p, cur, hrs, c_in)])[FEAT])[0])
                thr = VMAX - d
                hit = np.where(Vp >= thr)[0]
                if len(hit) and hit[0] >= 1:
                    j = int(hit[0])
                    tc = (j - 1 + (thr - Vp[j - 1]) / (Vp[j] - Vp[j - 1])) * DT
                    k = j + 1
                else:
                    k = int(len(Vp) * 1.6) + 2
                    bad = len(Vp) >= O.MAX_STEPS
                    tc = k * DT
                k = int(np.clip(k, 2, O.MAX_STEPS))
                new.append(k)
                h_new.append(tc)
            else:
                fr = dis_frac[p // 2] if dis_frac else O.DIS_FRAC
                td = fr * cur[p - 1] * h_new[-1] / cur[p]
                new.append(int(np.clip(int(np.ceil(td / DT)) + 1, 2, O.MAX_STEPS)))
                h_new.append(td)
        hrs = h_new
        # 시간 상한 (환경변수 ROLLOUT_TMAX, 단위 h). 제약 최적화에서 어차피 탈락할 긴 운전을 일찍 버려
        # 평가 비용을 줄인다 (260920). 설정하지 않으면 동작이 예전과 같다.
        _tmax = float(os.environ.get('ROLLOUT_TMAX', '0') or 0)
        if _tmax and sum(hrs) > _tmax:
            return None
        if new == steps or bad:
            break
        steps = new
    if bad:
        return None
    V, cm = predict(cur, steps, c_in)
    e = np.r_[0, np.cumsum(steps)]
    E = []
    cend = []
    for p in range(n_ph):
        Vp, cp = V[e[p]:e[p + 1]], cm[e[p]:e[p + 1]]
        tt = np.arange(len(Vp)) * DT
        h = hrs[p]
        if h > tt[-1]:
            return None
        tg = np.r_[tt[tt < h], h]
        E.append((1 if p % 2 == 0 else -1) * cur[p] * np.trapezoid(np.interp(tg, tt, Vp), tg))
        if p % 2 == 0:
            cend.append(float(np.interp(h, tt, cp)))
    if cmodel is not None:
        f = feats(4, cur, hrs, c_in)
        f['h_ch'] = hrs[4]
        cend[-1] -= float(cmodel.predict(pd.DataFrame([f])[FEAT_C])[0])
    return dict(hours=sum(hrs), hrs=hrs, removed=c_in - cend[-1], cend=cend, E_net=sum(E), E=E)


def evaluate(runs, pick_model, tag):
    rows = []
    t0 = time.time()
    for n, r in enumerate(runs):
        mdl, cmd = pick_model(r)
        for lab, kw in (('raw', {}), ('t', dict(model=mdl)), ('tc', dict(model=mdl, cmodel=cmd))):
            q = rollout(r['cur'], r['c_in'], **kw)
            d = dict(run=r['run'], protocol=r['protocol'], c_in=r['c_in'], const=r['const'], model=lab,
                     ok=q is not None, hours_c=r['hrs'].sum(), removed_c=r['removed'], E_net_c=r['E_net'])
            if q is not None:
                d.update(hours=q['hours'], removed=q['removed'], E_net=q['E_net'])
                for i, p in enumerate((0, 2, 4)):
                    d['tch%d_err' % (i + 1)] = 100 * (q['hrs'][p] / r['hrs'][p] - 1)
                    d['cend%d_err' % (i + 1)] = q['cend'][i] - r['cend'][i]
            rows.append(d)
        if n % 50 == 0:
            print('  %s %d/%d  (%.0fs)' % (tag, n, len(runs), time.time() - t0), flush=True)
    R = pd.DataFrame(rows)
    for m in ('hours', 'removed', 'E_net'):
        R[m + '_err'] = 100 * (R[m] / R[m + '_c'] - 1)
    return R


def summarize(R):
    cols = ['tch1_err', 'tch2_err', 'tch3_err', 'cend3_err', 'hours_err', 'removed_err', 'E_net_err']
    ok = R[R.ok]
    print('실패: %s' % R[~R.ok].groupby('model').size().to_dict())
    print('\n평균 오차 (시간·에너지 %, cend3 = 3사이클 끝 농도 mM, 양수 = 대리모델이 큼)')
    print(ok.groupby(['model', 'const'])[cols].mean().round(2).to_string())
    print('\nRMSE')
    print(ok.groupby(['model', 'const'])[cols].agg(lambda s: np.sqrt((s ** 2).mean())).round(2).to_string())
    gain_errors(ok)
    g = ok.groupby(['model', 'const'])[['hours_err', 'removed_err', 'E_net_err']].mean().unstack('const')
    print('\n가변 - 정전류 평균 오차 차이 (%p, 0 에 가까울수록 가짜 이득 없음)')
    for m in g.index:
        if (('hours_err', True) not in g.columns) or (('hours_err', False) not in g.columns):
            break
        print('  %-5s ' % m + '  '.join('%s %+.2f' % (k, g.loc[m, (k, False)] - g.loc[m, (k, True)])
                                        for k in ('hours_err', 'removed_err', 'E_net_err')))


def gain_errors(ok, curve=('f10base', 'f2scc', 'a3cc')):
    """같은 탈염량 정전류(방전=충전) 대비 % 의 오차. 대리모델은 자기 정전류 예측 곡선, COMSOL 은 COMSOL 곡선 (농도별)."""
    out = []
    for (m, c), z in ok.groupby(['model', 'c_in']):
        cc = z[z.const & z.protocol.str.startswith(curve) & ~z.protocol.str.contains('_d10')]
        if len(cc) < 3:
            continue
        for _, r in z[~z.const].iterrows():
            g = {}
            for src, h, R, E in (('p', 'hours', 'removed', 'E_net'), ('c', 'hours_c', 'removed_c', 'E_net_c')):
                s = cc.sort_values(R)
                if not s[R].min() <= r[R] <= s[R].max():
                    break
                g['dt_' + src] = 100 * (r[h] / np.interp(r[R], s[R], s[h]) - 1)
                g['de_' + src] = 100 * (r[E] / np.interp(r[R], s[R], s[E]) - 1)
            if len(g) == 4:
                out.append(dict(model=m, c_in=c, protocol=r.protocol, **g))
    G = pd.DataFrame(out)
    if G.empty:
        return G
    G['dt_err'] = G.dt_p - G.dt_c
    G['de_err'] = G.de_p - G.de_c
    G['dt_sign'] = np.sign(G.dt_p) == np.sign(G.dt_c)
    G['de_sign'] = np.sign(G.de_p) == np.sign(G.de_c)
    print('\n같은 탈염량 정전류 대비 %% 의 오차 (가변 전류, 대리모델 - COMSOL, %%p; 음수 = 가짜 이득)')
    print(G.groupby('model').agg(n=('dt_err', 'size'), dt_mean=('dt_err', 'mean'), dt_sd=('dt_err', 'std'),
                                 de_mean=('de_err', 'mean'), de_sd=('de_err', 'std'),
                                 dt_sign=('dt_sign', 'mean'), de_sign=('de_sign', 'mean')).round(2).to_string())
    return G


def evaluate_cv():
    with open(os.path.join(OUT, 'termcorr_model%s.pkl' % SUF), 'rb') as f:
        M = pickle.load(f)
    runs = [r for r in comsol_runs() if r['run'] in M['fold_of']]
    R = evaluate(runs, lambda r: (M['models'][M['fold_of'][r['run']]], M['cmodels'][M['fold_of'][r['run']]]), 'cv')
    R.to_csv(os.path.join(OUT, 'termcorr_runs%s.csv' % SUF), index=False, encoding='utf-8-sig')
    summarize(R)


def evaluate_d600():
    with open(os.path.join(OUT, 'termcorr_model%s.pkl' % SUF), 'rb') as f:
        M = pickle.load(f)
    # d600 19조건 + 같은 농도 정전류 곡선 (방전=충전; 보정 학습에 쓰인 운전이라 기준선 역할만 한다)
    runs = comsol_runs(only='d600') + [r for r in comsol_runs() if np.isclose(r['c_in'], 600.0) and r['const']
                                       and r['protocol'].startswith(('f10base', 'f2scc', 'a3cc'))
                                       and '_d10' not in r['protocol']]
    R = evaluate(runs, lambda r: (M['models']['all'], M['cmodels']['all']), 'd600')
    R.to_csv(os.path.join(OUT, 'termcorr_d600%s.csv' % SUF), index=False, encoding='utf-8-sig')
    summarize(R)


if 'build' in STEPS:
    build()
if 'fit' in STEPS:
    fit()
if 'eval' in STEPS:
    evaluate_cv()
if 'd600' in STEPS:
    evaluate_d600()
