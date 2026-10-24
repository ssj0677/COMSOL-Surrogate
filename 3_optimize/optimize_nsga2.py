# -*- coding: utf-8 -*-
"""4단계 — NSGA-II 로 전류 계획 최적화.

무엇을 정하는가
    충전1 · 방전1 · 충전2 · 방전2 · 충전3 · 방전3 의 전류 6개  (0.5 ~ 1.5 mA)

무엇을 최소화/최대화 하는가
    목표1  총 소요시간            최소화
    목표2  탈염량 (c_in − 최종 c_mix)   최대화   (= −탈염량 최소화)

어떻게 평가하는가
    COMSOL 대신 대리모델(TFT)을 쓴다. 다만 각 위상이 언제 끝나는지는 미리 알 수 없다
    (충전은 4.2 V 도달, 방전은 그 사이클 충전전하의 98%). 그래서 '롤아웃'을 한다:

        1) 각 위상 길이를 넉넉히 잡아 전체 궤적을 한 번에 예측
        2) 충전 구간에서 4.2 V 를 처음 넘는 지점을 찾아 그 위상의 길이를 확정
        3) 그 충전의 전하로 다음 방전의 길이를 확정
        4) 확정된 길이로 다시 예측 — 몇 번 반복하면 수렴한다

    인과성(뒤 전류가 앞 구간을 바꾸지 못함) 덕분에 반복이 빨리 수렴한다.

기준선
    전류 6개를 모두 같게 준 경우(= 정전류)를 0.5~1.5 mA 로 훑어 파레토 평면에 곡선으로 찍는다.
    최적화 결과가 그 곡선을 얼마나 밀어내는지가 이 연구의 결론이다.

실행:
    python optimize_nsga2.py
"""
import os
import sys

# 콘솔이 cp949 라 em-dash 같은 문자에서 print 가 죽는다. 최적화를 몇 시간 돌린 뒤
# 마지막 출력에서 터지면 결과를 통째로 잃으므로 시작할 때 utf-8 로 바꿔 둔다.
# (실제로 _check_sync 경고문의 '—' 에서 죽어 Fig.9 재생성이 통째로 날아갔다)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_BASE = _ROOT
sys.path.insert(0, _os.path.join(_ROOT, '1_extract'))      # COMSOL 코어(_D_swb_core)
sys.path.insert(0, os.path.join(_BASE, '2_surrogate'))
sys.path.insert(0, os.path.join(_BASE, '3_optimize'))
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import surrogate as S

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ===== 설정 =========================================================================
# 대리모델 학습에 쓸 데이터. surrogate.USE 를 그대로 따라간다 — 여기에 목록을
# 따로 적어 두었더니 ratio 를 추가했는데도 이 스크립트만 옛 목록으로 학습해서
# '재학습했는데 아무것도 안 바뀌는' 일이 있었다.
TRAIN_SETS = list(S.USE)
MODEL_NAME = 'TFT'                # 가장 정확했던 모델
EPOCHS = 150

# 이 농도들에 대해 각각 최적화한다.
# 농도 하나당 14초밖에 안 걸리므로(대리모델 롤아웃이라 COMSOL 이 필요 없다)
# 촘촘하게 훑어 '농도에 따라 최적 전류가 어떻게 변하는가'를 곡선으로 볼 수 있다.
# 학습 데이터의 c_in 범위가 500~700 이므로 그 밖으로 나가면 외삽이 된다.
C_IN_LEVELS = list(np.arange(500.0, 700.1, 12.5))     # 17개
I_BOUNDS = (0.5, 1.5)                 # 전류 범위 [mA]
# ★ 하한 0.5 는 임의 값이 아니라 학습 데이터의 하한이다. COMSOL 추출을 0.5~1.5 mA
#   로 뽑았으므로 그보다 낮은 전류는 대리모델이 한 번도 본 적이 없다. 하한을
#   내리면 그 구간은 외삽이고, NSGA-II 는 외삽 오차를 그대로 이용해 버린다.
#   실험 삼아 내려 보려면:  py optimize_nsga2.py imin=0.3
#   결과는 기존 파일을 덮지 않고 *_imin03 접미사로 따로 저장한다.
_SUF = ''
# rmax = 방전/충전 전류비 상한. 대리모델이 신뢰할 수 있는 영역 안에서만 찾게 한다.
# 기본값 = 학습 데이터의 방전/충전 비 상한. 탐색을 학습 범위 안에 묶어 외삽을
# 막는다. 격자를 60분으로 바꾸며 조건이 263 -> 355 로 늘어 상한도 2.41 -> 2.86
# 이 됐다 (855개 사이클, 중앙 1.13, 99% 2.244).
#   ※ 2.86 은 최댓값이라 그 근처 학습 표본은 몇 개뿐이다. 최적해가 이 경계에
#     몰리면 사실상 외삽이라는 뜻이므로 실행 후 반드시 확인할 것
#     (옛 판에서는 최대 2.34 로 제약에 물리지 않았다).
#   py optimize_nsga2.py rmax=2.24   ->  *_rmax224 로 따로 저장 (99% 로 조이기)
# 명령줄에서 rmax=... 로 덮어쓸 수 있고, rmax=0 이면 제한을 끈다.
RMAX = 2.86
for _a in sys.argv[1:]:
    if _a.startswith('rmax='):
        RMAX = float(_a.split('=')[1]) or None
        _SUF += '_rmax%03d' % round(RMAX * 100)
        print(f'[!] 방전/충전 비를 {RMAX} 이하로 제한한다 (대리모델 신뢰 영역).')
for _a in sys.argv[1:]:
    if _a.startswith('imin='):
        _lo = float(_a.split('=')[1])
        I_BOUNDS = (_lo, I_BOUNDS[1])
        _SUF = '_imin%02d' % round(_lo * 10)
        print(f'[!] 전류 하한 {_lo} mA — 0.5 미만은 학습 범위 밖(외삽)이다.')
N_CYCLES = 3
V_MAX = 4.2
DIS_FRAC = 0.98

POP = 60                # 개체 수
GEN = 40                # 세대 수  (평가 횟수 = POP x GEN)
SEED = 0

# 위상당 최대 길이는 '스텝 수'가 아니라 '시간'으로 정의한다.
# 스텝으로 박아 두면 격자를 바꿀 때마다 최대 운전시간이 따라 바뀐다
# (60분 격자에서 90스텝이면 180 h 가 아니라 90 h 가 된다).
MAX_PHASE_H = 180.0     # 위상당 최대 운전시간 [h]
ROLLOUT_ITER = 7        # 위상 길이 수렴 반복 횟수

OUT_DIR = _os.path.join(_ROOT, '_O_result')
# 파일명에 농도타깃을 박아 둔다. c_res 로 학습한 가중치를 c_mix 로 읽으면
# 예외 없이 조용히 틀린 결과가 나오기 때문에, 캐시를 섞지 못하게 한다.
CACHE = os.path.join(OUT_DIR, f'surrogate_{MODEL_NAME}_{S.CONC_TARGET}.pt')
# ===================================================================================

DT_H = S.DT_MIN / 60.0
MAX_STEPS = int(round(MAX_PHASE_H / DT_H))   # 60분 격자 -> 180
MOL_PER_MAH = 3.7311e-5
V_TOT_M3 = (S.V_CHAMBER_ML + S.V_TANK_ML) * 1e-6
C_PER_MAH = MOL_PER_MAH / V_TOT_M3          # 1 mAh 당 혼합농도 감소 [mol/m^3]


# ─────────────────────────────────────────────────── 대리모델 준비
SURR_SEED = 0        # 대리모델 학습 seed. 재학습 편차를 볼 때만 바꾼다.


def get_surrogate(seed=None, cache=True):
    """학습(또는 캐시 로드) 후 (net, scaler, V0_model) 반환."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print('[데이터]')
    pool = S.load(TRAIN_SETS)
    train, val, test = S.split(pool)
    scaler = S.Scaler().fit(train)

    # 초기 전압 V0(c_in, I1) — 롤아웃의 시작 상태로 필요하다. 데이터에서 회귀로 얻는다.
    rows = []
    for r in pool.values():
        g = r[r['phase'] == 'charge']
        g = g[g['cycle'] == g['cycle'].min()]
        if len(g):
            rows.append((float(r['c_in'].iloc[0]), float(abs(g['I'].iloc[0])),
                         float(g['Vcell'].iloc[0])))
    a = np.array(rows)
    A = np.c_[np.ones(len(a)), a[:, 0], a[:, 1]]
    coef, *_ = np.linalg.lstsq(A, a[:, 2], rcond=None)
    pred = A @ coef
    print(f'  V0 회귀: V0 = {coef[0]:.4f} + {coef[1]:.6f}*c_in + {coef[2]:.4f}*I   '
          f'RMSE {np.sqrt(np.mean((pred-a[:,2])**2))*1000:.1f} mV')

    m = S.build(MODEL_NAME, epochs=EPOCHS,
                seed=SURR_SEED if seed is None else seed)
    if cache and os.path.exists(CACHE):
        print(f'[대리모델] 캐시 로드 {os.path.basename(CACHE)}')
        # ★ 구조 인자 필수. S.build 가 TUNED 를 병합하므로 m.kwargs 가 저장된
        #   가중치와 같은 구조를 만든다. 인자 없이 만들면 TFT 는 heads 만 달라
        #   모양이 같으므로 오류 없이 '다른 모델' 이 되어 최적화 전체가 어긋난다.
        m.net = S.TORCH_MODELS[MODEL_NAME](**m.kwargs).to(S.DEVICE)
        m.net.load_state_dict(torch.load(CACHE, map_location=S.DEVICE))
        m.net.eval()
    else:
        print(f'[대리모델] {MODEL_NAME} 학습 ({len(train)}조건, {EPOCHS}에폭)...')
        t0 = time.time()
        m.fit(train, val, scaler)
        if cache:
            torch.save(m.net.state_dict(), CACHE)
            print(f'  {time.time()-t0:.0f}초, 저장 {CACHE}')
        else:
            print(f'  {time.time()-t0:.0f}초 (캐시 저장 안 함 — seed 실험)')

    r = S.evaluate(m, test, scaler)
    print(f'  검증: V RMSE {r["Vcell_RMSE"]*1000:.1f} mV, R2(seg) {r["Vcell_R2_seg"]:.4f}, '
          f'4.2V 도달시각 오차 {r["ceil_h"]:.2f} h')
    return m, scaler, coef


# ─────────────────────────────────────────────────── 롤아웃
def rollout(net, scaler, v0coef, currents, c_in, full=False):
    """전류 6개 -> (총시간[h], 탈염량[mol/m^3], 성공여부).

    currents = [I_chg1, I_dis1, I_chg2, I_dis2, I_chg3, I_dis3]  (전부 양수, mA)
    """
    n_ph = 2 * N_CYCLES
    steps = [MAX_STEPS] * n_ph                     # 위상별 스텝 수(초기 추정)
    V0 = float(v0coef[0] + v0coef[1] * c_in + v0coef[2] * currents[0])

    ok = True
    for _ in range(ROLLOUT_ITER):
        # --- 외생 시퀀스 조립 ---
        I_seq, ph_idx = [], []
        for p in range(n_ph):
            sign = 1.0 if p % 2 == 0 else -1.0     # 짝수=충전(+), 홀수=방전(-)
            I_seq += [sign * currents[p]] * steps[p]
            ph_idx += [p] * steps[p]
        I_seq = np.asarray(I_seq, float)
        ph_idx = np.asarray(ph_idx)
        T = len(I_seq)
        if T < 4:
            return np.nan, np.nan, False

        Q_cum = np.r_[0.0, np.cumsum(I_seq[:-1])] * DT_H          # mAh
        c_far = c_in - C_PER_MAH * Q_cum

        # --- 대리모델 예측 ---
        # 초기상태(t=0)는 [Vcell, 농도타깃] 두 개다. 농도타깃이
        #   c_mix 면 t=0 에 c_in,  c_res 면 c_mix(0)-c_far(0) = 0.
        raw = np.c_[np.zeros(T), np.zeros(T), I_seq, Q_cum, np.full(T, c_in)]
        raw[0, 0] = V0                                            # 초기 전압
        raw[0, 1] = c_in if S.CONC_TARGET == 'c_mix' else 0.0      # 초기 농도
        x = (raw - scaler.mean) / scaler.std
        with torch.no_grad():
            init = torch.as_tensor(x[:1, :S.N_OUT], dtype=torch.float32,
                                   device=S.DEVICE)
            exo = torch.as_tensor(x[None, :, S.N_OUT:], dtype=torch.float32,
                                  device=S.DEVICE)
            y = net(init, exo)[0].cpu().numpy()
        y = scaler.back(y)
        V = y[:, 0]
        c_mix = y[:, 1] + (c_far if S.CONC_TARGET == 'c_res' else 0.0)

        # --- 위상 길이 재추정 ---
        new_steps, q_chg = [], []
        bad = False
        for p in range(n_ph):
            idx = np.where(ph_idx == p)[0]
            if len(idx) < 2:
                bad = True
                new_steps.append(steps[p])
                continue
            if p % 2 == 0:                                        # 충전 : 4.2 V 까지
                hit = np.where(V[idx] >= V_MAX)[0]
                if len(hit):
                    k = int(hit[0]) + 1
                else:
                    # 못 닿았으면 실패가 아니라 '더 길게 잡아 다시' 시도한다.
                    # (앞 위상 길이가 바뀌면 이 위상에 들어오는 상태도 바뀌므로
                    #  한 번 짧게 잡혔다고 해서 도달 불가인 것이 아니다)
                    k = int(len(idx) * 1.6) + 2
                    if len(idx) >= MAX_STEPS:
                        bad = True                                # 상한에서도 못 닿음
                k = int(np.clip(k, 2, MAX_STEPS))
                new_steps.append(k)
                q_chg.append(currents[p] * k * DT_H)
            else:                                                 # 방전 : 충전전하의 98%
                q = q_chg[-1] if q_chg else 0.0
                k = int(round(DIS_FRAC * q / max(currents[p], 1e-6) / DT_H))
                new_steps.append(int(np.clip(k, 2, MAX_STEPS)))
        ok = not bad
        if new_steps == steps:
            break
        steps = new_steps

    total_h = sum(steps) * DT_H
    # 탈염량 = 마지막 '충전'이 끝난 시점의 혼합농도 (방전은 새 해수에서 진행되므로
    # 그 이후 값은 탈염 성능이 아니다).
    # ※ 누적합은 '다음 위상의 첫 인덱스'이므로 1을 빼야 마지막 충전의 끝이 된다.
    end_last_charge = int(np.sum(steps[:2 * N_CYCLES - 1])) - 1
    end_last_charge = int(np.clip(end_last_charge, 0, len(c_mix) - 1))
    removed = c_in - float(c_mix[end_last_charge])
    if full:
        # 검증용: 확정된 위상 길이까지의 궤적을 함께 돌려준다
        n = int(sum(steps))
        t_h = np.arange(n) * DT_H
        return dict(hours=total_h, removed=removed, ok=ok, steps=list(steps),
                    t=t_h, V=V[:n], c_mix=c_mix[:n], I=I_seq[:n])
    return total_h, removed, ok


# ─────────────────────────────────────────────────── 최적화 문제
def clip_ratio(x, rmax):
    """방전 전류를 충전 전류의 rmax 배 이하로 자른다 (사이클별).

    왜 필요한가
        대리모델은 방전/충전 전류비가 1 근처인 데이터만 봤다(중앙 1.00, 95% 1.27,
        최대 1.75). 최적화는 비 2.0 근처로 나가 버렸고, 그 영역에서 탈염량을
        +28~35% 과대예측한 것이 COMSOL 검증에서 드러났다.
        rmax 를 걸면 신뢰할 수 있는 영역 안에서만 해를 찾는다.
    """
    if not rmax:
        return x
    y = np.array(x, float, copy=True)
    y[1::2] = np.minimum(y[1::2], rmax * y[0::2])
    return np.maximum(y, I_BOUNDS[0])


class CurrentPlan(Problem):
    def __init__(self, net, scaler, v0coef, c_in, rmax=None):
        super().__init__(n_var=2 * N_CYCLES, n_obj=2, n_constr=0,
                         xl=np.full(2 * N_CYCLES, I_BOUNDS[0]),
                         xu=np.full(2 * N_CYCLES, I_BOUNDS[1]))
        self.net, self.scaler, self.v0, self.c_in = net, scaler, v0coef, c_in
        self.rmax = rmax

    def _evaluate(self, X, out, *args, **kwargs):
        f = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            x = clip_ratio(x, self.rmax)
            t, rem, ok = rollout(self.net, self.scaler, self.v0, x, self.c_in)
            if not np.isfinite(t) or not np.isfinite(rem) or not ok:
                f[i] = [1e4, 1e4]                 # 실패는 크게 벌점
            else:
                f[i] = [t, -rem]                  # 시간 최소화 / 탈염 최대화
        out['F'] = f


def baseline_curve(net, scaler, v0coef, c_in, n=21):
    """정전류(6개 모두 동일) 기준선."""
    rows = []
    for I in np.linspace(*I_BOUNDS, n):
        t, rem, ok = rollout(net, scaler, v0coef, np.full(2 * N_CYCLES, I), c_in)
        if ok and np.isfinite(t):
            rows.append((I, t, rem))
    return pd.DataFrame(rows, columns=['I', 'hours', 'removed'])


# ─────────────────────────────────────────────────── 실행
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m, scaler, v0coef = get_surrogate()
    net = m.net

    all_front, all_base = [], []
    for c_in in C_IN_LEVELS:
        print(f'\n{"="*70}\nc_in = {c_in:.0f} mol/m^3')
        t0 = time.time()

        base = baseline_curve(net, scaler, v0coef, c_in)
        base['c_in'] = c_in
        all_base.append(base)
        print(f'  기준선(정전류) {len(base)}점  '
              f'시간 {base.hours.min():.0f}~{base.hours.max():.0f} h  '
              f'탈염 {base.removed.min():.0f}~{base.removed.max():.0f} mol/m^3')

        prob = CurrentPlan(net, scaler, v0coef, c_in, rmax=RMAX)
        algo = NSGA2(pop_size=POP, sampling=LHS(),
                     crossover=SBX(prob=0.9, eta=15),
                     mutation=PM(eta=20), eliminate_duplicates=True)
        res = minimize(prob, algo, get_termination('n_gen', GEN),
                       seed=SEED, verbose=False)

        F, X = res.F, res.X
        good = F[:, 0] < 1e3
        F, X = F[good], X[good]
        # 평가에서 자른 것과 같은 값이 저장되도록 여기서도 자른다
        X = np.array([clip_ratio(x, RMAX) for x in X]) if RMAX else X
        df = pd.DataFrame(X, columns=['I_chg1', 'I_dis1', 'I_chg2',
                                      'I_dis2', 'I_chg3', 'I_dis3'])
        df['hours'] = F[:, 0]
        df['removed'] = -F[:, 1]
        df['c_in'] = c_in
        all_front.append(df)
        print(f'  파레토 {len(df)}점  시간 {df.hours.min():.0f}~{df.hours.max():.0f} h  '
              f'탈염 {df.removed.min():.0f}~{df.removed.max():.0f}   ({time.time()-t0:.0f}s)')

        # 같은 시간에서 얼마나 더 뺐는가
        gain = []
        for _, r in df.iterrows():
            nb = base.iloc[(base.hours - r.hours).abs().argsort()[:1]]
            if len(nb):
                gain.append(r.removed - float(nb.removed.iloc[0]))
        if gain:
            print(f'  같은 시간 대비 탈염 개선: 중앙 {np.median(gain):+.1f}, '
                  f'최대 {np.max(gain):+.1f} mol/m^3')

    front = pd.concat(all_front, ignore_index=True)
    base = pd.concat(all_base, ignore_index=True)
    front.to_csv(os.path.join(OUT_DIR, f'pareto_front{_SUF}.csv'),
                 index=False, encoding='utf-8-sig')
    base.to_csv(os.path.join(OUT_DIR, f'baseline_constant{_SUF}.csv'),
                index=False, encoding='utf-8-sig')

    # ---- 그림 ----
    # 농도를 촘촘히 훑으면 패널이 17개가 되어 못 쓴다. 그림은 별도 스크립트에
    # 맡긴다 — 거기서 색으로 농도를 표현하고, 최적화 재실행 없이 다시 그릴 수 있다.
    # 하한을 바꾼 실험 실행은 기존 그림을 덮으면 안 되므로 건너뛴다
    # (plot_pareto 는 기본 csv 를 읽어 기본 png 를 쓴다).
    if not _SUF:
        import plot_pareto
        plot_pareto.main()
    print('saved:', os.path.join(OUT_DIR, f'pareto_front{_SUF}.csv'))

    # ---- 대표 해 ----  (농도가 많으므로 양끝과 가운데만 찍는다)
    print('\n[대표 해] c_in 별로 빠른쪽/중간/많은쪽')
    cols = ['I_chg1', 'I_dis1', 'I_chg2', 'I_dis2', 'I_chg3', 'I_dis3']
    show = sorted({C_IN_LEVELS[0], C_IN_LEVELS[len(C_IN_LEVELS) // 2],
                   C_IN_LEVELS[-1]})
    for c_in in show:
        f = front[front.c_in == c_in].sort_values('hours').reset_index(drop=True)
        if not len(f):
            continue
        print(f'\n  c_in = {c_in:.0f}')
        for tag, i in (('빠름', 0), ('중간', len(f) // 2), ('많이', len(f) - 1)):
            r = f.iloc[i]
            plan = ' '.join(f'{r[c]:.2f}' for c in cols)
            print(f'    {tag}  {r.hours:6.0f} h  {r.removed:6.1f} mol/m3   [{plan}]')


if __name__ == '__main__':
    main()
