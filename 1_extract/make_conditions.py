# -*- coding: utf-8 -*-
"""데이터 추출용 조건표 생성 — 네 프로토콜 한 번에.

프로토콜
    cap       고정용량 사이클   충전 13.23 mAh(정격/3) → 방전 98% → 반복
    vlim      전압한계 사이클   충전 4.2 V 도달 → 방전 98% → 반복
    vstart    초기상태 가변     vlim 과 같되 담수부 초기농도·음극 초기재고를 조건마다 다르게
    multistep 다단전류(방전없음) 4.2 V 도달마다 전류를 낮춰 계속 충전

왜 vstart 가 필요한가
    서로게이트의 입력이 '초기상태 [Vcell, cNa_C2] + 전류계획' 이다. 학습 데이터가 전부
    같은 초기상태(600 mol/m^3, 3.33 V)에서 출발하면 초기상태 축의 정보량이 0이고,
    RL/NSGA-II 가 중간 상태에서 질문하는 순간 전부 외삽이 된다.
    담수부 초기농도를 낮추면 초기 전압이 높아지고 cNa_C2 도 낮아져, 그 축이 실제로 채워진다.

실행:
    python make_conditions.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT_DIR = _ROOT
N_COND = 200
SEED = 0

WATER = 'seawater'                 # 'seawater' | 'RObrine'
C_IN_RANGE = {'seawater': (500.0, 700.0), 'RObrine': (1000.0, 1400.0)}[WATER]
I_RANGE = (0.50, 1.50)             # 정전류 [mA]

# --- vstart 전용 ---------------------------------------------------------------
# 담수부 초기농도를 유입수 농도의 이 비율 범위로 흔든다. 1.0 = 신선(기본과 같음).
C2_FRAC_RANGE = (0.30, 1.00)
# 음극 초기 재고 [mol/m^2]. 0 = 완전 방전 상태. 정격 39.7 mAh ≈ 0.769 mol/m^2 (nNa 회귀)
NNA0_RANGE = (0.0, 0.30)

# --- multistep 전용 ------------------------------------------------------------
I1_RANGE = (0.60, 1.50)            # 1단 전류
R_RANGE = (0.55, 0.85)             # 다음 단 = 이전 단 × r
# 하한을 0.20 -> 0.40 으로 올렸다. 0.2~0.3 mA 단은 종료 전압에 닿지 못한 채
# 시간 상한(60 h)만 잡아먹는다 — 실측으로 정전류 0.6 mA 미만은 도달률 0% 였다.
# 정전류 데이터 범위(0.50~1.50 mA) 와도 맞춘다.
I_MIN = 0.40                       # 이보다 낮은 단은 만들지 않는다
V_SWITCH_MAX = 4.15                # 마지막 단은 항상 4.15 V 에서 종료 (수치 안정)
# 전환전압을 조절하는 조건의 비율(나머지는 전부 4.2 V 에서 전환)
FRAC_TUNED_VSW = 2 / 3


def _lhs(n, d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def _tag(n, seed=SEED):
    """train/val/test 를 조건 단위로 미리 박아둔다 — 추출 후에 나누면 실수하기 쉽다."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tag = np.empty(n, dtype=object)
    n_tr, n_va = int(round(n * 0.7)), int(round(n * 0.15))
    tag[idx[:n_tr]] = 'train'
    tag[idx[n_tr:n_tr + n_va]] = 'val'
    tag[idx[n_tr + n_va:]] = 'test'
    return tag


def make_cycle(name, n=N_COND, seed=SEED, vstart=False):
    """cap / vlim / vstart 용 조건표. 정전류 하나로 충방전을 반복한다."""
    d = 4 if vstart else 2
    u = _lhs(n, d, seed)
    df = pd.DataFrame({
        'cond_id': np.arange(1, n + 1),
        'c_in[mol/m3]': np.round(C_IN_RANGE[0] + u[:, 0] * np.ptp(C_IN_RANGE)),
        'I[mA]': np.round(I_RANGE[0] + u[:, 1] * np.ptp(I_RANGE), 3),
    })
    if vstart:
        frac = C2_FRAC_RANGE[0] + u[:, 2] * np.ptp(C2_FRAC_RANGE)
        df['c_C2_0[mol/m3]'] = np.round(df['c_in[mol/m3]'] * frac)
        df['nNa0[mol/m2]'] = np.round(NNA0_RANGE[0] + u[:, 3] * np.ptp(NNA0_RANGE), 4)
    else:
        df['c_C2_0[mol/m3]'] = df['c_in[mol/m3]']      # 담수부도 신선한 해수로 시작
        df['nNa0[mol/m2]'] = 0.0
    df['set'] = _tag(n, seed)

    path = os.path.join(OUT_DIR, f'cond_{name}_{WATER}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'[{name}] {len(df)}조건 -> {os.path.basename(path)}')
    print(f'   c_in {df["c_in[mol/m3]"].min():.0f}~{df["c_in[mol/m3]"].max():.0f}   '
          f'I {df["I[mA]"].min():.2f}~{df["I[mA]"].max():.2f}   '
          f'train/val/test {(df["set"]=="train").sum()}/'
          f'{(df["set"]=="val").sum()}/{(df["set"]=="test").sum()}')
    if vstart:
        print(f'   c_C2_0 {df["c_C2_0[mol/m3]"].min():.0f}~{df["c_C2_0[mol/m3]"].max():.0f}   '
              f'nNa0 {df["nNa0[mol/m2]"].min():.3f}~{df["nNa0[mol/m2]"].max():.3f}')
    return df


def make_lhs6(n=N_COND, seed=SEED, n_cycles=3):
    """사이클마다 충전·방전 전류를 따로 뽑는다 — 7차원 LHS.

    변수  c_in, I_chg1, I_dis1, I_chg2, I_dis2, I_chg3, I_dis3
    종료  충전 = 4.2 V,  방전 = 그 사이클 충전전하의 98%

    왜 필요한가
        정전류 데이터만 쓰면 서로게이트가 '전류가 도중에 바뀌는' 상황을 본 적이 없다.
        스텝전류 최적화(NSGA-II)나 RL 은 매 구간 전류를 바꾸므로, 그 축을 학습셋이
        덮고 있어야 한다. 사이클 경계마다 전류가 튀는 것을 배우게 하는 것이 목적이다.
    """
    d = 1 + 2 * n_cycles
    u = _lhs(n, d, seed)
    df = pd.DataFrame({
        'cond_id': np.arange(1, n + 1),
        'c_in[mol/m3]': np.round(C_IN_RANGE[0] + u[:, 0] * np.ptp(C_IN_RANGE)),
    })
    for k in range(1, n_cycles + 1):
        df[f'I_chg{k}'] = np.round(I_RANGE[0] + u[:, 2*k-1] * np.ptp(I_RANGE), 3)
        df[f'I_dis{k}'] = np.round(I_RANGE[0] + u[:, 2*k] * np.ptp(I_RANGE), 3)
    df['I[mA]'] = df['I_chg1']            # 호환용(요약 출력에 쓰임)
    df['set'] = _tag(n, seed)

    path = os.path.join(OUT_DIR, f'cond_cycle_lhs6_{WATER}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    cols = [c for c in df.columns if c.startswith('I_')]
    print(f'[cycle_lhs6] {len(df)}조건 x {n_cycles}사이클 -> {os.path.basename(path)}')
    print(f'   변수 {1+len(cols)}개: c_in + {cols}')
    print(f'   c_in {df["c_in[mol/m3]"].min():.0f}~{df["c_in[mol/m3]"].max():.0f}   '
          f'전류 {df[cols].values.min():.2f}~{df[cols].values.max():.2f} mA')
    return df


def make_multistep(n=N_COND, seed=SEED):
    """다단 전류 조건표. 전류 사다리 + 전환전압.

    한 조건 = I_1 로 충전 → V_1 도달 시 I_2 로 낮춤 → ... → 마지막은 4.2 V 에서 종료.
    전환전압을 4.2 로 고정한 것(문헌형)과 조금 낮춰 미리 전환하는 것을 섞는다.
    """
    rng = np.random.default_rng(seed)
    u = _lhs(n, 3, seed)
    rows = []

    for k in range(n):
        c_in = round(C_IN_RANGE[0] + u[k, 0] * np.ptp(C_IN_RANGE))
        I1 = round(I1_RANGE[0] + u[k, 1] * np.ptp(I1_RANGE), 3)
        r = R_RANGE[0] + u[k, 2] * np.ptp(R_RANGE)

        I = [I1]
        while len(I) < 5 and I[-1] * r >= I_MIN:
            I.append(round(I[-1] * r, 3))

        tuned = (k % 3) != 0                              # 정확히 1:2 (고정:조절)
        if tuned and len(I) > 1:
            # 마지막을 뺀 전환전압을 3.94~4.15 에서 단조증가하게 뽑는다
            lo, hi = 3.94, 4.15
            v = np.sort(rng.uniform(lo, hi, len(I) - 1))
            V = [round(float(x), 4) for x in v]
        else:
            V = [V_SWITCH_MAX] * (len(I) - 1)

        row = {'cond_id': k + 1, 'c_in[mol/m3]': c_in, 'n_seg': len(I),
               'I[mA]': I1, 'vsw_tuned': int(tuned)}
        for j, val in enumerate(I, 1):
            row[f'I_{j}'] = val
        for j, val in enumerate(V, 1):
            row[f'V_{j}'] = val
        row[f'V_{len(I)}'] = V_SWITCH_MAX                 # 마지막 단은 항상 4.2
        rows.append(row)

    df = pd.DataFrame(rows)
    df['set'] = _tag(n, seed)

    # 홀드아웃용 문헌 사다리를 1번 조건으로 박아둔다 (1.0→0.8→0.6→0.4→0.2 @ 4.2V)
    lit = {'cond_id': 1, 'c_in[mol/m3]': 600, 'n_seg': 5, 'I[mA]': 1.0, 'vsw_tuned': 0,
           'set': 'holdout'}
    for j, val in enumerate([1.0, 0.8, 0.6, 0.4, 0.2], 1):
        lit[f'I_{j}'] = val
    for j in range(1, 6):
        lit[f'V_{j}'] = V_SWITCH_MAX
    df.loc[0, list(lit)] = pd.Series(lit)

    path = os.path.join(OUT_DIR, f'cond_multistep_{WATER}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    n_tuned = int(df['vsw_tuned'].sum())
    print(f'[multistep] {len(df)}조건 -> {os.path.basename(path)}')
    print(f'   단수 {df["n_seg"].min()}~{df["n_seg"].max()}   '
          f'전환전압 고정 {len(df)-n_tuned} : 조절 {n_tuned}   '
          f'1번 = 문헌 사다리(holdout)')
    return df


RATIO_RANGE = (1.3, 2.6)           # 방전/충전 전류비 — lhs6 이 못 덮는 영역
RATIO_ICHG_RANGE = (0.50, 0.95)    # 그 비를 만들려면 충전은 낮아야 한다

# ratio2 — 위 설계의 수정판.
#   ratio 판(충전 0.50~0.95)은 44%가 버려졌다. 충전전류가 0.7 mA 미만이면
#   4.2 V 에 닿기 전에 80 mAh 상한(tlist)에 걸려 절단되고, 절단 세그먼트는
#   surrogate.DROP_TRUNCATED 에서 학습 제외되기 때문이다.
#   충전 하한을 0.70 으로 올리고 비 범위를 최적해 근처(1.5~2.1)로 좁혀
#   같은 영역을 덮으면서 절단을 피한다.
RATIO2_RANGE = (1.5, 2.1)
RATIO2_ICHG_RANGE = (0.70, 0.95)


def make_ratio(n=120, seed=7, n_cycles=3, name='ratio',
               rng=None, ichg_rng=None):
    """방전/충전 전류비가 큰 조건 — NSGA-II 최적해가 사는 영역을 학습셋에 넣는다.

    왜 필요한가
        lhs6 은 충전·방전 전류를 같은 범위에서 독립으로 뽑는다. 그러면 세 사이클
        평균의 방전/충전 비가 1 근처로 뭉쳐(중앙 1.00, 95% 1.27, 최대 1.75)
        '충전은 느리게, 방전은 빠르게' 하는 영역이 학습셋에 거의 없다.
        그런데 NSGA-II 최적해는 바로 그 영역(deep 해에서 비 1.70~2.12)에 있다.

        COMSOL 검증 결과 그 영역에서 대리모델이 탈염량을 +43~56 mol/m^3 과대예측했고
        (전압 RMSE 53 mV, 테스트셋 9 mV), 예측 이득 +27% 가 실제로는 +12% 였다.
        최적화가 모델 오차를 이용한 것이다. 이 조건표는 그 구멍을 메운다.

    설계
        충전전류를 낮은 쪽(0.50~0.95)에서 뽑고, 사이클마다 비를 따로 뽑아
        방전 = 충전 x 비 로 만든다. 상한 1.5 mA 는 하드웨어 제약이라 그대로 자른다.
    """
    d = 1 + 2 * n_cycles
    u = _lhs(n, d, seed)
    df = pd.DataFrame({
        'cond_id': np.arange(1, n + 1),
        'c_in[mol/m3]': np.round(C_IN_RANGE[0] + u[:, 0] * np.ptp(C_IN_RANGE)),
    })
    rng = rng or RATIO_RANGE
    ichg_rng = ichg_rng or RATIO_ICHG_RANGE
    for k in range(1, n_cycles + 1):
        ichg = ichg_rng[0] + u[:, 2*k-1] * np.ptp(ichg_rng)
        ratio = rng[0] + u[:, 2*k] * np.ptp(rng)
        df[f'I_chg{k}'] = np.round(ichg, 3)
        df[f'I_dis{k}'] = np.round(np.minimum(ichg * ratio, I_RANGE[1]), 3)
    df['I[mA]'] = df['I_chg1']
    df['set'] = _tag(n, seed)

    path = os.path.join(OUT_DIR, f'cond_cycle_{name}_{WATER}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    ch = df[[f'I_chg{k}' for k in range(1, n_cycles+1)]].mean(axis=1)
    di = df[[f'I_dis{k}' for k in range(1, n_cycles+1)]].mean(axis=1)
    r = di / ch
    print(f'[cycle_{name}] {len(df)}조건 x {n_cycles}사이클 -> {os.path.basename(path)}')
    print(f'   c_in {df["c_in[mol/m3]"].min():.0f}~{df["c_in[mol/m3]"].max():.0f}   '
          f'충전 {ch.min():.2f}~{ch.max():.2f}   방전 {di.min():.2f}~{di.max():.2f} mA')
    print(f'   방전/충전 비  {r.min():.2f}~{r.max():.2f}  (중앙 {r.median():.2f}) '
          f'— lhs6 은 최대 1.75 였다')
    return df


if __name__ == '__main__':
    print(f'수질 {WATER}   c_in {C_IN_RANGE}   I {I_RANGE}\n')
    make_cycle('cycle_cap', vstart=False)
    make_cycle('cycle_vlim', vstart=False)
    make_cycle('cycle_vstart', vstart=True)
    make_lhs6()
    make_ratio()
    make_ratio(n=120, seed=11, name='ratio2',
               rng=RATIO2_RANGE, ichg_rng=RATIO2_ICHG_RANGE)
    make_multistep()
    print('\n조건표 생성 완료.')
