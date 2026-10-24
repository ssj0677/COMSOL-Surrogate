# -*- coding: utf-8 -*-
"""SWB 서로게이트(대리모델) — 데이터 · 모델 · 학습 · 평가 를 한 파일에.

═══════════════════════════════════════════════════════════════════════════
왜 만드는가
═══════════════════════════════════════════════════════════════════════════
COMSOL 로 한 조건(충방전 3사이클)을 푸는 데 4~15분이 걸린다.
전류 계획을 최적화하려면 수천 번을 평가해야 하는데 현실적으로 불가능하다.
그래서 COMSOL 의 입출력 관계만 흉내 내는 가벼운 모델을 학습시킨다.

    COMSOL      한 조건 = 수 분
    대리모델     한 조건 = 0.01초        (약 10만 배)

═══════════════════════════════════════════════════════════════════════════
무엇을 입력받고 무엇을 내놓는가
═══════════════════════════════════════════════════════════════════════════
    입력 ①  초기상태     t=0 의 [Vcell, c_res]            ... 2개
    입력 ②  전류 계획     매 시점의 [I, Q_cum, c_in]       ... 3개 × T스텝
    출력     궤적         매 시점의 [Vcell, c_res]         ... 2개 × T스텝

  I      그 시점에 흘리는 전류 [mA].  충전 +, 방전 −
  Q_cum  지금까지 흘린 누적 전하 [mAh].  '얼마나 진행됐는가'를 담는다
  c_in   유입수(해수) 농도 [mol/m^3].  조건 내내 상수 = 정적 변수

  ※ 왜 궤적을 '한 번에' 내놓는가
     한 스텝씩 예측해 다시 입력으로 넣는 자기회귀 방식은 오차가 누적된다.
     여기서는 전류 시계열을 통째로 주고 궤적을 통째로 받는다. 학습 데이터에는
     COMSOL 이 실제로 흘린 I(t) 가 기록돼 있으므로 학습은 이대로 성립한다
     (열린 고리 학습).

  ※ 그런데 '전류 계획을 다 안다'는 말은 절반만 맞다
     아는 것   : 전류의 크기와 순서            (우리가 정하는 값)
     모르는 것 : 전류가 '언제' 바뀌는지         (충전은 4.2 V 도달 시,
                                              방전은 그 충전전하의 98% 방전 시)
     전환 시점은 해를 풀어야 알 수 있는 폐루프 값이다.

     그래서 최적화에서 쓸 때는 3_optimize/optimize_nsga2.py 의 rollout() 이
     반복으로 닫는다:
         길이를 넉넉히 잡고 예측 -> 4.2 V 교차점에서 위상 길이 확정
         -> 그 전하로 방전 길이 확정 -> 다시 예측 (수렴할 때까지)

     이 '길게 예측해 잘라 쓰기'가 유효한 이유는 모델이 인과적이기 때문이다.
     시점 t 의 출력이 t 이후의 전류에 영향받지 않아야, 뒤쪽을 아직 모르는 채로
     앞부분을 믿고 자를 수 있다. 네 모델 모두 구조적으로 인과적이며
     (LSTM 은 순방향, attention 은 상삼각 마스크, MLP 는 시점 독립),
     evaluate() 의 causality() 가 이를 매번 검사한다. 실측값 0.00e+00.

     남는 오차 : 교차점을 2시간 격자에서 찾으므로 위상당 최대 2시간이 어긋난다.

═══════════════════════════════════════════════════════════════════════════
농도는 '잔차'가 아니라 직접 맞힌다                    (실험으로 정한 설계)
═══════════════════════════════════════════════════════════════════════════
처음에는 Faraday 잔차를 학습시켰다. 소금이 빠지는 양의 대부분은

    c_faraday(t) = c_in − 1.866 × Q_cum(t)        [mol/m^3]
                            ↑ 1 mAh 가 20 mL 에서 낮추는 농도

로 '입력만으로' 계산되니, 그 큰 추세는 빼주고

    c_res(t) = c_mix(t) − c_faraday(t)

만 배우게 하면 학습이 쉬워진다는 발상이다. 그런데 여기선 통하지 않았다.

    타깃 표준편차   c_mix 143.8   vs   c_res 141.1   (거의 같다)

이유는 '리셋을 못 따라가서' 가 아니다 — Q_cum 은 부호가 있어(방전은 음)
c_far 도 톱니 모양을 갖는다. 문제는 **진폭**이다. 한 조건에서 재보면

    c_mix  470 ~ 890  (폭 420)      c_far  465 ~ 602  (폭 137)
    corr(c_mix, c_far) = +0.20      잔차 std 는 원본의 98%

c_far 는 방향만 맞고 크기가 1/3 이다. 계가 닫혀 있지 않기 때문이다 —
유입·유출과 탱크 순환으로 소금이 검사체적을 드나들어서, 전하보존만으로는
c_mix 가 닫히지 않는다. 그래서 빼도 배워야 할 양이 줄지 않는다.
(※ 이는 c_mix 에 전하보존을 '하드 제약'으로 걸면 안 되는 이유이기도 하다.
   닫히는 것은 음극 Na 재고 nNa 쪽이다.) A/B 결과(TFT, test 55조건)도

    c_res  전압 11.5 mV / 농도 RMSE 3.17      c_mix  10.2 mV / 3.54

로 사실상 동등하다. 이득이 없으므로 해석이 단순한 직접 예측을 쓴다.
(잔차 경로는 CONC_TARGET='c_res' 로 되살릴 수 있게 남겨 두었다.)

═══════════════════════════════════════════════════════════════════════════
파일 구성
═══════════════════════════════════════════════════════════════════════════
    1. 설정      어떤 데이터를 읽고 무엇을 입출력으로 볼지
    2. 데이터    cond_*.csv  ->  {조건이름: 궤적 DataFrame}
    3. 스케일러  표준화 (신경망은 입력 크기 차이에 민감하다)
    4. 모델      MLP / LSTM / LSTM-Attention / TFT / XGBoost — 서명이 전부 같다
    5. 학습      Trainer 하나가 torch 모델 넷을 모두 학습한다
    6. 평가      RMSE, R², 인과성, '4.2 V 에 언제 닿는가' 오차

═══════════════════════════════════════════════════════════════════════════
사용법
═══════════════════════════════════════════════════════════════════════════
    import surrogate as S

    train, val, test = S.split(S.load(['lhs6', 'vstart']))   # 데이터
    scaler = S.Scaler().fit(train)                            # 표준화 기준
    model  = S.build('TFT').fit(train, val, scaler)           # 학습
    print(S.evaluate(model, test, scaler))                    # 평가

    pred = scaler.back(model.predict(scaler.forward(run)))    # 예측
    pred = S.to_c_mix(pred, run)          # 잔차 -> 실제 농도로 복원

성능(2026-08-07, lhs6+vstart 365조건 학습)
    전압 RMSE            10.2 mV       (실측 셀 재현성 한계가 ~20 mV)
    4.2 V 도달시각 오차   0.39 h
    혼합농도 RMSE         2.9 mol/m^3   (600->450 변화폭의 0.5%)
    학습에 안 쓴 정전류 124조건에서도 7.2 mV — 외운 것이 아니라 일반화됨
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# ════════════════════════════════════════════════════════════════════════
# 1. 설정
# ════════════════════════════════════════════════════════════════════════
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
BASE = _ROOT

# 어느 판본을 읽을 것인가.
#   'regrid' : 1_extract/regrid.py 로 5분 균일격자에 다시 깐 것  ← 기본
#   'raw'    : 추출기가 낸 원본 (충전 5.9~14.5분, 방전 5분으로 간격이 제각각)
# 어느 쪽을 읽어도 학습 결과는 사실상 같다. 120분 격자로 다시 성기게 만들기 때문에
# 120분 지점의 값이 거의 동일하다(vstart 480구간 비교: 중앙 0.000 mV, 90% 0.24 mV,
# 최대 4.69 mV — 모델 RMSE 11.1 mV 의 2% 수준). 'regrid' 를 기본으로 두는 이유는
# 정확도가 아니라 데이터를 한 가지 형식으로 정리해 두기 위해서다.
SOURCE = 'regrid'

# 서로게이트가 쓰는 데이터는 이 셋이 전부다.
#   _E_seawater_verify  (최적화 검증)는 4_validate/plot_validation.py 가 자기 경로로
#   직접 읽고, _E_seawater_3cyc (구 데이터, revising39/260817)는 새 데이터와 기준이
#   달라 섞으면 안 된다. 둘 다 여기 두면 '쓰는 데이터'로 오해되므로 넣지 않는다.
# mph 판본 꼬리표 — run_extraction.py 의 DATA_TAG 와 같은 값을 쓴다.
#   'v49' : 260823(revising49) 로 재추출한 데이터. DNa 7.0e-10 + Eeq +0.0516.
#   ''    : 260821 로 뽑은 옛 데이터.
# ★ 둘은 물리가 다르다(같은 조건에서 충전시간이 21~29% 차이). 절대 섞지 말 것.
DATA_TAG = 'v49'
_T = f'_{DATA_TAG}' if DATA_TAG else ''

_RAW = {
    'lhs6':      os.path.join(BASE, f'_E_seawater_lhs6{_T}'),    # 사이클마다 전류가 다름
    'vlim':      os.path.join(BASE, f'_E_seawater_vlim{_T}'),    # 정전류, 4.2 V 종료
    'vstart':    os.path.join(BASE, f'_E_seawater_vstart{_T}'),  # 초기상태 가변
    # ratio = 방전/충전 전류비 1.4~2.3. lhs6 은 그 비가 1 근처(최대 1.75)라
    # NSGA-II 최적해가 사는 영역을 못 덮었고, 그 결과 COMSOL 검증에서 탈염량을
    # +43~56 mol/m^3 과대예측했다(예측 이득 +27% vs 실제 +12%). 그 구멍을 메운다.
    'ratio':     os.path.join(BASE, f'_E_seawater_ratio{_T}'),
    # ratio2 = ratio 의 수정판. 충전 0.70~0.95 로 올려 절단 폐기(44%)를 피했다.
    'ratio2':    os.path.join(BASE, f'_E_seawater_ratio2{_T}'),
}
_REGRID_ROOT = os.path.join(BASE, '_E_regrid')
_REGRID_DT = 5


def _regrid_path(raw_dir):
    return os.path.join(_REGRID_ROOT,
                        f'{os.path.basename(raw_dir)}_grid{_REGRID_DT}')


DATASETS = {k: (_regrid_path(v) if SOURCE == 'regrid' and os.path.isdir(_regrid_path(v))
                else v)
            for k, v in _RAW.items()}
USE = ['lhs6', 'vstart', 'ratio', 'ratio2']   # 학습셋. vlim 은 홀드아웃이라 뺀다
                                    # ratio 추출이 끝나기 전이면 자동으로 빠진다
# ── 환경변수로 학습셋과 출력 폴더를 갈아끼운다 (260922)
#    깨끗한 학습셋(ratio/ratio2 제외) 판본을 기존 결과를 덮지 않고 따로 쌓기 위한 장치다.
#    ratio, ratio2 는 방전/충전 평균비 1.57~2.03 인 세트인데 최적해 17개는 전부
#    0.89~1.07 이라 최적해가 살지 않는 영역만 채웠다. "왜 이 데이터가 학습셋에
#    있나" 에 답할 수 없으므로 빼는 판본을 따로 만든다 (사용자 결정 260922).
#        set SWB_USE=lhs6,vstart
#        set SWB_RESULT_DIR=<output directory>
if os.environ.get('SWB_USE'):
    USE = [v.strip() for v in os.environ['SWB_USE'].split(',') if v.strip()]
RESULT_DIR = os.environ.get('SWB_RESULT_DIR') or os.path.join(BASE, '_S_result')


def _check_sync(key, path):
    """재격자본이 원본보다 뒤처져 있으면 크게 경고한다.

    추출이 계속 도는 동안 regrid.py 를 안 돌리면 학습 데이터에서 조건이 조용히
    빠진다. 실제로 vlim 이 하루에 두 번(173->196->200) 어긋났다. 조용히 적게
    학습되는 것이 가장 나쁜 실패라서, 여기서 반드시 눈에 띄게 만든다.
    """
    raw = _RAW.get(key)
    if not raw or os.path.abspath(path) == os.path.abspath(raw) or not os.path.isdir(raw):
        return
    n_raw = len([f for f in os.listdir(raw) if f.startswith('cond_') and f.endswith('.csv')])
    n_now = len([f for f in os.listdir(path) if f.startswith('cond_') and f.endswith('.csv')])
    if n_raw != n_now:
        print(f'  [!!] {key}: 재격자 {n_now}조건 < 원본 {n_raw}조건 — {n_raw-n_now}개 누락.'
              f'\n       py 1_extract/regrid.py {key} {_REGRID_DT}  를 먼저 돌릴 것.')

# 충전이 전압(4.2 V)이 아니라 tlist 상한(80 mAh)으로 끊긴 세그먼트를 버린다.
# 저전류 조건에서 4.2 V 에 못 닿고 잘린 것들인데, 그대로 두면 서로게이트가
# '충전은 80 mAh 에서 끝난다'는 가짜 규칙을 배운다. (전체의 약 5%)
DROP_TRUNCATED = True
# 절단 판정은 '종료전압' 하나로 한다. 전하량 조건은 뺐다.
#   예전 규칙(Q>=79 AND V<4.15)은 tlist 상한(80 mAh)에 걸린 것만 잡았는데,
#   그보다 훨씬 많은 세그먼트가 다른 이유로 조기 종료하고 있었다:
#   담수부 농도가 100~150 mol/m^3 로 떨어지면 전도도가 0 에 가까워져
#   solver 가 4.2 V 에 닿기 전에 멈춘다 (vstart 충전의 절반, 약 300개).
#   그 데이터를 학습에 넣으면 '충전은 3.9~4.1 V 에서도 끝난다'를 배우게 되어
#   4.2 V 도달 시각을 못 맞힌다 — 실제로 소요시간 예측이 +50~130% 틀렸다.
#   종료전압 분포는 정상 4.10~4.20 / 조기 3.86~4.10 으로 갈리므로 4.10 에서 자른다.
#   기준값은 3.95~4.10 을 seed 3개씩 훑어 골랐다 (4_validate/sweep_trunc.py):
#       4.10  학습 141조건  탈염오차 11.8~19.3%  <- 데이터가 61% 날아가 악화
#       4.05  학습 212조건  탈염오차  4.2~ 7.8%
#       3.95  학습 290조건  탈염오차  3.3~ 4.8%  <- 채택
#   3.95 면 옛 필터와 규모가 비슷하면서(414 vs 408조건) 가장 심한 조기종료만 걷어낸다.
TRUNC_V_MAX = 4.08               # 종료전압이 이보다 낮으면 절단으로 본다

# 무엇을 맞힐 것인가
#   'c_mix'  : 혼합 농도를 직접 예측                    ← 채택
#   'c_res'  : Faraday 예측치와의 '잔차'만 예측
#              c_faraday = c_in − MOL_PER_MAH/V_total × Q_cum  은 입력만으로 계산되니
#              큰 추세는 빼주고 잔차만 배우게 하자는 발상. 롤아웃 때 c_far 를 도로 더한다.
#
# 잔차 학습은 '잔차가 원본보다 작을 때'만 이득인데, 여기선 그렇지 않아서 버렸다.
#   타깃 표준편차   c_mix 143.8  vs  c_res 141.1   — 사실상 동일
#   방전이 담수부를 원수로 되돌리므로 c_mix 는 톱니파가 되고, 단조감소하는 Faraday
#   항으로는 그 리셋을 설명할 수 없다. 그래서 빼도 크기가 안 줄어든다.
# A/B (lhs6+vstart, test 55조건, EPOCHS=100):
#   TFT   c_res  V 11.5 mV / c_mix RMSE 3.17      c_mix  V 10.2 mV / 3.54
#   LSTM  c_res  V 11.7 mV / 4.79                 c_mix  V 13.0 mV / 7.23
#   -> 실제 사용 모델(TFT)은 전압이 오히려 낫고 농도 차이는 0.37 mol/m^3(탈염량의 0.2%).
#      이득이 없으므로 해석이 단순한 직접 예측을 쓴다.
CONC_TARGET = 'c_mix'
TARGETS = ['Vcell', CONC_TARGET]
INPUTS  = ['I', 'Q_cum', 'c_in']  # 미리 아는 것(전류 계획 + 조건)
COLUMNS = TARGETS + INPUTS
N_OUT, N_IN = len(TARGETS), len(INPUTS)

# 담수측(챔버 1.8 + 호스 2개 0.285) / 탱크. 합 = 총 용액 20 mL.
# ※ COMSOL 의 c_mix 는 호스를 빼고 (1.8·cNa_C2 + 17.915·c_tank)/19.715 로 계산한다.
#   여기 2.085 는 호스까지 포함한 담수측 실부피이고, 아래 c_mix 공식은 CSV 에
#   c_mix 열이 없을 때만 쓰이는 대비책이라 실제로는 실행되지 않는다.
V_CHAMBER_ML, V_TANK_ML = 2.085, 17.915
V_TOTAL_M3 = (V_CHAMBER_ML + V_TANK_ML) * 1e-6
MOL_PER_MAH = 3.7311e-5                   # 1 mAh 가 옮기는 몰수
C_PER_MAH = MOL_PER_MAH / V_TOTAL_M3      # = 1.866 mol/m^3 per mAh (20 mL 기준)

DT_MIN = 60.0                    # 시간격자 [min].
#   120 -> 60 으로 줄였다. 선택 기준을 validation + COMSOL 검증 홀드아웃으로 바꿔
#   다시 잰 결과(2026-08-18, sweep_dt.py):
#     격자   조건   [val] V RMSE   농도RMSE   탈염오차   시간오차   학습
#     120분   263      6.28 mV      1.91      4.10%     30.2%    1018초
#      60분   355      6.55         1.39      1.96%     17.9%    1820초   <- 채택
#      30분   381      6.47         1.46      2.75%     15.4%    3470초
#   ★ 전압 RMSE 로는 격자를 못 고른다 — 세 값이 0.3 mV 안에 모여 있고, 이는
#     seed 변동(3.1 mV)보다 훨씬 작다. 예전 주석의 '9.5 -> 6.2 mV 개선' 은
#     test 로 재서 고른 값이라 낙관 편향이었다.
#   격자를 가르는 것은 농도 RMSE 와 COMSOL 검증 rollout 오차다. 60분이 농도
#   RMSE 와 탈염오차에서 최선이고, 30분은 시간오차만 근소하게 나으면서 탈염오차가
#   나빠지고(한계 조건이 26개 더 들어옴) 비용이 1.9배다.
SPLIT  = (0.7, 0.15, 0.15)       # train / val / test — 조건 단위로 나눈다
SEED   = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════════════════════════════════════════════
# 2. 데이터
#
#   COMSOL 이 뱉은 cond_XXX.csv 를 '조건 하나 = 궤적 하나' 로 바꾼다.
#   원본은 충전1·방전1·충전2·... 가 따로 저장돼 있고 시간격자도 불규칙하므로
#     ① 각 구간을 일정 간격(DT_MIN=2시간)으로 보간하고
#     ② 시간 순서대로 이어 붙이고
#     ③ 누적전하 Q_cum 과 Faraday 잔차 c_res 를 계산한다.
#
#   왜 2시간 격자인가 : 촘촘하면 스텝 수가 늘어 학습이 느려지고, 성기면
#   전압이 급변하는 구간(knee)을 놓친다. 2시간이면 조건당 13~300스텝이다.
# ════════════════════════════════════════════════════════════════════════
def _trim_truncated(run):
    """충전이 4.2 V 에 못 닿고 끝난 사이클 '앞까지'만 남긴다.

    조건을 통째로 버리면 데이터의 94%가 날아간다(475 -> 28). 대신 처음으로
    잘린 충전이 나오는 사이클부터 끝까지를 잘라내고, 그 앞의 온전한 사이클만
    쓴다. 궤적은 앞에서부터 이어지므로 앞부분만 남겨도 인과적으로 온전하다.

    남길 사이클이 하나도 없으면 None 을 돌려준다.
    """
    bad = None
    for cy, g in run[run['phase'] == 'charge'].groupby('cycle'):
        if float(g['Vcell'].iloc[-1]) < TRUNC_V_MAX:
            bad = int(cy)
            break
    if bad is None:
        return run
    if bad <= 1:
        return None
    out = run[run['cycle'] < bad]
    return out if len(out) >= 8 else None


def _is_truncated(run):
    """충전 구간 중 하나라도 tlist 상한(80 mAh)에 걸려 끝났으면 True."""
    for _, g in run[run['phase'] == 'charge'].groupby('cycle'):
        I = g['I'].to_numpy(float)
        t = g['t'].to_numpy(float)
        if len(t) < 2:
            continue
        if float(g['Vcell'].iloc[-1]) < TRUNC_V_MAX:
            return True
    return False


def _resample(seg, dt_min):
    """한 구간(충전 또는 방전)을 균일 시간격자에 보간한다."""
    seg = seg.sort_values('t')
    t = seg['t'].to_numpy(float)
    if len(seg) < 4 or t[-1] < dt_min:
        return None                                   # 격자점이 2개 미만이면 버린다

    grid = np.arange(0.0, t[-1] + 1e-9, dt_min)
    out = pd.DataFrame({'t': grid})
    for c in ('Vcell', 'cNa_C2', 'I_applied', 'nNa'):
        out[c] = np.interp(grid, t, seg[c].to_numpy(float))
    if 'c_mix' in seg.columns:                        # 추출기가 넣어준 혼합 농도
        out['c_mix'] = np.interp(grid, t, seg['c_mix'].to_numpy(float))

    # c_tank 는 방전 구간에서 NaN 이다(ODE 가 비활성). 유한한 점만으로 보간한다.
    ct = seg['c_tank'].to_numpy(float)
    ok = np.isfinite(ct)
    out['c_tank'] = np.interp(grid, t[ok], ct[ok]) if ok.any() else np.nan

    out = out.rename(columns={'I_applied': 'I'})
    out['c_in'] = float(seg['c_in'].iloc[0])
    out['cycle'] = int(seg['cycle'].iloc[0])
    out['phase'] = seg['phase'].iloc[0]
    return out


def load(use=None, dt_min=DT_MIN, verbose=True):
    """cond_*.csv 들을 읽어 {조건이름: 궤적 DataFrame} 으로 만든다.

    조건 하나 = 궤적 하나. 충전1 -> 방전1 -> 충전2 ... 순서로 이어 붙인다.
    폴더가 달라도 cond_id 가 1부터 겹치므로 '폴더#번호' 를 조건 이름으로 쓴다.
    """
    keys = list(use or USE)
    for k in keys:                       # 재격자본이 원본보다 뒤처졌으면 여기서 경고
        _check_sync(k, DATASETS[k])
    dirs = [DATASETS[k] for k in keys]
    runs = {}
    n_trunc = n_trim = 0

    for d in dirs:
        files = sorted(f for f in os.listdir(d)
                       if f.startswith('cond_') and f.endswith('.csv')) if os.path.isdir(d) else []
        if not files:
            if verbose:
                print(f'  [{os.path.basename(d)}] 데이터 없음 — 건너뜀')
            continue

        for f in files:
            raw = pd.read_csv(os.path.join(d, f))
            if raw.empty or 'I_applied' not in raw.columns:
                continue
            name = f'{os.path.basename(d)}#{int(raw["cond_id"].iloc[0]):03d}'

            # 충전 -> 방전 순으로 정렬해 시간순 궤적을 만든다
            parts = []
            for (cyc, ph), seg in raw.groupby(['cycle', 'phase']):
                s = _resample(seg, dt_min)
                if s is not None:
                    parts.append(((int(cyc), ph != 'charge'), s))
            if not parts:
                continue
            run = pd.concat([s for _, s in sorted(parts, key=lambda x: x[0])],
                            ignore_index=True)

            # 방전 구간의 c_tank 를 직전 충전 끝값으로 채운다(방전 중 탱크는 보존)
            run['c_tank'] = run['c_tank'].ffill().fillna(run['c_in'])

            # 혼합 농도 — 실제로 회수하는 물의 농도. 추출기(comsol_setup)가 이미 넣어주지만
            # 구 데이터에는 없으므로 여기서 만들어 준다.
            if 'c_mix' not in run.columns:
                run['c_mix'] = ((V_CHAMBER_ML * run['cNa_C2'] + V_TANK_ML * run['c_tank'])
                                / (V_CHAMBER_ML + V_TANK_ML))

            # 누적 전하 [mAh] — 궤적 전체에 걸쳐 다시 계산한다
            I = run['I'].to_numpy(float)
            run['Q_cum'] = np.r_[0.0, np.cumsum(I[:-1])] * dt_min / 60.0

            # Faraday 예측과 그 잔차. 전하보존만으로 정해지는 값이라 정확하다.
            run['c_far'] = run['c_in'] - C_PER_MAH * run['Q_cum']
            run['c_res'] = run['c_mix'] - run['c_far']

            if DROP_TRUNCATED:
                full = len(run)
                run = _trim_truncated(run)
                if run is None:
                    n_trunc += 1
                    continue
                if len(run) < full:
                    n_trim += 1
            if len(run) >= 8:
                runs[name] = run

        if verbose:
            n = sum(1 for k in runs if k.startswith(os.path.basename(d)))
            print(f'  [{os.path.basename(d)}] 조건 {n}개')

    if verbose and n_trunc:
        print(f'  [절단] 1사이클부터 잘려 통째로 제외 {n_trunc}개, '
              f'뒷사이클만 잘라내고 사용 {n_trim}개  (기준 {TRUNC_V_MAX} V)')
    if not runs:
        raise FileNotFoundError(f'데이터가 없다: {dirs}')
    if verbose:
        steps = [len(r) for r in runs.values()]
        print(f'  총 {len(runs)}조건, 스텝수 {min(steps)}~{max(steps)} '
              f'(중앙 {int(np.median(steps))}), 격자 {dt_min:.0f}분')
    return runs


def split(runs, frac=SPLIT, seed=SEED, verbose=True):
    """조건 단위로 나눈다. 같은 조건이 train 과 test 에 섞이면 성능이 부풀려진다."""
    names = sorted(runs)
    np.random.default_rng(seed).shuffle(names)
    n_tr = max(1, round(len(names) * frac[0]))
    n_va = max(1, round(len(names) * frac[1]))
    cut = [names[:n_tr], names[n_tr:n_tr + n_va], names[n_tr + n_va:]]
    parts = [{k: runs[k] for k in c} for c in cut]
    if verbose:
        print(f'  분할 train/val/test = {len(parts[0])}/{len(parts[1])}/{len(parts[2])}')
    return parts


class Scaler:
    """학습셋의 평균·표준편차로 표준화한다.

    왜 필요한가 : 입력들의 크기가 제각각이다.
        Vcell   3~4          (V)
        c_res  −300~0        (mol/m^3)
        I      −1.5~1.5      (mA)
        Q_cum   0~200        (mAh)
        c_in    500~700      (mol/m^3)
    이대로 넣으면 큰 값(c_in)이 손실을 지배해 작은 값(Vcell)을 못 배운다.
    각 열을 (값−평균)/표준편차 로 바꿔 모두 비슷한 크기로 만든다.

    ※ 평균·표준편차는 '학습셋'에서만 구한다. 검증셋 정보가 새면 성능이 부풀려진다.
    """

    def fit(self, runs):
        data = np.vstack([r[COLUMNS].to_numpy(float) for r in runs.values()])
        self.mean, self.std = data.mean(0), data.std(0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def forward(self, run):
        """DataFrame -> (T, 5) 표준화 배열. 앞 2열이 타깃, 뒤 3열이 입력."""
        return (run[COLUMNS].to_numpy(float) - self.mean) / self.std

    def back(self, y):
        """표준화된 타깃 (T, 2) -> 원래 단위."""
        return y * self.std[:N_OUT] + self.mean[:N_OUT]


# ════════════════════════════════════════════════════════════════════════
# 3. 모델
#
#   다섯 모델이 전부 같은 서명을 갖는다. 그래야 공정한 비교가 된다.
#
#       forward(init, exo) -> traj
#          init (B, 2)     t=0 의 [Vcell, c_res]
#          exo  (B, T, 3)  매 시점의 [I, Q_cum, c_in]
#          traj (B, T, 2)  매 시점의 [Vcell, c_res]
#       B = 배치(여기서는 1, 궤적 하나씩 학습), T = 시간 스텝 수
#
#   왜 다섯 개나 비교하는가
#     MLP·XGBoost 는 '각 시점을 서로 독립'으로 본다. 이들이 시퀀스 모델을
#     이기면 시간 구조가 필요 없다는 뜻이고, 지면 필요하다는 증거가 된다.
#     실제 결과 : MLP 42.9 mV / XGBoost 49.5 mV  vs  TFT 10.2 mV
#                 -> 시간 구조가 반드시 필요하다는 것이 데이터로 증명됨
# ════════════════════════════════════════════════════════════════════════
# 모델 정의는 파일별로 분리되어 있다(발표·설명하기 쉽도록).
#   model_mlp.py        각 시점 독립 — 시간 구조가 필요한지 판별하는 기준선
#   model_lstm.py       이력을 은닉상태로 나름
#   model_lstm_attn.py  + 먼 시점 직접 참조 (미래 차단 마스크)
#   model_tft.py        + 정적문맥 · 게이트 잔차 (가장 정확)
#   XGBoost 는 torch 가 아니므로 이 파일 아래쪽에 그대로 둔다.
#
# 지연 로딩(_Lazy)을 쓰는 이유 : 모델 파일이 이 파일을 import 하므로,
# 여기서 모델 파일을 즉시 import 하면 순환참조가 된다.
# 라인업은 '설계 공간을 나누는가'로 고른다.
#   시점 독립      MLP, XGBoost
#   순환만         LSTM
#   어텐션만       Transformer
#   순환+어텐션+VSN+정적공변량   TFT
# LSTM-Attn(= LSTM + 어텐션)은 TFT 의 부분집합이라 겹쳐서 뺐다. 값은
# compare_all.csv 에 남으므로 SI 표에는 계속 쓸 수 있다.
_MODEL_SPEC = {
    'MLP':         ('model_mlp', 'MLP'),
    'LSTM':        ('model_lstm', 'LSTM'),
    'Transformer': ('model_transformer', 'Transformer'),
    'TFT':         ('model_tft_re', 'TFT'),
}


class _Lazy(dict):
    def __missing__(self, key):
        import importlib
        mod, cls = _MODEL_SPEC[key]
        self[key] = getattr(importlib.import_module(mod), cls)
        return self[key]


TORCH_MODELS = _Lazy()


# ════════════════════════════════════════════════════════════════════════
# 4. 학습
#
#   궤적 하나를 배치 하나로 쓴다. 조건마다 길이가 달라 묶기 어렵기 때문이다.
#   손실은 표준화된 공간에서의 MSE 이고, 매 에폭 검증셋으로 확인해
#   가장 좋았던 순간의 가중치를 남긴다(조기종료와 같은 효과).
# ════════════════════════════════════════════════════════════════════════
def _tensor(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)


# 4.2 V 근처 가중 (Trainer._loss). 0 이면 균등 MSE(예전 동작).
#   충전 종료 시각이 이 근처 몇 점으로 정해지는데 균등 MSE 는 그걸 반영하지 못한다.
V_CUT = 4.2
# 실측: w=3 이면 시간오차 28.3->27.9% (거의 그대로)인데 탈염오차 3.9->4.5%,
# w=8 이면 탈염 5.7% / 시간 34.4% 로 전부 악화. 효과가 없어 0 으로 되돌린다.
WEIGHT_NEAR_CUT = 0.0        # 차단선에서의 추가 가중 (0 = 끔)
WEIGHT_SIGMA = 0.08          # V 단위 — 4.12~4.28 구간이 주로 가중된다


class Trainer:
    """torch 모델 하나를 학습한다. 궤적 하나가 배치 하나다(길이가 조건마다 달라서)."""

    def __init__(self, model_class, epochs=150, lr=1e-3, seed=SEED, **kwargs):
        self.model_class, self.epochs, self.lr = model_class, epochs, lr
        self.seed, self.kwargs = seed, kwargs
        self.history = []                                     # (train_loss, val_loss)

    def _loss(self, run):
        """run: (T, 5) 표준화 텐서. 앞 2열이 타깃.

        4.2 V 근처에 가중치를 준다.
            충전 종료 시각은 전압이 4.2 V 를 언제 지나느냐로 정해지는데,
            균등 MSE 는 그 몇 점을 전 구간의 한 점으로만 취급한다. 실제로
            전압 RMSE 8.8 mV 로 좋은데도 rollout 의 소요시간이 +28% 틀렸다.
            차단선 근처 오차가 그대로 시각 오차가 되므로 그 구간을 더 본다.

            w = 1 + WEIGHT_NEAR_CUT * exp(-((V - V_MAX)/WEIGHT_SIGMA)^2)
        가중치는 표준화 전 전압으로 계산해야 하므로 스케일러 값을 되돌린다.
        """
        init = run[:1, :N_OUT]
        exo = run[None, :, N_OUT:]
        pred = self.net(init, exo)
        tgt = run[None, :, :N_OUT]
        if not WEIGHT_NEAR_CUT:
            return F.mse_loss(pred, tgt)
        v = tgt[..., 0] * self._v_std + self._v_mean          # 실제 전압 [V]
        w = 1.0 + WEIGHT_NEAR_CUT * torch.exp(
            -((v - V_CUT) / WEIGHT_SIGMA) ** 2)
        return (w[..., None] * (pred - tgt) ** 2).mean()

    def fit(self, train, val, scaler, verbose=False, on_epoch=None):
        """on_epoch(ep, val_loss) -> bool. True 를 돌려주면 그 자리에서 멈춘다.

        하이퍼파라미터 탐색이 가망 없는 설정을 일찍 접기 위한 훅이다.
        기본값 None 이면 아무 일도 하지 않으므로 기존 학습은 그대로다.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        # 4.2 V 가중을 계산하려면 표준화를 되돌려야 한다 — 전압은 0번 열이다.
        self._v_mean = float(scaler.mean[0])
        self._v_std = float(scaler.std[0])
        self.net = self.model_class(**self.kwargs).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)

        train = [_tensor(scaler.forward(r)) for r in train.values()]
        val = [_tensor(scaler.forward(r)) for r in val.values()]
        best, best_state = np.inf, None

        for ep in range(self.epochs):
            self.net.train()
            losses = []
            for run in random.sample(train, len(train)):
                opt.zero_grad()
                loss = self._loss(run)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            sched.step()

            vl = np.nan
            if val:
                self.net.eval()
                with torch.no_grad():
                    vl = float(np.mean([self._loss(r).item() for r in val]))
                if vl < best:                                 # 가장 좋았던 순간을 남긴다
                    best = vl
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net.state_dict().items()}
            self.history.append((float(np.mean(losses)), vl))
            if verbose and (ep + 1) % 25 == 0:
                print(f'    epoch {ep+1:3d}  train {np.mean(losses):.5f}  val {vl:.5f}')
            if on_epoch is not None and on_epoch(ep, vl):
                break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, scaled):
        """scaled: (T, 5) 표준화 배열 -> (T, 2) 표준화된 예측."""
        run = _tensor(scaled)
        with torch.no_grad():
            return self.net(run[:1, :N_OUT], run[None, :, N_OUT:])[0].cpu().numpy()


class XGBoost:
    """트리 기준선. 각 시점을 독립 표본으로 보고 타깃마다 회귀기를 하나씩 둔다.

    시계열 구조를 전혀 안 쓰므로, 신경망이 이걸 못 이기면 '시간 모델이 필요없다'는
    증거가 된다. 학습이 몇 초로 빨라서 비교 기준으로 유용하다.
    """

    def __init__(self, n_estimators=600, lr=0.05, max_depth=6, seed=SEED, **_):
        self.args = dict(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth,
                         random_state=seed, subsample=0.8, colsample_bytree=0.8, verbosity=0)
        self.history = []                                     # Trainer 와 인터페이스를 맞춘다

    @staticmethod
    def _features(scaled):
        init = np.repeat(scaled[:1, :N_OUT], len(scaled), axis=0)
        return np.c_[init, scaled[:, N_OUT:]]

    def fit(self, train, val, scaler, verbose=False):
        from xgboost import XGBRegressor
        rows = [scaler.forward(r) for r in train.values()]
        X = np.vstack([self._features(r) for r in rows])
        Y = np.vstack([r[:, :N_OUT] for r in rows])
        self.models = [XGBRegressor(**self.args).fit(X, Y[:, j]) for j in range(N_OUT)]
        return self

    def predict(self, scaled):
        X = self._features(scaled)
        return np.column_stack([m.predict(X) for m in self.models])


def xgb_cached(seed, train, val, scaler, ckpt_dir):
    """XGBoost 를 파일에 남겨 두고 두 번째부터는 그것을 읽는다.

    ★ 왜 캐시하나 — 학습이 4초라 매번 새로 돌려도 된다고 봤는데, random_state
      를 seed 로 고정했는데도 실행마다 값이 달랐다(seed 0 전압 RMSE 30.57 vs
      30.83 mV, 4.2 V 도달오차 4.29 vs 3.96 h). 논문 표의 한 행이 돌릴 때마다
      움직이면 안 된다. torch 모델이 figures_{이름}_s{seed}.pt 를 쓰는 것과
      같은 방식으로 부스터를 남긴다. 타깃(전압·농도)마다 회귀기가 하나씩이라
      파일도 둘이다.

    값을 새로 뽑고 싶으면 figures_XGBoost_s*.json 을 지우고 다시 돌릴 것.
    """
    m = build('XGBoost', seed=seed)
    files = [os.path.join(ckpt_dir, f'figures_XGBoost_s{seed}_t{j}.json')
             for j in range(N_OUT)]
    if all(os.path.exists(f) for f in files):
        from xgboost import XGBRegressor
        m.models = []
        for f in files:
            r = XGBRegressor(**m.args)
            r.load_model(f)
            m.models.append(r)
        return m
    m.fit(train, val, scaler)
    os.makedirs(ckpt_dir, exist_ok=True)
    for r, f in zip(m.models, files):
        r.save_model(f)
    return m


# 하이퍼파라미터 탐색 결과 (2026-08-19, tune_hparams.py)
#   Optuna TPE, 모델마다 25시도 x 60에폭, validation 손실 기준으로만 선택.
#   test 는 탐색에 쓰지 않았다. 전 시도 기록은 _S_result/tune_hparams.csv.
#   ※ 각 model_*.py 의 __init__ 기본값은 손대지 않는다 — 탐색 전 값이 무엇이었는지
#     남겨 두어야 "무엇을 어떻게 바꿨는가" 를 나중에 확인할 수 있다.
TUNED = {
    'MLP':         dict(lr=2.912e-3, hidden=512, dropout=0.0),
    'LSTM':        dict(lr=1.902e-3, hidden=256, layers=3, dropout=0.30),
    'Transformer': dict(lr=4.574e-4, hidden=128, heads=4, layers=4, ff=128,
                        dropout=0.0),
    'TFT':         dict(lr=6.337e-4, hidden=64, heads=8, dropout=0.0),
    'AR':          dict(lr=6.801e-4, hidden=128, lag=8, dropout=0.0),
}


def build(name, **kwargs):
    """이름으로 모델을 만든다. 'MLP' | 'LSTM' | 'LSTM-Attn' | 'TFT' | 'XGBoost'"""
    if name == 'XGBoost':
        return XGBoost(**kwargs)
    if name not in _MODEL_SPEC:                 # ※ TORCH_MODELS 는 지연 로딩이라
        raise KeyError(f'모르는 모델: {name}  '   #   비어 있을 수 있다. 명세로 검사한다.
                       f'(가능: {ALL_MODELS})')
    cfg = dict(TUNED.get(name, {}))             # 탐색으로 고른 설정을 기본으로
    cfg.update(kwargs)                          # 호출부가 준 값이 우선한다
    return Trainer(TORCH_MODELS[name], **cfg)


ALL_MODELS = list(_MODEL_SPEC) + ['XGBoost']


# ════════════════════════════════════════════════════════════════════════
# 5. 평가
#
#   RMSE·R² 만으로는 부족하다. 이 연구에서 실제로 중요한 것은
#     · '4.2 V 에 언제 닿는가'  — 충전이 언제 끝나는지가 운전을 정한다
#     · 미래 정보를 훔쳐보지 않았는가 — 시퀀스 모델의 흔한 함정
#   그래서 ceiling_error() 와 causality() 를 따로 둔다.
# ════════════════════════════════════════════════════════════════════════
def to_c_mix(pred, run):
    """예측 (T,2) 를 [Vcell, c_mix] 로 되돌린다. CONC_TARGET 이 c_res 면 Faraday 를 더한다."""
    out = pred.copy()
    if CONC_TARGET == 'c_res':
        out[:, 1] = out[:, 1] + run['c_far'].to_numpy(float)
    return out


def r2(pred, true):
    var = np.var(true)
    return np.nan if var < 1e-12 else float(1 - np.mean((pred - true) ** 2) / var)


def r2_text(v):
    """R^2 를 그림에 적을 때 쓰는 문자열.

    소수 넷째 자리로 적으면 이 문제에서는 거의 모든 패널이 1.0000 이 된다
    (실제 값 0.99995). '완벽한 예측' 으로 읽혀 불필요한 의심을 부르므로
    다섯째 자리까지 적고, 그래도 1 로 반올림되면 부등호로 적는다.
    """
    return '> 0.99999' if round(v, 5) >= 1.0 else f'{v:.5f}'


def causality(model, run, scaler, bump=5.0):
    """마지막 시점의 전류를 흔들었을 때 '그 이전' 예측이 얼마나 변하는가.

    물리적으로는 0이어야 한다(미래가 과거를 바꿀 수 없다). 0이 아니면 모델이
    미래 정보를 훔쳐보고 있다는 뜻이고, 그 성능은 실제로는 못 낸다.
    """
    a = scaler.forward(run)
    b = a.copy()
    b[-1, N_OUT] += bump                                      # 마지막 I 만 흔든다
    return float(np.abs(model.predict(a)[:-1] - model.predict(b)[:-1]).max())


def ceiling_error(pred, run, dt_min=DT_MIN):
    """충전 구간 끝(= 전압 상한 도달점)의 전압 오차와, 그것이 시간으로 몇 시간인지.

    운전에서 중요한 건 곡선 전체보다 '언제 상한에 닿는가'다. 전압 오차를 그 지점의
    기울기로 나누면 '몇 시간 일찍/늦게 도달한다고 예측했는가'가 된다.
    """
    mV, hours = [], []
    dt_h = dt_min / 60.0
    for (_, phase), seg in run.groupby(['cycle', 'phase']):
        if phase != 'charge' or len(seg) < 3:
            continue
        i = seg.index.to_numpy()
        err = abs(pred[i[-1], 0] - run['Vcell'].iloc[i[-1]])
        slope = abs(pred[i[-1], 0] - pred[i[-3], 0]) / (2 * dt_h)
        mV.append(err * 1000)
        hours.append(err / max(slope, 0.005))                 # 기울기 0 방어
    return mV, hours


def evaluate(model, runs, scaler):
    """모델을 궤적 묶음에 대해 평가한다."""
    true_all, pred_all = [], []
    seg_r2 = {t: [] for t in TARGETS}
    cyc_rmse, ceil_mV, ceil_h = {}, [], []

    for run in runs.values():
        pred = scaler.back(model.predict(scaler.forward(run)))
        true = run[TARGETS].to_numpy(float)
        true_all.append(true)
        pred_all.append(pred)

        for (cyc, _), seg in run.groupby(['cycle', 'phase']):
            i = seg.index.to_numpy()
            for j, name in enumerate(TARGETS):
                seg_r2[name].append(r2(pred[i, j], true[i, j]))
            cyc_rmse.setdefault(int(cyc), []).append(
                float(np.sqrt(np.mean((pred[i, 0] - true[i, 0]) ** 2))))

        v, h = ceiling_error(pred, run)
        ceil_mV += v
        ceil_h += h

    true_all, pred_all = np.vstack(true_all), np.vstack(pred_all)
    out = {'n_run': len(runs), 'n_seg': len(ceil_mV)}
    for j, name in enumerate(TARGETS):
        out[f'{name}_RMSE'] = float(np.sqrt(np.mean((pred_all[:, j] - true_all[:, j]) ** 2)))
        out[f'{name}_R2'] = r2(pred_all[:, j], true_all[:, j])
        # 구간별 R² — 전체 R² 는 사이클 간 큰 변동 덕에 쉽게 높아진다. 이쪽이 정직하다.
        out[f'{name}_R2_seg'] = float(np.nanmedian(seg_r2[name]))
    out['cyc_rmse'] = {k: float(np.median(v)) for k, v in sorted(cyc_rmse.items())}
    out['ceil_mV'] = float(np.median(ceil_mV)) if ceil_mV else np.nan
    out['ceil_h'] = float(np.median(ceil_h)) if ceil_h else np.nan
    return out

# ════════════════════════════════════════════════════════════════════════
# 6. 시각화
#
#   COMSOL 값(observed)과 예측값(expectation)을 같은 축에 겹쳐 그린다.
#   가로 = train / validation / test,  세로 = Vcell / c_mix
#   각 칸의 오른쪽 아래에 그 분할 '전체'에 대한 R² 와 RMSE 를 적는다.
#   (그려진 궤적만이 아니라 분할 전체 기준이라는 점에 주의)
# ════════════════════════════════════════════════════════════════════════
def split_metrics(model, runs, scaler):
    """분할 전체에 대한 지표. R² 를 두 가지로 낸다.

    R2_all : 시계열을 전부 이어붙여 계산.
             전압은 충전 3.3~4.2 V / 방전 2.8~3.0 V 로 위상 간 차이가 커서,
             '충전이면 대략 3.7, 방전이면 대략 2.9' 만 맞혀도 0.99 가 나온다.
             분산의 대부분이 위상 차이에서 오기 때문이다. -> 부풀려진 값.
    R2_seg : 구간(사이클×위상)별 R² 의 중앙값.
             구간 안에서는 전압 폭이 0.9 V 뿐이라 곡선 모양을 실제로 따라가야 한다.
             -> 이쪽이 모델을 가르는 정직한 지표.
    """
    obs = {'Vcell': [], 'c_mix': []}
    prd = {'Vcell': [], 'c_mix': []}
    seg = {'Vcell': [], 'c_mix': []}
    for run in runs.values():
        p = to_c_mix(scaler.back(model.predict(scaler.forward(run))), run)
        pair = {'Vcell': p[:, 0], 'c_mix': p[:, 1]}
        for k in obs:
            o = run[k].to_numpy(float)
            obs[k].append(o)
            prd[k].append(pair[k])
        for _, g in run.groupby(['cycle', 'phase']):
            i = g.index.to_numpy()
            for k in obs:
                seg[k].append(r2(pair[k][i], run[k].to_numpy(float)[i]))
    out = {}
    for k in obs:
        a, b = np.concatenate(obs[k]), np.concatenate(prd[k])
        out[k] = dict(r2_all=r2(b, a),
                      r2_seg=float(np.nanmedian(seg[k])),
                      rmse=float(np.sqrt(np.mean((b - a) ** 2))),
                      # MAE 도 같이 낸다. RMSE 는 제곱이라 큰 오차 몇 점에
                      # 끌려가므로, 둘을 나란히 보면 '전반적으로 어긋나는지
                      # 몇 군데서 크게 튀는지'가 갈린다.
                      mae=float(np.mean(np.abs(b - a))))
    return out


# 그림 공통 서식 — 논문/발표용
PLOT_FONT = 'Times New Roman'
OBS_COLOR = 'black'          # observed = COMSOL
PRED_COLOR = 'tab:red'       # expectation = surrogate


def _style():
    """그림 서식은 프로젝트 공통 모듈에서 가져온다(plotstyle.py 한 군데서 관리)."""
    sys.path.insert(0, BASE)
    import matplotlib
    matplotlib.use('Agg')
    import plotstyle
    return plotstyle.apply()


def plot_overlay(name, model, scaler, splits, out_path, n_show=1):
    """관측(COMSOL) vs 예측(surrogate) 궤적 겹쳐 그리기.

    가로 = train / validation / test,  세로 = Vcell / c_mix
    색은 두 가지뿐이다 — 검정 실선 = 관측(COMSOL), 빨강 파선 = 예측.

    각 칸에는 '조건 하나'만 그린다(n_show=1). 그 분할에서 길이가 중앙값에 가장
    가까운 조건을 대표로 고르고, 제목에 어떤 조건인지(c_in, 전류) 적는다.
    여러 개를 겹쳐 그리면 어느 선이 어느 조건인지 알 수 없어 오히려 읽기 나쁘다.

    오른쪽 아래 박스의 RMSE·R² 는 '그려진 조건'이 아니라 그 분할 '전체' 기준이다.
    """
    plt = _style()
    keys = list(splits)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.8 * len(keys), 7.2),
                             squeeze=False)

    for j, sname in enumerate(keys):
        runs = splits[sname]
        met = split_metrics(model, runs, scaler)
        med = np.median([len(v) for v in runs.values()])
        show = sorted(runs, key=lambda k: abs(len(runs[k]) - med))[:n_show]

        for row, tgt in enumerate(('Vcell', 'c_mix')):
            ax = axes[row, j]
            for key in show:
                run = runs[key]
                th = np.arange(len(run)) * DT_MIN / 60.0
                p = to_c_mix(scaler.back(model.predict(scaler.forward(run))), run)
                ax.plot(th, run[tgt].to_numpy(float), '-',
                        color=OBS_COLOR, lw=1.8, alpha=.9)
                ax.plot(th, p[:, 0 if tgt == 'Vcell' else 1], '--',
                        color=PRED_COLOR, lw=1.5, alpha=.95)

            m = met[tgt]
            unit = 'V' if tgt == 'Vcell' else 'mol m$^{-3}$'
            txt = (f'RMSE = {m["rmse"]:.3g} {unit}\n'
                   f'$R^2_{{\\rm all}}$ = {r2_text(m["r2_all"])}\n'
                   f'$R^2_{{\\rm seg}}$ = {m["r2_seg"]:.4f}')
            ax.text(0.975, 0.05, txt, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=12, linespacing=1.5,
                    bbox=dict(boxstyle='round,pad=0.45', fc='white',
                              ec='0.5', lw=0.8, alpha=.92))
            ax.set_ylabel('Cell voltage (V)' if tgt == 'Vcell'
                          else 'Na$^+$ concentration (mol m$^{-3}$)')
            ax.grid(True, ls=':', alpha=.5)
            if row == 0:
                r0 = runs[show[0]]
                ich = '/'.join(f'{abs(g["I"].iloc[0]):.2f}' for _, g in
                               r0[r0['phase'] == 'charge'].groupby('cycle'))
                sub = ('shown: $c_{in}$ = '
                       f'{r0["c_in"].iloc[0]:.0f}' + ' mol m$^{-3}$,  '
                       '$I_{chg}$ = ' + ich + ' mA')
                ax.set_title(f'{sname}  (n = {len(runs)} conditions)'
                             + chr(10) + sub, fontsize=13)
            else:
                ax.set_xlabel('Time (h)')

    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[
        Line2D([], [], color=OBS_COLOR, ls='-', lw=1.8, label='Observed (COMSOL)'),
        Line2D([], [], color=PRED_COLOR, ls='--', lw=1.5,
               label='Predicted (surrogate)')],
        loc='upper left', framealpha=.9)

    fig.suptitle(name, fontsize=17, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975], w_pad=3.4)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return out_path


def run_single(name, use=None, epochs=150, out_dir=None, verbose=True):
    """모델 하나를 학습하고 궤적 그림까지 만든다. 각 model_*.py 가 이걸 부른다."""
    out_dir = out_dir or RESULT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f'[데이터]')
    runs = load(use or USE)
    train, val, test = split(runs)
    scaler = Scaler().fit(train)

    print(f'\n[{name}] 학습 ({len(train)}조건, {epochs}에폭)...')
    import time
    t0 = time.time()
    model = build(name, epochs=epochs).fit(train, val, scaler)
    sec = time.time() - t0

    res = evaluate(model, test, scaler)
    print(f'  {sec:.0f}초   test  V RMSE {res["Vcell_RMSE"]*1000:.1f} mV   '
          f'R2(seg) {res["Vcell_R2_seg"]:.4f}   '
          f'4.2V 도달시각 오차 {res["ceil_h"]:.2f} h')

    p = plot_overlay(name, model, scaler,
                     {'train': train, 'validation': val, 'test': test},
                     os.path.join(out_dir, f'overlay_{name.replace("-","")}.png'))
    print('saved:', p)
    return model, scaler, (train, val, test)
