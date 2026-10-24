# -*- coding: utf-8 -*-
"""충방전 사이클 데이터 추출 — 세 프로토콜을 스위치 하나로.

    PROTOCOL = 'cap'     충전 = 정격/N (13.23 mAh) 또는 4.2 V 중 먼저   → 방전 98% → 반복
    PROTOCOL = 'vlim'    충전 = 4.2 V 도달까지 (용량 제한 없음)          → 방전 98% → 반복
    PROTOCOL = 'vstart'  vlim 과 같되 담수부 초기농도·음극 초기재고를 조건마다 바꾼다

셋의 차이는 '충전을 언제 끝내는가'와 '어떤 상태에서 시작하는가' 뿐이라 러너 하나로 둔다.
방전은 셋 다 동일: 그 사이클 충전 실측전하의 98% (쿨롱효율).

실행:
    python 1_extract/run_extraction.py

중단해도 됨 — cond_XXX.csv 가 있으면 건너뛴다(resume).
비수렴 조건은 _skipped.txt 에 기록되고 프로세스가 자기재시작한다.
"""
import os
import sys

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_BASE = _ROOT
sys.path.insert(0, _os.path.join(_ROOT, '1_extract'))      # COMSOL 코어(_D_swb_core)
sys.path.insert(0, os.path.join(_BASE, '1_extract'))


import pandas as pd
import comsol_setup as CAL

C = CAL.apply()

NEWLINE = chr(10)

# ===== 설정 =========================================================================
# 프로토콜과 코어 수를 명령행으로 받는다 — 두 프로토콜을 동시에 돌리기 위해서다.
#     창1:  python 1_extract/run_extraction.py cap  4
#     창2:  python 1_extract/run_extraction.py vlim 4
# 출력 폴더가 프로토콜마다 다르므로 충돌하지 않고, COMSOL 서버도 각자 띄운다
# (skip_and_restart 는 자기가 띄운 서버 PID 만 죽인다).
# ※ 두 번째 창은 5~10초 뒤에 띄울 것 — 동시에 시작하면 서버 PID 를 서로 오인할 수 있다.
# 인자 없이 그냥 실행하면 아래 기본값으로 돈다 (VS Code 의 실행 버튼 등).
# 지금 뽑아야 할 것이 ratio2 라서 기본값을 그것으로 둔다. 다른 것을 뽑을 때는
# 이 두 줄만 고치거나, 명령행에 이름을 주면 된다.
DEFAULT_PROTOCOL = 'ratio2'
DEFAULT_CORES = 8

PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROTOCOL
CORES = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CORES

# 방전 동역학 민감도 확인용 (260912)
#   i0=<값>  ORR 교환전류밀도를 덮어쓴다. mph 저장값 1 A/m^2 은 보정된 적이 없다
#            (파라미터표 A-5 '고정한 파라미터'). 이 값을 바꿔 돌리면 방전 전압만
#            달라지고 방전 '시간' 은 쿨롱 수지(0.98 x 충전전하 / 방전전류)로
#            정해지므로 그대로다. 탈염량이 방전 동역학에 둔감한지, 회수율 eta 는
#            얼마나 민감한지를 가른다.
#   tag=<꼬리표>  출력 폴더 이름 뒤에 붙여 판끼리 섞이지 않게 한다.
I0_ORR_OVERRIDE = None
RUN_TAG = ''
for _a in sys.argv[3:]:
    if _a.startswith('i0='):
        I0_ORR_OVERRIDE = float(_a.split('=')[1])
    elif _a.startswith('tag='):
        RUN_TAG = '_' + _a.split('=')[1]
WATER = 'seawater'

N_CYCLES = 3

# 충전 tlist 안전 상한 [mAh]. 종료조건이 아니라 '이보다 오래 풀지 않는다'는 뜻이다.
# 셀 용량은 율속 의존이며 저율속에서 70~80 mAh 까지 충전 가능(제조사).
CAP_CEIL_MAH = 80.0
MAX_OUT_PTS = 600      # 충전 1회당 출력점 상한 — 격자가 자동으로 성겨진다
N_TEST = None                 # None = 전부. 처음엔 3~5 로 시간을 재고 나서 전체를 걸 것.

# ratio = 방전/충전 전류비가 큰 조건. lhs6 은 그 비가 1 근처(최대 1.75)라
#   NSGA-II 최적해가 사는 영역(1.7~2.1)을 학습셋이 못 덮었다. 그 구멍을 메운다.
#   전류를 사이클마다 읽는 방식은 lhs6 과 같으므로 별도 분기가 필요 없다.
#   ratio2 = 그 수정판. ratio 판은 충전전류를 0.50 까지 낮게 뽑는 바람에 44%가
#   4.2 V 전에 80 mAh 상한에 걸려 절단·폐기됐다. 충전 0.70~0.95 로 올렸다.
# f10opt  = Fig.10 재작성용. 17농도의 파레토 평균 전류(사이클마다 다름).
# f10base = 그 짝이 되는 정전류 격자. 둘 다 종료조건은 vlim/lhs6 과 같으므로
#           (충전 4.2 V, 방전 98%) 별도 분기가 필요 없다. 이름만 등록한다.
# f10base3 = f10base 의 방전 규약을 최적 프로토콜에 맞춘 판. 충전은 정전류지만
#           방전은 min(1.5, 2.24*I) 로 빠르게 한다. f10base 는 방전/충전 비가
#           1.00 인데 f10opt/f10uto 는 1.68/2.01 이라 짝이 맞지 않았다. 그래서
#           Fig.10 의 +11.3% 중 상당 부분이 계단 충전이 아니라 짧은 방전에서
#           왔다. 이 집합이 그 교란을 없앤다.  (1_extract/make_f10base3.py)
# *_d10   = 위와 같되 방전을 1.0 mA 로 한 판. 1.5 mA 는 추출 범위의 위끝일 뿐
#           실측이 없고(실험은 1.00 / 0.503 mA), 1.0 이면 rmax 가드도 필요
#           없어진다(2.24 x 0.5 = 1.12 > 1.0). 본문은 이쪽으로 간다.
#           f10base_d10 / _d10b 는 같은 기준선 곡선을 짝·홀로 나눈 것이다 —
#           두 프로세스로 동시에 돌리려고 쪼갰고, 둘 다 17농도를 고루 덮는다.
#           (1_extract/make_f10_d10.py)
# f10base_d10c = 그 곡선의 짧은 시간 쪽 보강. f10opt_d10(147~269 h) 이 곡선의
#           아래끝(160~270 h)보다 빨라 17농도 중 2개만 짝이 맞았다. 목표 전류는
#           이미 돌린 62조건의 실측으로 hours = a + b/I 를 맞춰 역산했다 —
#           대리모델의 +11% 시간 편향을 거치지 않는다.
#           (1_extract/make_f10base_d10c.py)
# f10lit10 / f10lit20 = 문헌식 '등간격 감소 계단'. 충전을 일정 간격으로 낮추고
#           방전은 그 사이클 충전 전류와 같게 둔다(선행 연구 관행). 감소폭 0.10 /
#           0.20 mA 두 가지. 짝이 되는 기준선은 f10base/f10base2 (관행 정전류,
#           방전=충전) 다. 대리모델 예측은 관행 대비 +0.48% / -0.58% 로, 충전
#           파형만 계단화해서는 정전류를 못 넘는다 — 그것을 실측으로 확인한다.
#           (1_extract/make_f10lit.py)
# f10uto3 = 3목적 (시간, 탈염량, 순에너지) 최적화의 농도별 utopia 해 17개.
#           충·방전 6전류가 모두 자유 변수다 — 방전을 고정하지 않았으므로
#           짝이 되는 기준선은 관행 정전류(f10base/f10base2, 방전=충전) 다.
#           (pareto_front_3obj_rmax224.csv -> fig_3obj_utopia.csv 에서 생성)
# f10dom  = 3목적 전선에서 '관행 정전류를 같은 탈염량에서 시간·에너지 둘 다
#           이기는' 해 중 이상점에 가장 가까운 것. 대리모델 기준 여유가 얇다
#           (에너지 최악 0.02%) — COMSOL 에서 뒤집히는지 보는 것이 목적이다.
# f10dom2 = 같은 조건에서 min(시간여유, 에너지여유) 를 최대화한 해. 여유 중앙
#           6.7%, 최악 1.3% 로 두꺼워 COMSOL 에서도 지배가 남을 가능성이 크다.
_KNOWN = ('cap', 'vlim', 'vstart', 'lhs6', 'ratio', 'ratio2', 'verify',
          'f10opt', 'f10uto', 'f10base', 'f10base2', 'f10base3',
          'f10uto_d10', 'f10opt_d10', 'f10base_d10', 'f10base_d10b',
          'f10base_d10c', 'f10lit10', 'f10lit20', 'orrtest', 'f10uto3', 'f10dom', 'f10dom2',
          'f2s_a', 'f2s_b', 'f2s_c', 'f2sref_a', 'f2sref_b', 'f2sref_c', 'f2s_r',
          'f2scc_a', 'f2scc_b', 'f2scc_c', 'f2sm_a', 'f2sm_b', 'f2sm_c', 'f2sm_r',
          'f2su_a', 'f2su_b', 'f2su_c', 'f2su_r',
          'f2se_a', 'f2se_b', 'f2se_c', 'f2st', 'f2se_r',
          'j3u_a', 'j3u_b', 'j3u_c', 'j3c_a', 'j3c_b', 'j3c_c', 'j3d_a', 'j3d_b', 'j3d_c',
          'j2k_a', 'j2k_b', 'j2k_c', 'j2e_a', 'j2e_b', 'j2e_c',
          'j2e5_a', 'j2e5_b', 'j2e5_c', 'j2e7_a', 'j2e7_b', 'j2e7_c', 'j2c_a', 'j2c_b', 'j2c_c',
          'j2p_a', 'j2p_b', 'j2p_c', 'j2p_r', 'j2pB_a', 'j2pB_b', 'j2pB_c', 'j2pB_r',
          'a3cc_a', 'a3cc_b', 'a3cc_c', 'd600_a', 'd600_b', 'd600_c', 'r600_a', 'r600_b', 'r600_c', 'r600g_a', 'r600b_a', 'r600b_b', 'p4v', 'p4v2', 'perm')
# p4v2      = p4v 의 나머지 14농도 42조건 (1_extract/make_p4v2.py). I_dis1 비수렴 대역 2건은
#             1.025 mA 로 올렸다 -> _O_result/p4v2_perturbation.csv
# p4v       = prio4 선택점 COMSOL 검증 9조건 (500/600/700 mM: 스텝 1 + 탈염량 감싸는 정전류 2)
#             (1_extract/make_p4v.py)
# r600b_*   = r600g 판정 보강 분리 정전류 6 (342h·290h 후보 주변) (1_extract/make_r600b.py)
# r600g_a   = gE<=-5 % 제약 재최적화 후보 4 (충전 <= 76 mAh), 수정 러너 (1_extract/make_r600g.py)
# r600_*    = 600 mM 재최적화 후보 4 + 분리 정전류 격자 16, 방전 strict·종료시각 수정 러너 (1_extract/make_r600.py)
# d600_*    = 600 mM 단일 농도 진단 19조건: 충전·방전 분리 정전류 격자, j2e5 방전 변형, joint2r 전선 5점, ±0.01 mA 잡음 (1_extract/make_d600.py)
# j2pB_r    = j2pB cond 17 (700 mM) 3사이클 방전 I_dis3 1.048 -> 1.06 mA 재시도 (_O_result/j2pB_r_perturbation.csv)
# a3cc_*    = 세 지표 모두 개선 사후 후보 14농도의 탈염량 맞춘 정전류 I* 확인 (1_extract/make_a3cc.py)
# j2p_r     = j2p 비수렴 3조건(cond 1, 8, 9) 재시도. 1사이클 방전 I_dis1 = 1.009-1.018 mA 에서만 98.3 s 에
#             비수렴한다 (f2sm, lhs6, f10lit20 도 같은 대역에서 같은 시각 실패; 1.007 이하 · 1.023 이상은 정상).
#             같은 전류 재시도는 f2sm_r 에서 똑같이 실패했으므로 I_dis1 만 1.025 mA 로 올린다 (+0.7-1.6 %).
#             원래 값은 _O_result/j2p_r_perturbation.csv.
# j2p_*     = 생산성 · 비에너지 최적화, 점수 최대 + 비에너지 절감 >= 5 % (3_optimize/optimize_joint2p.py, 1_extract/make_j2p.py)
# j2e5/j2e7 = joint2r 전선에서 대리모델 순에너지 <= 정전류 -5 / -7 % 중 최단 시간 (1_extract/make_j2x.py)
# j2c_*     = 편차 보정 모델로 고른 에너지 이득 후보 (3_optimize/bias_joint2r.py)
# j2k_*    = 탈염량 고정 (joint3 선택점) 6전류 2목적 (시간, 순에너지) 이상점 최근접 해
#            (3_optimize/optimize_joint2r.py -> 1_extract/make_j2r.py)
# j2e_*    = 같은 전선에서 순에너지 <= 같은 탈염량 정전류 인 해 중 최단 시간
# j3u_*    = 6전류 동시 3목적 (시간, 탈염량, 순에너지) utopia, 전류비 제약 없음 (3_optimize/optimize_joint3.py)
# j3c_*    = 충전만: j3u 의 충전 + 방전은 탈염량 맞춘 정전류 I* (1_extract/make_j3.py)
# j3d_*    = 방전만: 충전은 I* + j3u 의 방전
# f2se_r   = f2se_b 의 587.5 M(cond_008) 재시도 (비수렴).
# f2se_*   = f2su 와 같은 충전 + 방전은 'E_net <= 정전류' 중 최단시간 (에너지 제약 후보)
# f2st     = f2su 와 같은 충전 + 방전은 utopia 보다 시간 쪽 후보 (대표 6농도)
#            (1_extract/make_f2s_candidates.py)
# f2su_*   = 순수 utopia (충전·방전 모두 이상점 최근접). 논문에 쓰는 해. (1_extract/make_f2su.py)
# f2sm_r   = f2sm_c 의 637.5 M(cond_012) 재시도 (1사이클 방전 부근에서 비수렴).
# f2sm_*   = 방전 전선에서 관행 정전류를 두 축 모두 이기는 여유(max-min)가 가장 큰 해.
#            (1_extract/make_f2sm.py)
# f2scc_*  = f2s 와 COMSOL 탈염량이 같아지는 관행 정전류 I* (6전류 모두 I*).
#            f10base 곡선 보간 불확실성을 없애려고 직접 돌린다. (1_extract/make_f2scc.py)
# f2s_r    = f2s_b 의 587.5 M(cond_008) 재시도. 2사이클 방전에서 시간 솔버가 수렴하지
#            못해 부분 결과만 남았다. 폴더를 따로 두어 부분 파일과 섞이지 않게 한다.
# f2s_*    = 2단계 최적화(충전 -> 방전) 최종 6전류 17개를 세 묶음으로. 짝 기준선은
#            관행 정전류 f10base/f10base2.  (1_extract/make_f2s.py)
# f2sref_* = 같은 충전 + 방전 = 충전. f2s 와의 차이 = 방전 분리의 몫.
if PROTOCOL not in _KNOWN:
    raise SystemExit(f'모르는 프로토콜: {PROTOCOL!r}  ({" | ".join(_KNOWN)})')

# mph 판본 꼬리표. 폴더 이름에 붙어 판본마다 데이터가 갈린다.
#   ''     -> _E_seawater_vlim        (260821 로 뽑은 옛 데이터)
#   'v49'  -> _E_seawater_vlim_v49    (260823 / revising49, DNa 7.0e-10 + Eeq +0.0516)
# ★ 이걸 비워 두고 돌리면 옛 폴더에 cond_001~200 이 이미 있어 전부 resume 스킵된다.
#   판본이 바뀌면 반드시 꼬리표도 바꿀 것 — 섞이면 물리가 다른 데이터가 한 폴더에 있게 된다.
DATA_TAG = 'v49'

COND_CSV = _os.path.join(_ROOT, f'cond_cycle_{PROTOCOL}_{WATER}.csv')
OUT_DIR = (_os.path.join(_ROOT, f'_E_{WATER}_{PROTOCOL}')
           + (f'_{DATA_TAG}' if DATA_TAG else '') + RUN_TAG)
ALL_NAME = (f'{WATER}_{PROTOCOL}{chr(95)+DATA_TAG if DATA_TAG else chr(32)*0}'
            f'{RUN_TAG}_all.csv')
# ===================================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(COND_CSV):
        raise SystemExit(f'조건표 없음: {COND_CSV}\n  make_conditions.py 를 먼저 실행할 것.')

    conds = pd.read_csv(COND_CSV)
    if N_TEST is not None:
        conds = conds.head(N_TEST)
    skips = C.load_skips(OUT_DIR)

    client, model, known = C.start(cores=CORES)
    if I0_ORR_OVERRIDE is not None:
        before = model.parameter('i0_orr')
        C.set_param(model, 'i0_orr', f'{I0_ORR_OVERRIDE}[A/m^2]', known)
        print('  [민감도] i0_orr %s -> %s A/m^2' % (before, I0_ORR_OVERRIDE),
              flush=True)
    plan = C.build_plan(N_CYCLES)
    CAL.verify(model, N_CYCLES)

    cap = CAL.per_cycle_mah(N_CYCLES)
    print(f'프로토콜 {PROTOCOL} | {len(conds)}조건 x {N_CYCLES}사이클')
    print(f'  충전 종료 : ' + ('4.2 V 도달' if PROTOCOL != 'cap'
                              else f'{cap:.2f} mAh 또는 {C.V_MAX} V 중 먼저'))
    print(f'  방전      : 충전 실측전하의 {C.DIS_FRAC*100:.0f}%')
    if PROTOCOL == 'vstart':
        print('  초기상태  : 조건마다 담수부 초기농도(c_C2_0)·음극 초기재고(nNa0) 가변')
    print(f'  타임아웃  : solve {C.SOLVE_TIMEOUT_MIN}분 / 조건 {C.COND_TIMEOUT_MIN}분')
    if skips:
        print(f'  [스킵목록] {sorted(skips)}')

    all_dfs, n_done = [], 0
    for _, row in conds.iterrows():
        cid = int(row['cond_id'])
        c_in = float(row['c_in[mol/m3]'])
        I0 = float(row['I[mA]'])
        subset = str(row.get('set', ''))

        # lhs6 = 사이클마다 충전·방전 전류가 다르다. 없으면 정전류로 되돌아간다.
        I_chg = [float(row.get(f'I_chg{k}', I0)) for k in range(1, N_CYCLES + 1)]
        I_dis = [float(row.get(f'I_dis{k}', I0)) for k in range(1, N_CYCLES + 1)]
        pick = lambda lst, cyc: lst[min(int(cyc), len(lst)) - 1]

        if cid in skips:
            continue
        cpath = os.path.join(OUT_DIR, f'cond_{cid:03d}.csv')
        if os.path.exists(cpath):
            print(f'[resume] cond_{cid:03d} 존재 -> 스킵')
            try:
                all_dfs.append(pd.read_csv(cpath))
                n_done += 1
            except Exception:
                pass
            continue

        print(f'\n===== 조건 {cid}/{len(conds)} [{subset}]: '
              f'c_in={c_in:.0f} mol/m^3, I={I0:.3f} mA =====')

        # --- 초기상태 ---------------------------------------------------------
        C.set_conc(model, c_in, known)                     # c_bulk_C1 = c_bulk_C2 = c_in
        if PROTOCOL == 'vstart':
            c2_0 = float(row['c_C2_0[mol/m3]'])
            nna0 = float(row['nNa0[mol/m2]'])
            # 담수부만 따로 낮춘다 — 유입 해수는 c_in 그대로
            C.set_param(model, 'c_bulk_C2', f'{c2_0}[mol/m^3]', known)
            C.set_param(model, 'nNa0', f'{nna0}[mol/m^2]', known)
            print(f'  초기상태: 담수부 {c2_0:.0f} mol/m^3, nNa0 {nna0:.3f} mol/m^2')

        # --- 충전 콜백 --------------------------------------------------------
        def charge_run(model, st, cyc, budget, I0=I0, I_chg=I_chg):
            I0 = pick(I_chg, cyc)
            C.set_its(model, f'{I0}[mA]')
            if PROTOCOL == 'cap':
                t_end = min(cap / I0 * 60.0, C.CHARGE_MAX_MIN)   # 목표 전하 도달 시간
                note = f'->{C.V_MAX}V / {cap:.1f}mAh@{t_end:.0f}분'
            else:
                # 종료는 4.2 V 뿐이다. tlist 상한은 '용량 제한'이 아니라 안전장치다.
                # 셀 용량은 율속 의존이라 저율속에서 70~80 mAh 까지 충전된다(제조사).
                # 실제로 0.635 mA 조건이 58.5 mAh 에서 4.2 V 에 닿았으므로
                # 정격 39.7 로 자르면 유효한 데이터가 잘린다.
                t_end = min(CAP_CEIL_MAH / I0 * 60.0, C.CHARGE_MAX_MIN)
                note = f'->{C.V_MAX}V (tlist 상한 {CAP_CEIL_MAH:.0f}mAh@{t_end:.0f}분)'
            # 출력 격자: 기본 5분이지만 긴 충전에서 점이 폭주하지 않도록 상한을 둔다.
            # (0.5 mA·80 mAh 면 9600분 = 5분격자로 1921점. 저장·평가 비용이 커진다)
            step = max(C.TIME_STEP_MIN, round(t_end / MAX_OUT_PTS, 1))
            C.set_tlist(model, st, round(t_end, 1), step=step)
            print(f'  cyc{cyc} 충전 {I0:.3f}mA ({note}) solving...')
            sol = C.run_phase(model, st, f'comp1.Vcell>={C.V_MAX}[V]', timeout_min=budget)
            return sol, (lambda t: I0), f'정전류 {I0:.3f}mA'

        dfs, ok, timed_out = C.run_condition(model, known, cid, c_in, plan,
                                             charge_run,
                                             dis_current=lambda cyc: pick(I_dis, cyc))
        # 세션이 죽었는지 확인한다. COMSOL 서버가 죽으면 run_condition 은 조용히
        # ok=False 로 돌아오고, 다음 조건의 set_conc 에서 JVM 없음으로 스택이 터진다.
        # 여기서 잡아 통합본을 저장하고 깔끔히 끝낸다(재실행하면 resume).
        if not ok:
            try:
                model.parameter('i0')                      # 값싼 왕복 — 살아있으면 통과
            except Exception:
                CAL.flush_all(all_dfs, OUT_DIR, ALL_NAME, n_done, force=True)
                raise SystemExit(
                    NEWLINE + '[중단] COMSOL 세션이 종료되었습니다 '
                    '(메모리 부족 또는 solve 중 크래시).' + NEWLINE
                    + f'  여기까지 저장됨: {OUT_DIR}' + NEWLINE
                    + '  COMSOL GUI 를 모두 닫고 남은 comsolmphserver 를 종료한 뒤' + NEWLINE
                    + '  다시 실행하면 완료된 조건은 건너뛰고 이어집니다.')

        cdf = C.save_condition(dfs, ok, OUT_DIR, cid)
        if cdf is not None:
            all_dfs.append(cdf)
            n_done += 1
        CAL.flush_all(all_dfs, OUT_DIR, ALL_NAME, n_done)

        if timed_out:
            CAL.flush_all(all_dfs, OUT_DIR, ALL_NAME, n_done, force=True)
            C.skip_and_restart(OUT_DIR, cid, '비수렴 timeout')

    CAL.flush_all(all_dfs, OUT_DIR, ALL_NAME, n_done, force=True)
    client.clear()
    print(f'\n종료. {n_done}조건 저장 -> {OUT_DIR}')


if __name__ == '__main__':
    main()
