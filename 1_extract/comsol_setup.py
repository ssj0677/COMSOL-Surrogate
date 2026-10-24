# -*- coding: utf-8 -*-
"""revising47 / 260821 mph 용 캘리브레이션 오버레이.

_D_swb_core.py(2026.08) 를 그대로 쓰되 '어느 모델을, 어떤 값으로' 만 여기서 정한다.
코어를 복사하지 않는 이유는, 복사본이 생기면 버그 수정이 한쪽에만 반영돼 결과가 갈리기 때문이다.

쓰는 쪽:
    import comsol_setup as CAL
    C = CAL.apply()                 # 코어 모듈(설정이 덮인 상태)을 돌려준다
    client, model, known = C.start(cores=8)
    CAL.verify(model, n_cycles)     # mph 의 실제 상태를 출력 — 가정과 다르면 여기서 보인다
"""
import json
import os
import sys
import threading
import time

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_BASE = _ROOT
sys.path.insert(0, _os.path.join(_ROOT, '1_extract'))      # COMSOL 코어(_D_swb_core)
sys.path.insert(0, os.path.join(_BASE, '1_extract'))


import pandas as pd
import _D_swb_core as C

# ===== 모델 =========================================================================
# 2026-08-08: 260821 -> 260823 (revising49).
#   실험 충전곡선(0.6 M, 1 mA, 29.8 h)에 맞춰 재보정한 판이다.
#   바뀐 것은 딱 두 가지 —
#     DNa            1.5e-9 -> 8.603e-10 m^2/s   (자유용액의 0.57배, 굴곡도 ~1.7)
#     per1/per2 Eeq  전위 오프셋 추가
#   i0/alpha 6개는 260821 값 그대로 둔다. 부분조합 실험(_B_/param_subsets.csv)에서
#   DNa 하나만 바꾼 경우(R2 0.9295)와 7개를 다 바꾼 경우(0.9297)가 사실상 같았고,
#   전자가 필요 오프셋도 절반(+51.6 vs +109.6 mV)이라 설명하기 쉽다.
#   ※ 260821 로 뽑은 기존 데이터(_E_seawater_*)와 섞으면 안 된다. 물리가 다르다.
MODEL_PATH = _os.environ.get('SWB_MPH', 'charging_discharging.mph')

# ===== Eeq 오프셋 ===================================================================
# ★파이썬은 Eeq_ref 를 절대 정하지 않는다 — mph(GUI) 값이 유일한 진실이다★
#
# 코어의 setup_inits() 는 phase 마다 per1/per2 의 Eeq_ref 를 C.CHG_OFF/C.OER_STD 등으로
# 다시 쓴다. 여기에 숫자를 박아두면 GUI 에서 오프셋을 바꿔도 파이썬이 되돌려 버린다
# (실제로 1.019/0.267 로 바꾼 것이 1.362 로 덮어써지고 있었다).
#
# 그래서 값을 맞추는 대신 '되돌리는' 방식으로 바꾼다:
#   start()  에서 per1/per2/per3 의 Eeq_ref 를 스냅샷
#   setup_inits() 뒤에 그 스냅샷을 그대로 복원
# 이러면 GUI 에서 무엇으로 바꾸든 파이썬을 고칠 필요가 없다.
KEEP_MPH_EEQ = True

# ===== 운전 규약 ====================================================================
V_MAX = 4.2                # 충전 종료 전압
DIS_FRAC = 0.98            # 방전 전하 = 그 사이클 충전 실측전하 × 이 값 (쿨롱효율 98%)
ANODE_CAP_MAH = 39.7       # 음극 정격 [mAh]

# ===== 탈염 지표 ====================================================================
# 담수부 챔버와 순환 탱크는 농도가 다르지만, 실제로 회수하는 물은 둘을 합친 것이다.
# 그래서 '섞은 뒤의 농도'를 따로 계산해 c_mix 컬럼으로 남긴다 — 이게 탈염 성능 지표다.
#     c_mix = (V_ch·cNa_C2 + V_tank·c_tank) / (V_ch + V_tank)
# ※ V_CHAMBER_ML 은 mph 의 dz_dom(담수부) 정의와 반드시 같아야 한다.
#   revising47 기준 dz_dom = 1.8e-6[m^3]/intop_desal(1) 이므로 1.8 mL.
V_CHAMBER_ML = 1.8
# 20.0 이 아니라 17.915 다. 20.0 은 '총 용액' 이지 탱크가 아니다.
#   물리 구성: 총 20.0 mL = 챔버 1.8 + 배관(호스 2개) 0.285 + 탱크 17.915
#   mph 의 V_tank 파라미터도 17.915e-6 이고, 데이터에서 역산한 c_mix 의 챔버
#   가중치도 1.8/19.715 = 0.0913 으로 이 값과 일치한다.
#   ※ 잘못된 값은 방전 구간 c_mix fallback 계산에만 쓰였다(COMSOL 이 c_mix 를
#     내놓지 않는 구간). 충전 구간 값은 COMSOL 이 직접 계산하므로 영향이 없다.
V_TANK_ML = 17.915

# 사이클 사이(방전 중)에도 펌프는 계속 돈다. 챔버 재평형 시상수는 V_ch/Q 이고
# 방전 시간이 그와 같은 자릿수이므로, 다음 충전은 '섞인 농도'에서 시작하는 것이 맞다.
#   False 로 두면 코어 기본 동작(직전 충전 끝의 챔버 농도를 그대로 이월)이 되는데,
#   그러면 챔버가 고갈된 채로 2사이클이 시작해 곧바로 4.2 V 라 충전이 안 일어난다.
# ※ 탱크 농도로 두면 안 된다 — 챔버의 고갈분이 사라져 소금이 창조된다.
MIX_BETWEEN_CYCLES = True

# c_tank(Global ODE 종속변수)의 단위. COMSOL Global Equations 는 기본이 무차원이라
# mol/m^3 인 양과 그냥 더하면 'inconsistent unit' 오류가 난다.
#   ODE 에 단위를 지정하지 않았으면 '[mol/m^3]'  (값은 이미 mol/m^3 스케일이다)
#   Units ▸ Dependent variable quantity 를 Concentration 으로 지정했으면 ''
C_TANK_UNIT = ''           # ge2 에 Concentration(mol/m^3) 지정 완료 -> 보정 불필요

# ===== 방전 반응 ====================================================================
# 방전은 ORR(per3)만 돌려야 한다. 코어의 prep_discharge() 가 study 의 disabledphysics 를
# 자기 목록으로 덮어쓰므로, 그 뒤에 per1/per2 를 다시 꺼 준다.
DIS_DISABLE_PER = ['tcd/pce1/per1', 'tcd/pce1/per2']

# ===== 피팅 파라미터 ================================================================
# 260822 mph 에는 보정값이 이미 들어가 있다. 그래서 여기서는 아무것도 덮어쓰지
# 않는다(APPLY_FIT_PARAMS=False). 아래 값들은 '지금 mph 에 무엇이 들어 있어야
# 하는가'를 기록해 두는 용도이며, 코어의 기동 검증이 mph 실제값과 대조해 준다.
#   ※ BO_JSON 은 260821 기준(구 판)이므로 가리키지 않는다. 다시 쓰면 이중 적용된다.
BO_JSON = ''      # 비워 둔다 — mph 저장값이 유일한 진실

# 260823 mph 에 들어 있어야 할 값 (i0/alpha 는 260821 과 동일)
FALLBACK_PARAMS = {
    'i0_oer': 6.81e-8, 'i0_cer': 0.00553, 'i0_na': 0.256,
    'alpha_oer': 0.4473, 'alpha_cer': 0.5030, 'alpha_na': 0.6287,
}

# ===== 기타 ========================================================================
APPLY_FIT_PARAMS = False   # ★mph 저장값을 진실로 둔다★ True 면 BO json 으로 덮어씀
# 방전 완결성 (260917 수정). 이 전의 추출 데이터는 방전이 계획(0.98 x 충전전하)보다 일찍 끊긴 경우가 많다
#   (COMSOL 658운전 중 525 에 전하비 < 0.97 위상; 1사이클 69 %).
#   원인: 방전 끝 nNa ≈ 0.1 이라 BDF 의 마지막 큰 스텝이 tlist 끝을 넘어 nNa<=0 정지조건이 걸리고,
#   storestopcondsol='no' 라 그 스텝 안 출력이 버려진다. 1_extract/run_convergence.py L0 로 확인.
#   DIS_STRICT_STEPS : 방전 시간 스텝을 출력 시각에 맞춘다 -> tlist 끝을 넘지 않는다.
#   DIS_EXACT_END    : 방전 tlist 에 계획 종료 시각을 정확히 넣는다 (5 분 격자면 마지막 출력이 최대 5 분 모자라
#                      전하비가 0.977 까지 내려갔다).
DIS_STRICT_STEPS = True
DIS_EXACT_END = True
FLUSH_EVERY = 20           # 조건 이만큼마다 통합 csv 를 갱신 (중간에 봐야 하므로)
_applied = False
EEQ = {}                   # start() 에서 채워지는 mph 의 Eeq_ref 스냅샷


def _load_params():
    if os.path.exists(BO_JSON):
        try:
            with open(BO_JSON, encoding='utf-8') as f:
                d = json.load(f)
            p = d.get('params', {})
            if p:
                print(f'[cal47] BO 결과 사용 (RMSE {d.get("rmse", float("nan")):.5f} V)'
                      f'  {os.path.basename(BO_JSON)}')
                return p, 'BO'
        except Exception as e:
            print(f'[cal47] BO json 읽기 실패 → mph 저장값 사용: {e}')
    print('[cal47] BO 결과 없음 → mph 저장값(FALLBACK_PARAMS) 사용')
    return dict(FALLBACK_PARAMS), 'mph'


def apply():
    """코어 모듈에 이 파일의 설정을 덮어쓰고 돌려준다. 여러 번 불러도 안전하다."""
    global _applied
    if _applied:
        return C

    C.MODEL_PATH = MODEL_PATH
    C.V_MAX, C.DIS_FRAC = V_MAX, DIS_FRAC
    C.CAP_MAH = ANODE_CAP_MAH

    params, src = _load_params()
    C.FIT_PARAMS = dict(params)
    # mph 에 이미 피팅값을 써넣었으면 파이썬이 건드리지 않는다(Eeq 와 같은 원칙).
    C.APPLY_FIT_PARAMS = APPLY_FIT_PARAMS

    # 추출 컬럼의 Vcell 을 BO 가 맞춘 식과 같게 둔다.
    # (코어 기본값은 tcd.phis0_ec1 이고 BO 는 comp1.Vcell 로 맞췄다. 두 스케일이 다르면
    #  '맞춘 모델'과 '뽑은 데이터'의 전압 기준이 어긋난다.)
    C.EXPR = dict(C.EXPR)
    C.EXPR['Vcell'] = ('comp1.Vcell', 'V')
    # 혼합 농도 — mph 에 변수로 정의돼 있으면 그대로 뽑는다.
    # (방전 구간은 ge2 가 비활성이라 평가되지 않는다 -> 아래 eval_series 가 채운다)
    C.EXPR['c_mix'] = ('comp1.c_mix', 'mol/m^3')

    # 방전에서 per1/per2 를 다시 끄도록 prep_discharge 를 감싼다
    _orig = C.prep_discharge

    def prep_discharge(model, st, verbose=True):
        _orig(model, st, verbose)
        if not DIS_DISABLE_PER:
            return
        try:
            # ★ 태그를 'time' 으로 박으면 안 된다.
            #   study 마다 transient feature 태그가 time / time1 / time2 로 다르다
            #   (코어 study_feats 주석 참조). std1~std6 은 우연히 'time' 이었는데
            #   std8(4사이클 방전)만 달라서 조회가 실패했고, per1/per2 가 켜진 채로
            #   방전이 돌아 "Failed to find consistent initial values" 로 죽었다.
            #   그 결과 80조건 전부가 (충전 4, 방전 3) 으로 잘렸다. 타입으로 찾는다.
            tt = C.study_feats(model, st)[0]
            if not tt:
                raise RuntimeError(f'{st}: transient feature 를 못 찾았다')
            f = model.java.study(st).feature(tt)
            cur = list(f.getStringArray('disabledphysics'))
            add = [p for p in DIS_DISABLE_PER if p not in cur]
            if add:
                f.set('disabledphysics', cur + add)
                if verbose:
                    print(f'    [방전] 추가 비활성: {add}')
        except Exception as e:
            print(f'    [!] 방전 per 비활성 실패: {e}')

    C.prep_discharge = prep_discharge

    # --- Eeq_ref 를 mph 값으로 고정 -------------------------------------------
    # start() 에서 스냅샷 -> setup_inits() 뒤에 복원. 파이썬이 오프셋을 정하지 않는다.
    _orig_start = C.start

    def start(cores=None, model_path=None):
        client, model, known = _orig_start(cores, model_path)
        if KEEP_MPH_EEQ:
            tcd = model.java.component('comp1').physics('tcd')
            for node in ('per1', 'per2', 'per3'):
                try:
                    EEQ[node] = str(tcd.feature('pce1').feature(node).getString('Eeq_ref'))
                except Exception:
                    pass
            print(f'[cal47] Eeq_ref 스냅샷 (이 값으로 고정): {EEQ}')
        return client, model, known

    C.start = start

    _orig_inits = C.setup_inits

    def setup_inits(model, phase, prev_sol, prev_chg_sol):
        _orig_inits(model, phase, prev_sol, prev_chg_sol)      # 여기서 Eeq 가 덮인다
        if KEEP_MPH_EEQ and EEQ:
            tcd = model.java.component('comp1').physics('tcd')
            for node, expr in EEQ.items():                     # 곧바로 되돌린다
                try:
                    tcd.feature('pce1').feature(node).set('Eeq_ref', expr)
                except Exception:
                    pass

        # 사이클 사이 재평형 — 다음 충전은 챔버·탱크가 섞인 농도에서 시작한다
        if MIX_BETWEEN_CYCLES and phase == 'charge' and prev_chg_sol:
            vc, vt = V_CHAMBER_ML, V_TANK_ML
            s = prev_chg_sol

            tank = f"withsol('{s}',c_tank)"
            if C_TANK_UNIT:                       # 무차원 ODE 변수에 단위를 붙여준다
                tank = f"({tank}*1{C_TANK_UNIT})"

            def mix(field, _t=tank):
                return (f"({vc}[ml]*withsol('{s}',{field})+{vt}[ml]*{_t})"
                        f"/{vc + vt}[ml]")

            try:
                C._init_node(model, 'init3', mix('cNa_C2'), mix('cCl_C2'), '0', '0')
                # ge2 는 무차원 변수이므로 mol/m^3 을 나눠 벗긴다
                ge_expr = mix('cNa_C2')
                if C_TANK_UNIT:
                    ge_expr = f'({ge_expr})/1{C_TANK_UNIT}'
                C._ge_init(model, C.GE2, ge_expr)
            except Exception as e:
                print(f'    [!] 사이클 간 혼합 초기값 설정 실패: {e}')

    C.setup_inits = setup_inits

    # --- 결과에 c_mix(챔버+탱크 혼합 농도) 추가 --------------------------------
    _orig_eval = C.eval_series

    def eval_series(model, sol, quiet_missing=('c_tank', 'c_mix')):
        df = _orig_eval(model, sol, quiet_missing)
        if 'cNa_C2' not in df.columns:
            return df
        vc, vt = V_CHAMBER_ML, V_TANK_ML
        # 방전 구간은 ge2 가 비활성이라 c_tank / c_mix 가 없다. 탱크는 방전 중 보존되므로
        # 직전 값을 이어 쓰고, 그것으로 c_mix 를 채운다(모든 행이 값을 갖게).
        tank = (df['c_tank'] if 'c_tank' in df.columns else df['cNa_C2'])
        tank = tank.ffill().bfill().fillna(df['cNa_C2'])
        calc = (vc * df['cNa_C2'] + vt * tank) / (vc + vt)
        df['c_mix'] = df['c_mix'].fillna(calc) if 'c_mix' in df.columns else calc
        return df

    C.eval_series = eval_series

    # --- 방전 완결성: 계획 종료 시각을 출력에 넣고, 시간 스텝이 그 시각을 넘지 않게 한다 ------------
    def _is_discharge(st):
        try:
            return int(str(st)[3:]) % 2 == 0          # plan: std1 충전, std2 방전, ...
        except ValueError:
            return False

    _orig_tlist = C.set_tlist

    def set_tlist(model, st, minutes, step=None):
        _orig_tlist(model, st, minutes, step)
        if not (DIS_EXACT_END and _is_discharge(st)):
            return
        step = C.TIME_STEP_MIN if step is None else step
        if abs(minutes / step - round(minutes / step)) < 1e-9:
            return                                     # 이미 격자 위
        tt, _ = C.study_feats(model, st)
        feat = model.java.study(st).feature(tt)
        k = C.tunit_factor(feat)
        feat.set('tlist', f'range(0,{step*k:.10g},{minutes*k:.10g}) {minutes*k:.10g}')

    C.set_tlist = set_tlist

    def run_phase(model, st, stop_expr=None, timeout_min=None):
        """코어 run_phase 와 같되, 방전이면 시간 스텝을 'strict' 로 둔다 (시퀀스 생성 뒤에 걸어야 남는다)."""
        model.java.study(st).createAutoSequences('all')
        sol = C.sol_of_study(model, st)
        if sol is None:
            raise RuntimeError(f'{st}: solver 시퀀스를 찾을 수 없음')
        if stop_expr:
            C.set_stop(model, sol, stop_expr, 0)
        if DIS_STRICT_STEPS and _is_discharge(st):
            model.java.sol(sol).feature('t1').set('tstepsbdf', 'strict')
        timeout_min = C.SOLVE_TIMEOUT_MIN if timeout_min is None else timeout_min
        if not timeout_min or timeout_min <= 0:
            model.java.sol(sol).runAll()
            return sol
        box = {}

        def _work():
            try:
                model.java.sol(sol).runAll()
                box['ok'] = True
            except BaseException as e:
                box['err'] = e

        th = threading.Thread(target=_work, daemon=True)
        t0 = time.time()
        th.start()
        th.join(timeout_min * 60.0)
        if th.is_alive():
            raise C.SolveTimeout(f'{st}/{sol}: {timeout_min:.0f}분 내 미수렴 (경과 {time.time()-t0:.0f}s)')
        if 'err' in box:
            raise box['err']
        return sol

    C.run_phase = run_phase
    print(f'[cal47] 방전 시간 스텝 strict={DIS_STRICT_STEPS}, 종료 시각 출력={DIS_EXACT_END}')

    print(f'[cal47] model={os.path.basename(MODEL_PATH)}  V_MAX={V_MAX}  '
          f'DIS_FRAC={DIS_FRAC}  파라미터={src}')
    print(f'[cal47] Eeq_ref = mph 저장값 고정 (파이썬이 정하지 않음)')
    print(f'[cal47] c_mix = ({V_CHAMBER_ML}·cNa_C2 + {V_TANK_ML}·c_tank)/'
          f'{V_CHAMBER_ML + V_TANK_ML}  [mol/m^3]')
    _applied = True
    return C


# ---------------------------------------------------------------------------------
def per_cycle_mah(n_cycles):
    """사이클당 충전 목표 [mAh] = 음극 정격 / 사이클 수. 조건과 무관한 고정값이다.

    근거: Na 도금/탈리 시험은 사이클당 면적용량을 고정하는 것이 표준 프로토콜이고,
    부분 심도 사이클링은 전용량 사이클이 수명을 해칠 때 흔히 쓰인다.
    여기서는 '한 배치 안에서 n 사이클을 완주하기 위해' 정격을 n 등분한다.
    """
    return float(ANODE_CAP_MAH) / int(n_cycles)


def verify(model, n_cycles=None):
    """mph 의 실제 상태를 출력한다. 스크립트의 가정과 다르면 solve 전에 보인다."""
    print('\n--- mph 실제 상태 ---')
    tcd = model.java.component('comp1').physics('tcd')
    for node in ('per1', 'per2', 'per3'):
        try:
            print(f'  Eeq_ref {node} = {tcd.feature("pce1").feature(node).getString("Eeq_ref")}')
        except Exception:
            print(f'  Eeq_ref {node} = (조회 실패)')
    for k in ('i0', 'c_bulk_C1', 'c_bulk_C2', 'v0', 'Q', 'V_tank', 'nNa0'):
        try:
            print(f'  param {k:11s} = {model.parameter(k)}')
        except Exception:
            print(f'  param {k:11s} = (없음)')

    if n_cycles:
        have = C.existing_studies(model)
        need = {st for _, _, st in C.build_plan(n_cycles)}
        miss = sorted(need - have, key=lambda s: int(s[3:]))
        print(f'  study 필요 {len(need)}개, 없는 것: {miss if miss else "없음"}')
    print('---------------------\n')


def flush_all(dfs, out_dir, name, n_done, force=False):
    """FLUSH_EVERY 마다(또는 force) 통합 csv 를 갱신한다.
    중간에 프로세스가 죽어도 여기까지는 남는다."""
    if not dfs:
        return
    if not force and n_done % FLUSH_EVERY:
        return
    path = os.path.join(out_dir, name)
    pd.concat(dfs, ignore_index=True).to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [통합저장] {name}  ({n_done}조건)')
