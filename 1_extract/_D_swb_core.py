# -*- coding: utf-8 -*-
"""SWB 충방전 COMSOL 자동화 공통 코어 (260808 mph / revising25.java 기준).

세 러너(_D_mph_const_v2 / _D_mph_cascade2_lhs / _D_mph_timestep_cycles_lhs)가 전부 이 모듈을
쓴다. 예전엔 러너마다 setup_phase/disabledphysics 복사본이 갈라져 있어서 같은 버그가 반복됐음.

── 모델 실측 사실 (inspect로 직접 확인, 추측 아님) ─────────────────────────────────────────
physics : tcd, br, ge, spf(Laminar Flow_C1).  ※ revising25에서 spf2(Laminar Flow_C2) 삭제됨.
대류    : ice1(C1,해수)  u_src=root.comp1.u2 → spf 유동장. 충전·방전 모두 항상 ON.
          ice2(C2,담수)  u_src=userdef, u=[0,v0,0] → 파라미터 v0 직접 사용.
          ice3(C3)       u=[0,0,0].
  → "방전에 담수부를 닫는다"는 spf를 끄는 게 아니라 v0=0 으로 만드는 것.
    방전에 spf(C1)까지 끄면 C1이 v0 등속류가 되어 이온이 쌓이고 비수렴 → 예전 실패 원인.
방전 study: auxiliary sweep(useparam=True)으로 i0=-i0, v0=0 이 이미 걸려 있음.
  → 전류 부호는 sweep이 뒤집으므로 파이썬은 i0 파라미터에 '양수' 크기만 넣는다.
study별 feature 태그가 제각각(std1~6은 cdi/time, std7~는 time1·cdi1·time2…) → 반드시 타입으로 탐색.
저장된 disabledphysics는 std2만 올바름. 나머지 방전(std4,6,8,10,12,14,16)은 아직 spf를 끄고 있어
  매 실행 시 prep_discharge()가 그 항목만 제거한다(배열 전체를 덮어쓰면 COMSOL이
  'Invalid property value'로 거부 — cdi는 'ge/ge2' 같은 서브피처를 못 받는 등 형식이 다름).
solve  : GUI 'Compute'와 같은 순서(createAutoSequences → runAll)로 실행해야 정상 속도가 나온다.
  study.run()만 부르면 낡은 solver 시퀀스로 돌아 수십 배 느려짐(1조건 1시간+ → 60초대).

── 단계별 이월 규칙 (사용자 확정) ──────────────────────────────────────────────────────────
C1(init1) : 항상 fresh (c_bulk_C1, phis=3.23)
C3(init2) : 직전 '단계' 이월. 첫 충전만 fresh(1000/10)
C2(init3) : 충전은 직전 '충전' 이월 / 방전은 항상 fresh(c_bulk_C2)
c_tank(ge2): 충전은 직전 '충전' 이월 / 첫 충전은 fresh. 방전은 ODE 자체가 disabled
nNa(ge1)  : 직전 '단계' 이월. 첫 충전만 nNa0
Eeq offset: 충전 0.8 / 방전 0.55
용량      : 충전 ≤ CAP_MAH(39.7mAh) / 방전 = 그 사이클 충전 실측전하 × DIS_FRAC(98%)
"""
import os
import subprocess
import sys
import threading
import time
import jpype
import mph
import numpy as np
import pandas as pd

# 콘솔이 cp949면 COMSOL 예외 메시지의 유니코드 문자에서 print가 죽어 에러가 잘림 → UTF-8 강제.
for _s in ('stdout', 'stderr'):
    _f = getattr(sys, _s, None)
    if _f is not None and getattr(_f, 'encoding', '') and _f.encoding.lower() != 'utf-8':
        try:
            _f.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# ===== 공통 설정 ====================================================================
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
MODEL_PATH = _os.environ.get('SWB_MPH', 'charging_discharging.mph')
CORES      = 12

CAP_MAH        = 39.7     # 충전 용량 상한 [mAh]
DIS_FRAC       = 0.98     # 방전 = 충전 실측전하의 98%
V_MAX          = 4.2      # 충전 종료 전압
CHG_OFF        = '0.8'    # 충전 OER/CER Eeq 오프셋
# 방전 Eeq 오프셋. 문헌의 방전 평탄부 2.8 V vs Na+/Na 에 맞춘 값이다.
#   (Kim et al., "the real voltage plateau of 2.8 V ... formed by the reverse of the
#    electrochemical reaction", 방전 중 ORR 발생을 보고)
# 260812 모델(dz 를 코인셀 실부피 2.893 cm^3 로 보정)에서 실측:
#     DIS_OFF 0.40 -> 2.827 V,  0.55 -> 2.675 V,  0.70 -> 2.525 V
#     평탄부 = -1.0071*DIS_OFF + 3.2296  (순수 평행이동)  ->  2.8 V 는 0.427
# 구 값 0.55 는 dz 보정 전 기준이라 지금은 125 mV 낮게 나온다.
# ※ 이 값은 '위상별 경험적 보정'이다. CHG_OFF 와 같은 성격이며, 방전 실측 곡선이 없어
#   평탄부 1점에만 맞춘 것이다. 논문에는 그렇게 명시할 것.
DIS_OFF        = '0.427'
CER_STD        = '1.23'   # CER 표준전위. Eeq(CER) = CER_STD + 2.71 − off
OER_STD        = '0.934'  # OER 표준전위. Eeq(OER) = OER_STD + 2.71 − off
# --- 방전 반응 귀속 (2026-08-03 해결) ------------------------------------------------
# [문제] 종전 모델은 방전 전류의 99.97 % 를 per2(CER 역방향 = 염소환원)가 날랐다.
#   문헌은 방전을 ORR 로 본다(plateau 2.8 V vs Na+/Na). 원인은 두 가지였다.
#     · i0_cer / i0_oer = 1.6e4  — ORR 이 경쟁이 안 됨
#     · per1/per2 모두 Species=[] — 어떤 화학종도 소비하지 않아, 전류는 흐르되 물질수지가 없음
#   당시 per2 를 끄면 전극 전위를 정할 것이 사라져 phil/phis 가 NaN 이 됐다.
#
# [해결] O2 를 화학종으로 추가해 per1 을 물질수지에 연결했다.
#     · cO2 (z=0, 중성이라 전기중성 방정식에 들어가지 않음 -> Na/Cl 구속 구조 불변)
#     · D(O2) = 2.1e-9 m^2/s, 초기·경계 0.25 mol/m^3 (공기포화 해수)
#     · per1 의 Vi0[cO2] = -1  (환원 방향에서 소비 = ORR, 산화 방향에서 생성 = OER)
#     · i0_oer 를 방전 실측에 맞춰 상향
#   이제 per2 를 꺼도 ORR 이 전류를 나르므로 발산하지 않는다.
#
# [충전은 그대로] i0_oer 를 올려도 충전에서는 OER 이 거의 안 흐른다(분담 ~1e-21).
#   과전압은 0.16~0.26 V 로 충분히 걸리는데도 그렇다 — 억제는 동역학이 아니라
#   WaterBasedReferenceState='OH' 의 활동도 항에서 온다:
#     알칼리 경로 4OH- -> O2 + 2H2O + 4e- 이고 중성 해수 pH~7 이면 a(OH-)~1e-7 M,
#     기준 1 M 대비 (1e-7)^4 = 1e-28 억제. 관측된 1e-23 배 억제와 자릿수가 맞는다.
#   산성 경로(2H2O -> O2 + 4H+)로 바꾸면 억제가 사라지지만 충전 실측과 맞지 않았다.
#   따라서 알칼리 경로를 채택한다 — 충전 시 염소발생이 지배한다는 문헌과도 일치한다.
DIS_DISABLE_CER = True    # 방전에서 per2(CER) 비활성 -> ORR 이 방전을 담당
# -----------------------------------------------------------------------------------
# ※ setup_inits() 가 phase 마다 Eeq_ref 를 위 값들로 '다시 쓴다'. 그러므로 러너에서
#   model.java...feature('per2').set('Eeq_ref', ...) 를 직접 불러도 run_condition() 안에서
#   덮여 무효가 된다. 오프셋/표준전위를 바꾸려면 반드시 C.CHG_OFF / C.CER_STD 를 바꿀 것.
#   (2026-08 발견: _H_ 오프셋 실험 41조건이 이 때문에 전부 off=0.8·CER=1.23 으로 돌았다.
#    증상은 '네 Eeq 군의 V(t=0) 이 소수점 4자리까지 동일' 이었다.)
TIME_STEP_MIN  = 5        # 출력 간격 [min]
CHARGE_MAX_MIN = 30000    # tlist 안전 상한 [min]
C3_FRESH       = ('1000', '10')   # 첫 충전 C3(init2) fresh (cNa, cCl)
NNA0_EXPR      = 'nNa0'           # 첫 충전 nNa(ge1) 초기값
PARAM_CONC     = ('c_bulk_C1', 'c_bulk_C2')

# --- 피팅 파라미터 -----------------------------------------------------------------
# _B_parameter_optimization.py 결과. 러너가 start() 에서 모델에 써넣는다.
#
# [2026-08-03 확정] 260812 mph 에 저장된 값을 그대로 채택한다. 베이지안 최적화를 돌리지
# 않은 값이다 — 돌려봤지만 결과가 시작점보다 나빴다.
#     저장값   alpha_na 0.5407 alpha_cer 0.3241 i0_na 1e-3  i0_cer 9.8e-3 -> RMSE  9.27 mV
#     BO 최적  alpha_na 0.35   alpha_cer 0.20   i0_na 0.153 i0_cer 1e-4   -> RMSE 24.86 mV
# BO 최적은 자유 4개 중 셋이 탐색 하한에 붙은 해였고, 4차원을 24회로 훑느라 좁은
# 최적점을 놓친 결과다(RMSE 표면이 24.9~421 mV 로 매우 거칠다).
# 채택 근거: dz 를 코인셀 카탈로그 실부피(2.893 cm^3)로 보정한 것만으로
#     충전 초기전압 3.3344 V (실측 3.3344, 일치)
#     피팅창 200~650분 RMSE 9.27 mV (셀 간 재현성 한계 ~20 mV 이하)
#     방전 평탄부 2.800 V (문헌 2.8, DIS_OFF 0.427 로 보정)
# 이 세 가지가 kinetics 재조정 없이 맞았다. 파라미터를 더 만지면 개체차를 맞추는 것이 된다.
FIT_PARAMS = {
    'alpha_na':  0.5407,
    'alpha_cer': 0.3241,
    'i0_na':     0.001,         # A/m^2
    'i0_cer':    0.0098,        # A/m^2
    'alpha_oer': 0.6648,        # 충전 전압에 영향 없어 식별 불가 — 고정
    'i0_oer':    6.02e-7,       # A/m^2, 물리값
}
PARAM_UNITS = {'alpha_na': '', 'alpha_cer': '', 'alpha_oer': '',
               'i0_na': 'A/m^2', 'i0_cer': 'A/m^2', 'i0_oer': 'A/m^2'}
APPLY_FIT_PARAMS = True   # False 면 mph 에 저장된 값을 그대로 쓴다

GE1 = ('ge', 'ge1')       # nNa Global ODE
GE2 = ('ge', 'ge2')       # c_tank(탱크 20mL) Global ODE
TANK_ODE = 'c_tankt + tcd.Its_ec1/(F_const*V_tank)'   # Faraday 직접식(차분식은 동결됨)
# mph 에 저장된 식은 유량 혼합식 'c_tankt - Q/V_tank*(c_out - c_tank)' 이다.
# 두 식은 물리가 다르다. 1.05 mA, 600 mol/m^3, 같은 시간에서 실측 비교:
#     유량식     끝 c_tank 176.7  (423 제거)  4.13 V @1785분   <- 전하가 허용하는 양의 8배
#     Faraday   끝 c_tank 525.1  ( 74.9 제거) 3.90 V @2500분   <- 전하 보존 정확
# 유량식은 Q=3.6 mL/min, V_tank=21.8 mL 로 시상수가 6분이라 탱크가 c_out 을 즉시 따라가고,
# 그 c_out 이 다시 탱크에서 공급되는 순환 고리가 폭주한다. 질량수지가 깨지므로 쓰지 않는다.
# 다만 GUI 에서 유량식으로 돌리면 실측 타이밍(4.2 V @1787분)이 맞는 것처럼 보인다 —
# 과잉 탈염이 만든 가짜 분극이다. 답은 맞고 이유가 틀린 경우이므로 속지 말 것.
# [2026-08-03 결정] False 로 둔다 — mph 의 유량식을 그대로 쓴다.
#   dz 를 코인셀 실부피(2.893 cm^3)로 고친 260812 모델에서 두 식을 비교한 결과:
#       유량식    V(0) 3.3344,  3.974 V @ 30.0 mAh,  기울기 21.3 mV/mAh
#       Faraday  V(0) 3.3344,  4.167 V @  190 mAh,  기울기  4.4 mV/mAh
#       실측      V(0) 3.3344,  4.200 V @ 29.8 mAh,  기울기 29.1 mV/mAh
#   Faraday 식은 190 mAh 가 필요해 쓸 수 없다. 두 식의 차이는 정확히 부피 하나로,
#   유량식의 Δc_tank 375.8 은 Faraday 를 V_tank≈3 mL 로 놓은 것과 같다
#   (1119 umol / 375.8 mol/m^3 = 2.98 mL).
#   해석: 전압은 담수부 '국소' 농도(~3 mL 규모)에 반응하고, 탈염량은 배치 전체(21.8 mL)
#   기준이다. 서로 다른 두 양이므로 c_tank 하나로 둘 다 감당할 수 없다.
#
#   따라서 역할을 나눈다.
#     전압/컷오프 : mph 의 유량식 (실측과 맞음)
#     탈염량      : c_tank 가 아니라 전하에서 계산  c_batch = c_in - 1.7116*Q[mAh]
#   c_tank 는 분극을 만드는 내부 변수일 뿐 탈염 지표로 보고하지 않는다.
#   이렇게 하면 유량식의 Δc_tank 가 전하 대비 7.3배인 것이 결론에 관여하지 않는다.
OVERRIDE_TANK_ODE = False

# --- 비수렴 스킵(타임아웃) ---
# COMSOL은 실행 중인 solve를 외부에서 중단할 방법이 없다(ModelUtil에 abort/stop 없음).
# 그래서 solve를 워커 스레드에서 돌리고 제한시간을 넘기면: 부분저장 → 스킵목록 기록 →
# 이 파이썬이 띄운 comsolmphserver만 종료 → 스크립트 자기재실행(resume) 으로 빠져나온다.
# 스킵된 조건은 _skipped.txt 에 남아 재실행 때 건너뛰므로 무한루프가 안 생긴다.
SOLVE_TIMEOUT_MIN = 25    # 한 번의 solve(충전 1pass 또는 방전) 최대 대기 [min]
COND_TIMEOUT_MIN  = 120   # 한 조건(사이클 전체) 최대 [min]
AUTO_RESTART      = True  # False면 타임아웃 시 재시작 없이 종료(수동 재실행 필요)
SKIP_FILE         = '_skipped.txt'


class SolveTimeout(Exception):
    """제한시간 내 solve가 안 끝남 = 비수렴으로 간주."""

EXPR = {   # 추출 컬럼 -> (COMSOL 식, 단위)
    't':      ('t',              'min'),
    'Vcell':  ('tcd.phis0_ec1',  'V'),
    'cNa_C2': ('cNa_C2',         'mol/m^3'),
    'cCl_C2': ('cCl_C2',         'mol/m^3'),
    'nNa':    ('comp1.nNa',      ''),
    'c_tank': ('comp1.c_tank',   'mol/m^3'),
}
# ===================================================================================


# ---------- COMSOL 서버 프로세스 추적(타임아웃 시 우리가 띄운 것만 종료) ----------
_OUR_SERVERS = []


def _server_pids():
    """현재 떠 있는 comsolmphserver.exe PID 목록."""
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='comsolmphserver.exe'\" "
             "| ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=60).stdout
        return {int(x) for x in out.split() if x.strip().isdigit()}
    except Exception:
        return set()


def _kill_our_servers():
    """이 스크립트가 띄운 comsolmphserver만 종료(사용자가 따로 켠 COMSOL은 건드리지 않음)."""
    for pid in _OUR_SERVERS:
        try:
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True, timeout=60)
        except Exception as e:
            print(f'[!] 서버 종료 실패(PID {pid}): {e}')


# ---------- 모델 열기 ----------
def start(cores=None, model_path=None):
    """COMSOL 시작 + 모델 로드 + 탱크 ODE 세팅. (client, model, known_params) 반환."""
    cores = CORES if cores is None else cores
    print(f'COMSOL 시작... (cores={cores})')
    before = _server_pids()
    client = mph.start(cores=cores)
    _OUR_SERVERS.extend(_server_pids() - before)      # 타임아웃 때 죽일 대상
    model = client.load(model_path or MODEL_PATH)
    # 탱크 ODE 를 덮어쓴다 — 무엇을 무엇으로 바꾸는지 반드시 찍는다.
    # (2026-08: 이걸 조용히 하고 있어서, mph 에 저장된 유량식으로 피팅한 _B_ 와
    #  Faraday 식으로 도는 추출이 서로 다른 물리를 푸는 것을 오래 못 알아챘다.
    #  두 식의 차이는 크다 — 같은 조건에서 끝 c_tank 가 176.7 vs 525.1 mol/m^3 이고,
    #  유량식은 전류가 허용하는 양의 8배를 제거해 전하 보존이 깨진다.)
    try:
        ge = model.java.component('comp1').physics(GE2[0]).feature(GE2[1])
        old = str(ge.getStringMatrix('equation')[0][0])
        if OVERRIDE_TANK_ODE:
            if old.strip() != TANK_ODE.strip():
                ge.setIndex('equation', TANK_ODE, 0, 0)
                print(f'[탱크 ODE] 덮어씀\n    이전: {old}\n    이후: {TANK_ODE}')
            else:
                print(f'[탱크 ODE] 이미 Faraday 직접식 — 변경 없음')
        else:
            print(f'[탱크 ODE] mph 저장값 그대로 사용: {old}')
    except Exception as e:
        print(f'[!] 탱크 ODE 설정 실패: {e}')
    known = set(model.parameters())
    if APPLY_FIT_PARAMS:
        apply_fit_params(model, known)
    return client, model, known


def apply_fit_params(model, known):
    """FIT_PARAMS 를 모델에 써넣는다. 러너마다 손으로 넣지 않도록 한 곳에 모은다.

    mph 에 저장된 값과 다르면 반드시 알린다 — 지금은 두 값이 같지만, mph 를 GUI 에서
    고치고 여기를 안 고치면 조용히 덮여서 '모델을 고쳤는데 결과가 그대로'가 된다.
    """
    done, diff = [], []
    for k, v in FIT_PARAMS.items():
        u = PARAM_UNITS.get(k, '')
        try:
            cur = float(str(model.parameter(k)).split('[')[0].strip())
            if abs(cur - v) > abs(v) * 1e-6:
                diff.append(f'{k}: mph {cur:g} -> 코드 {v:g}')
        except Exception:
            pass
        if set_param(model, k, f'{v}[{u}]' if u else f'{v}', known):
            done.append(f'{k}={v:g}')
        else:
            print(f'[!] 파라미터 {k} 가 모델에 없다 — 건너뛴다')
    print(f'[피팅값 적용] {", ".join(done)}')
    if diff:
        print('[!] mph 저장값과 다르다 — 코드 값으로 덮어썼다:')
        for d in diff:
            print(f'      {d}')
        print('    mph 를 고친 것이 의도라면 FIT_PARAMS 도 같이 고칠 것.')


def archive_outdir(out_dir, tag=None):
    """기존 결과 폴더를 '이름만 바꿔' 보관하고 빈 폴더를 새로 만든다.

    러너의 resume 는 cond_XXX.csv 가 있으면 무조건 건너뛴다. 파라미터를 바꾸고 재추출할
    때 기존 폴더를 그대로 두면 한 조건도 새로 풀지 않고 조용히 끝난다.
    다른 곳으로 옮길 수 없는 상황을 위해 이동이 아니라 같은 자리에서 rename 을 쓴다.
    _skipped.txt 도 함께 딸려가므로 옛 비수렴 기록이 새 실행을 막지 않는다.
    """
    import glob
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return None
    if not glob.glob(os.path.join(out_dir, 'cond_*.csv')):
        return None                      # 비어 있으면 보관할 것이 없다
    tag = tag or time.strftime('%Y%m%d_%H%M')
    dst = f'{out_dir}_old_{tag}'
    n = 1
    while os.path.exists(dst):
        dst = f'{out_dir}_old_{tag}_{n}'
        n += 1
    n_files = len(glob.glob(os.path.join(out_dir, 'cond_*.csv')))
    os.rename(out_dir, dst)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[보관] {os.path.basename(out_dir)} ({n_files}조건) '
          f'-> {os.path.basename(dst)}  ... 새 폴더로 재추출한다')
    return dst


# ---------- 스킵 목록 ----------
def load_skips(out_dir):
    """타임아웃으로 이미 포기한 조건 id 집합."""
    path = os.path.join(out_dir, SKIP_FILE)
    skips = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                tok = line.split('\t')[0].strip()
                if tok.isdigit():
                    skips.add(int(tok))
    return skips


def skip_and_restart(out_dir, cid, reason='timeout'):
    """타임아웃 조건을 영구 스킵으로 기록하고 COMSOL을 정리한 뒤 스크립트를 자기재실행.
    (실행 중인 solve를 중단할 방법이 없어서 프로세스를 갈아끼우는 것이 유일한 탈출구)"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, SKIP_FILE), 'a', encoding='utf-8') as f:
        f.write(f'{cid}\t{reason}\t{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    print(f'[스킵] 조건 {cid} → {SKIP_FILE} 기록 ({reason})')
    sys.stdout.flush()
    _kill_our_servers()
    if not AUTO_RESTART:
        print('[종료] AUTO_RESTART=False → 다시 실행하면 이어서 진행됩니다.')
        os._exit(1)
    print('[재시작] COMSOL 정리 후 스크립트 재실행 (완료된 조건은 자동 skip)...')
    sys.stdout.flush()
    time.sleep(3)
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------- study/sol 탐색 ----------
def study_feats(model, st):
    """(transient_tag, cdi_tag). study마다 태그가 time/time1/time2, cdi/cdi1... 로 달라서 타입으로 찾는다."""
    tt = ct = None
    for f in model.java.study(st).feature().tags():
        f = str(f)
        ft = str(model.java.study(st).feature(f).getType())
        if 'Transient' in ft:
            tt = f
        elif 'CurrentDistribution' in ft:
            ct = f
    return tt, ct


def sol_of_study(model, st):
    """study에 실제로 연결된 solver 시퀀스 태그. createAutoSequences가 새 태그를 만들 수도 있으므로
    sol1/sol3/... 을 가정하지 않고 매번 조회한다(withsol 이월이 엉뚱한 sol을 가리키면 조용히 오염됨).
    study 하나에 sol이 2개 달림(실제 시퀀스 + 빈 저장용) → feature가 있는 쪽을 고른다."""
    found = None
    for s in model.java.sol().tags():
        s = str(s)
        try:
            if str(model.java.sol(s).study()) != st:
                continue
            if len(list(model.java.sol(s).feature().tags())) > 0:
                found = s          # 여러 개면 마지막(=가장 최근 생성)
        except Exception:
            pass
    return found


# ---------- 단계 설정 ----------
def set_param(model, name, value, known):
    if name in known:
        model.parameter(name, value)
        return True
    return False


def set_conc(model, c_in, known):
    for p in PARAM_CONC:
        set_param(model, p, f'{c_in}[mol/m^3]', known)


def set_its(model, expr):
    """ec1 전류. 방전은 반드시 'i0' — study aux sweep(i0=-i0)이 부호를 뒤집는다."""
    model.java.component('comp1').physics('tcd').feature('ec1').set('Its', str(expr))


def set_tlist(model, st, minutes, step=None):
    """step=None 이면 호출 시점의 TIME_STEP_MIN 을 읽는다.
    기본인자로 두면 정의 시점 값(5)이 고정되어 러너에서 C.TIME_STEP_MIN 을 바꿔도
    반영되지 않는다 — 저전류(_G3_) 실행에서 출력이 5분 간격으로 나와 발견한 문제."""
    step = TIME_STEP_MIN if step is None else step
    tt, _ = study_feats(model, st)
    feat = model.java.study(st).feature(tt)
    k = tunit_factor(feat)                      # 분 -> 모델의 시간 단위
    feat.set('tlist', f'range(0,{step*k:.10g},{minutes*k:.10g})')


# 분 -> 모델 시간단위 환산. mph 마다 tunit 이 다르다.
#   260811  tunit = min      260812  tunit = h
# 이 코드는 tlist 를 '분'으로 쓰는데, tunit=h 인 모델에서는 그것이 시간으로 해석되어
# 60배가 된다. 실제로 260812 에서 range(0,5,670) 이 "670시간을 5시간 간격"이 되어
# 출력이 8점(0,5,...,35h)만 나왔고, 피팅창 200~650분에 시뮬 점이 2개뿐이라
# RMSE 가 곡선 일치도가 아니라 직선 보간값이 되어 있었다.
_TUNIT = {'s': 60.0, 'min': 1.0, 'h': 1.0 / 60.0, 'd': 1.0 / 1440.0}


def tunit_factor(study_time_feature):
    try:
        u = str(study_time_feature.getString('tunit')).strip()
    except Exception:
        return 1.0
    if u not in _TUNIT:
        print(f'[!] 모르는 시간 단위 {u!r} — 분으로 가정한다')
        return 1.0
    return _TUNIT[u]


def set_stop(model, sol, expr, idx=0):
    """정지조건. 반드시 createAutoSequences 뒤에 호출(시퀀스 재생성 시 덮일 수 있음)."""
    try:
        model.java.sol(sol).feature('t1').feature('st1').setIndex('stopcondarr', expr, idx)
    except Exception as e:
        print(f'    [!] stop({sol}) 실패: {e}')


def _init_node(model, node, cNa, cCl, phil, phis):
    f = model.java.component('comp1').physics('tcd').feature(node)
    f.setIndex('initc', cNa, 0)
    f.setIndex('initc', cCl, 1)
    f.set('initphil', phil)
    f.set('initphis', phis)


def _ge_init(model, path, expr):
    try:
        (model.java.component('comp1').physics(path[0]).feature(path[1])
              .setIndex('initialValueU', expr, 0, 0))
    except Exception as e:
        print(f'    [!] {path[1]} init({expr}) 실패: {e}')


def setup_inits(model, phase, prev_sol, prev_chg_sol):
    """C1/C3/C2 초기값 + nNa/c_tank withsol 이월 + Eeq 오프셋.
    prev_sol      : 직전 '단계'의 sol 태그 (C3, nNa 이월). None이면 첫 충전.
    prev_chg_sol  : 직전 '충전'의 sol 태그 (C2 담수, c_tank 이월). None이면 fresh.
    """
    W = lambda sol, var: f"withsol('{sol}', {var})"

    _init_node(model, 'init1', 'c_bulk_C1', 'c_bulk_C1', '0', '3.23')          # C1 항상 fresh

    if prev_sol is None:                                                        # 첫 충전
        _init_node(model, 'init2', C3_FRESH[0], C3_FRESH[1], '0', '0')          # C3 fresh
        _init_node(model, 'init3', 'c_bulk_C2', 'c_bulk_C2', '0', '0')          # C2 fresh
        _ge_init(model, GE1, NNA0_EXPR)                                         # nNa = nNa0
        _ge_init(model, GE2, 'c_bulk_C2')                                       # 탱크 fresh
    else:
        _init_node(model, 'init2', W(prev_sol, 'cNa'), W(prev_sol, 'cCl'), '0', '0')   # C3 이월
        _ge_init(model, GE1, W(prev_sol, 'nNa'))                                        # nNa 이월
        if phase == 'charge' and prev_chg_sol is not None:
            _init_node(model, 'init3', W(prev_chg_sol, 'cNa'), W(prev_chg_sol, 'cCl'), '0', '0')
            _ge_init(model, GE2, W(prev_chg_sol, 'c_tank'))                     # 탱크 이월(inflow)
        else:
            _init_node(model, 'init3', 'c_bulk_C2', 'c_bulk_C2', '0', '0')      # 방전=fresh 해수
            if phase == 'charge':
                _ge_init(model, GE2, 'c_bulk_C2')

    # 모듈 전역을 호출 시점에 읽는다 — 러너가 C.CHG_OFF / C.CER_STD 를 바꾸면 반영된다.
    off = CHG_OFF if phase == 'charge' else DIS_OFF
    tcd = model.java.component('comp1').physics('tcd')
    tcd.feature('pce1').feature('per1').set('Eeq_ref', f'({OER_STD}+2.71)-{off}')
    tcd.feature('pce1').feature('per2').set('Eeq_ref', f'({CER_STD}+2.71)-{off}')

    # per2 는 항상 켜 둔다. 아래 실패 기록을 반드시 읽을 것.
    tcd.feature('pce1').feature('per2').active(True)


# 방전 study의 정답 설정 = std2(파일에서 유일하게 정상 동작하는 방전) 실측값.
#   Transient: spf(C1 유동)는 켜둔 채 담수 순환/탱크/AEM만 끔. 담수부는 v0=0 sweep으로 닫힘.
#   cdi(초기화, Stationary): spf까지 끔. std4/6/8/10...은 여기서 spf가 켜져 있어 방전2가
#     "Repeated error test failures / singularity"로 죽었음 → std2와 동일하게 맞춘다.
#   ※ cdi에는 'ge/ge2' 같은 서브피처를 넣으면 COMSOL이 'Invalid property value'로 거부.
#     인터페이스 전체인 'ge'를 써야 한다(std2 실측값 그대로).
DIS_DISABLED_TIME = ['tcd/tel2', 'ge/ge2', 'tcd/in3', 'tcd/out2']
DIS_DISABLED_CDI  = ['tcd/tel2', 'ge', 'tcd/in3', 'tcd/out2', 'spf']


def prep_discharge(model, st, verbose=True):
    """방전 study를 std2(정상 동작 확인된 유일한 방전)와 동일한 설정으로 맞춘다.
    파일에 저장된 값이 study마다 제각각이라(특히 Transient에서 spf를 잘못 끔) 매번 강제."""
    tt, ct = study_feats(model, st)
    for feat_tag, want in ((tt, DIS_DISABLED_TIME), (ct, DIS_DISABLED_CDI)):
        if feat_tag is None:
            continue
        feat = model.java.study(st).feature(feat_tag)
        try:
            cur = [str(x) for x in feat.getStringArray('disabledphysics')]
        except Exception:
            cur = None
        if cur is not None and set(cur) == set(want):
            continue                      # 순서만 다른 건 같은 설정 → 불필요한 쓰기/로그 방지
        try:
            feat.set('disabledphysics', jpype.JArray(jpype.JString)(want))
            if verbose:
                print(f'    [보정] {st}/{feat_tag}: {cur} → {want}')
        except Exception as e:
            print(f'    [!] disabledphysics({st}/{feat_tag}) 설정 실패, 기존값 유지: {str(e)[:120]}')

    # 방전 auxiliary sweep: i0=-i0(전류 부호반전) + v0=0(담수부 폐쇄) 둘 다 있어야 한다.
    # 파일에 std14/std16(사이클 7·8)만 v0가 빠져 있어 그대로 두면 그 방전에서 담수부가
    # 닫히지 않는다(C2에 v0 유속이 계속 흘러 물리가 틀어짐) → 없으면 채워 넣는다.
    for feat_tag in (tt, ct):
        if feat_tag is None:
            continue
        try:
            feat = model.java.study(st).feature(feat_tag)
            if not feat.getBoolean('useparam'):
                if verbose:
                    print(f'    [경고] {st}/{feat_tag}: useparam=False → i0 부호반전/v0=0 sweep 없음')
                continue
            names = []
            for k in range(8):
                try:
                    names.append(str(feat.getString('pname', k)))
                except Exception:
                    break
            if 'v0' not in names:
                idx = len(names)
                feat.setIndex('pname', 'v0', idx)
                feat.setIndex('plistarr', '0', idx)
                feat.setIndex('punit', 'm/s', idx)
                if verbose:
                    print(f'    [보정] {st}/{feat_tag}: v0=0 sweep 추가(담수부 폐쇄)')
            if 'i0' not in names and verbose:
                print(f'    [경고] {st}/{feat_tag}: i0=-i0 sweep 없음 → 방전 전류 부호가 안 뒤집힘')
        except Exception as e:
            print(f'    [!] aux sweep({st}/{feat_tag}): {str(e)[:100]}')


def run_phase(model, st, stop_expr=None, timeout_min=None):
    """GUI 'Compute'와 동일: createAutoSequences → (정지조건) → runAll. 실제 sol 태그 반환.
    timeout_min을 넘기면 비수렴으로 보고 SolveTimeout 발생(스레드로 돌리고 기다리기만 중단.
    COMSOL 자체를 멈출 수는 없어서, 호출부가 skip_and_restart()로 프로세스를 갈아끼워야 함)."""
    model.java.study(st).createAutoSequences('all')
    sol = sol_of_study(model, st)
    if sol is None:
        raise RuntimeError(f'{st}: solver 시퀀스를 찾을 수 없음')
    if stop_expr:
        set_stop(model, sol, stop_expr, 0)

    timeout_min = SOLVE_TIMEOUT_MIN if timeout_min is None else timeout_min
    if not timeout_min or timeout_min <= 0:
        model.java.sol(sol).runAll()
        return sol

    box = {}

    def _work():
        try:
            model.java.sol(sol).runAll()
            box['ok'] = True
        except BaseException as e:                      # Java 예외 포함
            box['err'] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    th.join(timeout_min * 60.0)
    if th.is_alive():
        raise SolveTimeout(f'{st}/{sol}: {timeout_min:.0f}분 내 미수렴 (경과 {time.time()-t0:.0f}s)')
    if 'err' in box:
        raise box['err']
    return sol


# ---------- 결과 추출 ----------
def _eval(model, sol, expr, unit, dset='wf_dset'):
    res = model.java.result()
    try:
        ds = res.dataset(dset)
    except Exception:
        ds = res.dataset().create(dset, 'Solution')
    ds.set('solution', sol)
    tag = 'wf_eval'
    try:
        res.numerical().remove(tag)
    except Exception:
        pass
    ev = res.numerical().create(tag, 'EvalGlobal')
    ev.set('data', dset)
    ev.set('expr', expr)
    if unit:
        ev.set('unit', unit)
    data = np.array(ev.getReal(), dtype=float)
    res.numerical().remove(tag)
    return data.ravel()


def eval_series(model, sol, quiet_missing=('c_tank',)):
    """모든 EXPR 컬럼을 가진 DataFrame. 방전은 ge/ge2(탱크)가 disabled라 c_tank가 존재하지 않는데,
    이는 정상이므로 조용히 NaN으로 채운다(컬럼 구성이 phase마다 달라지면 concat이 어긋남)."""
    cols = {}
    missing = []
    for k, (e, u) in EXPR.items():
        try:
            cols[k] = _eval(model, sol, e, u)
        except Exception as ex:
            missing.append(k)
            if k not in quiet_missing:
                print(f'    [!] {k}({e}) 추출 실패: {ex}')
    if not cols:
        raise RuntimeError('결과 추출 전부 실패')
    n = min(len(v) for v in cols.values())
    df = pd.DataFrame({k: v[:n] for k, v in cols.items()})
    for k in missing:
        df[k] = np.nan
    return df[list(EXPR.keys())]


def trapz_mAh(I_series, t_min_series):
    """∫I dt [mAh] (I:mA, t:min)."""
    f = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(f(np.asarray(I_series, dtype=float), np.asarray(t_min_series, dtype=float)) / 60.0)


# ---------- 사이클 계획 ----------
def build_plan(n_cycles):
    """[(cycle, phase, study_tag)] — 충전1=std1, 방전1=std2, 충전2=std3 ... 순서."""
    plan = []
    k = 0
    for c in range(1, n_cycles + 1):
        for ph in ('charge', 'discharge'):
            k += 1
            plan.append((c, ph, f'std{k}'))
    return plan


def existing_studies(model):
    return set(str(t) for t in model.java.study().tags())


# ---------- 러너 공통 루프 ----------
def run_condition(model, known, cid, c_in, plan, charge_run, dis_current=None,
                  discharge_run=None, log=print):
    """한 조건(c_in)에 대해 plan 전체를 돌리고 (DataFrame 리스트, 정상완료여부, 타임아웃여부) 반환.

    charge_run(model, st, cyc, budget_min) -> (sol_tag, i_of_t, label)
        충전 phase를 실제로 solve하는 콜백. 러너마다 전류 프로파일이 달라서 위임한다.
        정전류/시간조각은 1회 solve, 캐스케이드는 내부에서 2-pass solve 후 마지막 sol을 반환.
        budget_min : 이 phase에 허용된 제한시간[min] — run_phase(timeout_min=)에 그대로 넘길 것.
        i_of_t : t[min] -> 인가전류[mA]  (전하적분/기록용)

    방전은 둘 중 하나를 준다:
      dis_current(cyc) -> 정전류 크기[mA] (양수).  간단한 경우용 기본 경로.
      discharge_run(model, st, cyc, prev_q, budget_min) -> (sol_tag, i_of_t, label)
          방전 중에도 전류를 바꾸고 싶을 때. prev_q = 직전 충전 전하[mAh].
          ※ 전류 부호: study aux sweep이 i0 → -i0 로 뒤집으므로, Its를 'i0*<배수식>' 형태로
            쓰면 부호가 자동으로 음수가 된다. i0 파라미터엔 '양수' 기준전류를 넣을 것.
    """
    dfs, t_off, prev_q = [], 0.0, None
    prev_sol = prev_chg_sol = None
    ok = True
    timed_out = False
    have = existing_studies(model)
    t_cond0 = time.time()

    for cyc, ph, st in plan:
        if st not in have:
            log(f'  cyc{cyc} {ph}: {st} 없음 → 스킵')
            if ph == 'charge':
                prev_q = None
            continue
        left_min = COND_TIMEOUT_MIN - (time.time() - t_cond0) / 60.0    # 조건 전체 예산 잔량
        if left_min <= 0:
            log(f'    [!] 조건 전체 제한({COND_TIMEOUT_MIN}분) 초과 → 나머지 사이클 스킵')
            ok = False
            break
        budget = min(SOLVE_TIMEOUT_MIN, left_min)
        try:
            setup_inits(model, ph, prev_sol, prev_chg_sol)

            if ph == 'charge':
                sol, i_of_t, label = charge_run(model, st, cyc, budget)
            else:
                if prev_q is None:
                    log(f'  cyc{cyc} 방전 스킵(직전 충전 없음)')
                    continue
                prep_discharge(model, st)
                if discharge_run is not None:              # 방전 중 전류 가변
                    sol, i_of_t, label = discharge_run(model, st, cyc, prev_q, budget)
                else:                                      # 방전 정전류(기본)
                    Idis = dis_current(cyc)
                    set_its(model, 'i0')                   # aux sweep이 -i0 로 뒤집음
                    set_param(model, 'i0', f'{Idis}[mA]', known)
                    dmin = max(round(DIS_FRAC * prev_q / Idis * 60.0, 1), TIME_STEP_MIN)
                    set_tlist(model, st, dmin)
                    log(f'  cyc{cyc} 방전 I={Idis:.3f}mA ({DIS_FRAC*100:.0f}% = {dmin:.0f}분) solving...')
                    i_of_t, label = (lambda t, v=Idis: -v), f'정전류 {Idis:.3f}mA'
                    sol = run_phase(model, st, timeout_min=budget)  # 정지조건은 파일 그대로(V<=2.0, nNa<=0)
        except SolveTimeout as e:
            log(f'    [!] {e} → 이 조건 포기(스킵 기록 후 재시작)')
            ok = False
            timed_out = True
            break
        except Exception as e:
            log(f'    [!] {st} 실패 → 이 조건 나머지 스킵:\n{e}')
            ok = False
            break

        df = eval_series(model, sol)
        t_end = float(df['t'].iloc[-1])
        df['t_cont'] = df['t'] + t_off
        df['cycle'] = cyc
        df['phase'] = ph
        df['cond_id'] = cid
        df['c_in'] = c_in

        if ph == 'charge':
            df['I_applied'] = df['t'].apply(i_of_t)
            prev_q = trapz_mAh(df['I_applied'].values, df['t'].values)
            log(f'    완료 t={t_end:.0f}분 [{label}] cNa {df["cNa_C2"].iloc[0]:.0f}→{df["cNa_C2"].iloc[-1]:.0f} '
                f'Q={prev_q:.2f}mAh V끝={df["Vcell"].iloc[-1]:.2f} tank={df["c_tank"].iloc[-1]:.0f}')
            prev_chg_sol = sol
        else:
            df['I_applied'] = df['t'].apply(i_of_t)        # 방전은 음수
            q_dis = -trapz_mAh(df['I_applied'].values, df['t'].values)
            log(f'    완료 t={t_end:.0f}분 [{label}] Q={q_dis:.2f}mAh '
                f'nNa끝={df["nNa"].iloc[-1]:.3f} V끝={df["Vcell"].iloc[-1]:.2f}')

        dfs.append(df)
        t_off += t_end
        prev_sol = sol

    return dfs, ok, timed_out


def save_condition(dfs, ok, out_dir, cid, log=print):
    if not dfs:
        return None
    cdf = pd.concat(dfs, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)     # 실행 도중 폴더가 지워져도 결과를 잃지 않도록 매번 보장
    path = os.path.join(out_dir, f'cond_{cid:03d}.csv')
    cdf.to_csv(path, index=False)
    log(f'  → 저장 cond_{cid:03d}.csv ({len(cdf)}행){"" if ok else " [부분]"}')
    return cdf
