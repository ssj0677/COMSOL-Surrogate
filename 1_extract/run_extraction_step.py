# -*- coding: utf-8 -*-
"""다단(step) 전류 데이터 추출 — 방전 없음.

프로토콜 (문헌형)
    I_1 로 충전 → V_1 도달 → I_2 로 낮춤 → V_2 도달 → ... → 마지막 단은 4.2 V 에서 종료

왜 pass 를 반복해서 t=0 부터 다시 푸는가
    전류가 '전압에 의해' 정해지므로 전류 계획을 미리 알 수 없다(폐루프). 그래서
    pass k 에서는 이미 확정된 전환시각 t_1..t_{k-1} 을 시간조건으로 박아 넣고 t=0 부터
    다시 풀어 V_k 도달 시각 t_k 를 얻는다. 인과성 덕분에 앞 구간은 뒤 전류에 영향받지
    않으므로, 반복 수렴이 아니라 한 번에 하나씩 확정된다.

        pass1: Its = I1                              stop V_1   -> t_1
        pass2: Its = if(t<t1, I1, I2)                stop V_2   -> t_2
        pass3: Its = if(t<t1,I1, if(t<t2,I2, I3))    stop V_3   -> t_3
        ...
        마지막: 4.2 V 에서 종료

    비용은 단일 충전의 약 2~2.5배다(pass k 는 V_k 까지만 푸므로).

전환 시각을 '해가 끝난 시각'으로 쓰지 않는 이유
    COMSOL 은 정지조건을 솔버 내부 스텝에서 평가하고, 다음 스텝이 조건을 넘으면 '이전
    스텝'에서 끝낸다. 그래서 기록된 마지막 값은 목표보다 늘 조금 낮다. 전환 시각은
    시계열에서 목표 전압을 처음 넘는 지점을 보간해 찾는다.

실행:
    python _D_run_step.py
"""
import os
import sys

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_BASE = _ROOT
sys.path.insert(0, _os.path.join(_ROOT, '1_extract'))      # COMSOL 코어(_D_swb_core)
sys.path.insert(0, os.path.join(_BASE, '1_extract'))


import numpy as np
import pandas as pd
import comsol_setup as CAL

C = CAL.apply()

# 솔버 예산을 늘린다.
#   charge_run 이 per_pass = budget / 단수 로 쪼개 쓰는데, 코어의 기본값
#   SOLVE_TIMEOUT_MIN=25 이면 5단 조건은 pass 당 5분뿐이라 그대로 미수렴 스킵된다
#   (실제로 조건 1이 그렇게 날아갔다). 다단은 pass 를 여러 번 푸는 구조라
#   단일 충전보다 오래 걸리는 것이 정상이므로 예산을 따로 잡는다.
C.SOLVE_TIMEOUT_MIN = 75      # 5단이면 pass 당 15분
# 충전 종료 전압을 4.20 -> 4.15 로 낮춘다.
#   단일염 모델은 담수부가 마르면 전해질 전도도가 0 으로 가고, 그 직전 구간에서
#   solver 가 죽는다. 실측으로 확인한 결과 정전류 594위상 중 4.20 V 에 정확히
#   닿은 것은 0%, 최대전압 중앙은 4.177 V 였다. 정지조건을 4.15 에 걸면 그
#   악조건 구간에 들어가기 전에 정상 종료한다 — 도달률 0% -> 76%.
#   마지막 50 mV 는 전압만 오르고 전하는 거의 안 들어가므로 탈염량 손실은 미미하다.
C.V_MAX = 4.15
C.COND_TIMEOUT_MIN = 300      # 조건 하나 전체

# ===== 설정 =========================================================================
WATER = 'seawater'
COND_CSV = _os.path.join(_ROOT, f'cond_multistep_{WATER}.csv')
OUT_DIR = _os.path.join(_ROOT, f'_E_{WATER}_step_v49')
ALL_NAME = f'{WATER}_step_v49_all.csv'

N_TEST = 10                  # None = 전부
CORES = 8

MAX_CHARGE_MIN = 3600.0      # 충전 1회 시간 상한 [min] — 저전류에서 tlist 폭주 방지
STOP_MARGIN = 0.05           # 정지조건을 '목표+마진' 에 건다 (아래 설명)
STOP_TOL = 0.08              # 목표에 이만큼 못 미쳐도 최고점에서 전환 (top-out 대비)

# 시간격자: 앞은 성기게, 뒤는 촘촘하게.
#   충전 곡선은 뒤로 갈수록 가팔라지므로(knee) 앞을 성기게 두어도 정보 손실이 적고,
#   뒤가 촘촘해야 전환 시각 보간이 정확하다.
COARSE_MIN = 5.0
FINE_MIN = 0.5
FINE_FROM = 0.85             # tlim 의 이 비율 지점부터 촘촘하게
# 출력점 상한. 정상 추출(run_extraction.py) 은 MAX_OUT_PTS=600 으로 격자를 자동으로
# 성기게 만드는데 여기엔 그 장치가 없었다. 그 결과 t_lim 3600분 x (앞 5분 + 뒤 0.5분)
# = 약 3960점이 되어 solve 가 예산 안에 못 끝났다(10조건 중 9개 미수렴 스킵).
# COMSOL 은 tlist 의 모든 시각에서 해를 맞춰야 하므로 출력점 수가 곧 비용이다.
MAX_OUT_PTS = 800
# ===================================================================================


def read_schedule(row):
    """조건 행 -> (전류 리스트[mA], 전환전압 리스트[V]). 마지막 단은 V_MAX 로 끝난다."""
    n = int(row['n_seg'])
    I = [float(row[f'I_{k}']) for k in range(1, n + 1) if pd.notna(row.get(f'I_{k}'))]
    V = [float(row[f'V_{k}']) for k in range(1, n) if pd.notna(row.get(f'V_{k}'))]
    return I, V[:len(I) - 1]


def its_expr(I, tsw):
    """COMSOL Its 식. 단위는 밖에서 한 번만 곱해 안쪽은 무차원으로 둔다."""
    e = f'{I[len(tsw)]:.6g}'
    for k in range(len(tsw) - 1, -1, -1):
        e = f'if(t<{tsw[k]:.4f}[min], {I[k]:.6g}, {e})'
    return f'1[mA]*({e})'


def i_of_t(I, tsw):
    def f(t):
        for k, ts in enumerate(tsw):
            if t < ts:
                return I[k]
        return I[len(tsw)]
    return f


def set_tlist_2stage(model, st, t_end):
    """앞 COARSE_MIN, 뒤 FINE_MIN 의 2단 출력격자. 모델의 시간단위로 환산해 넣는다."""
    tt, _ = C.study_feats(model, st)
    feat = model.java.study(st).feature(tt)
    k = C.tunit_factor(feat)                      # 분 -> 모델 시간단위
    split = min(max(COARSE_MIN * 2, t_end * FINE_FROM), t_end)
    # 두 구간의 점수 합이 MAX_OUT_PTS 를 넘으면 각 격자를 비례해 늘린다.
    coarse, fine = COARSE_MIN, FINE_MIN
    n_pts = split / coarse + (t_end - split) / fine
    if n_pts > MAX_OUT_PTS:
        f = n_pts / MAX_OUT_PTS
        coarse, fine = coarse * f, fine * f
    expr = (f'range(0,{coarse*k:.10g},{split*k:.10g}),'
            f'range({split*k:.10g},{fine*k:.10g},{t_end*k:.10g})')
    feat.set('tlist', expr)
    return split, coarse, fine


def cap_time(I, tsw):
    """이 스케줄로 음극 정격에 도달하는 시간[min]. tlist 상한으로 쓴다."""
    f = i_of_t(I, tsw)
    q = t = 0.0
    while t < MAX_CHARGE_MIN:
        q += f(t) / 60.0                          # 1분당 mAh
        t += 1.0
        if q >= CAL.ANODE_CAP_MAH:
            return t
    return MAX_CHARGE_MIN


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(COND_CSV):
        raise SystemExit(f'조건표 없음: {COND_CSV}\n  make_conditions.py 를 먼저 실행할 것.')

    conds = pd.read_csv(COND_CSV)
    if N_TEST is not None:
        conds = conds.head(N_TEST)
    skips = C.load_skips(OUT_DIR)

    client, model, known = C.start(cores=CORES)
    # 충방전 1사이클. 순에너지(E_chg - E_dis) 를 보려면 방전이 있어야 한다.
    plan = C.build_plan(1)
    CAL.verify(model, 1)

    print(f'다단 전류 | {len(conds)}조건 | 충방전 1사이클 | 마지막 단 {C.V_MAX} V 종료')
    print(f'  정지조건은 목표+{STOP_MARGIN} V 에 건다 (해가 목표를 확실히 통과해야 보간이 정확)')
    if skips:
        print(f'  [스킵목록] {sorted(skips)}')

    all_dfs, n_done = [], 0
    for _, row in conds.iterrows():
        cid = int(row['cond_id'])
        c_in = float(row['c_in[mol/m3]'])
        I_sched, V_sched = read_schedule(row)
        subset = str(row.get('set', ''))

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

        ladder = ' -> '.join(f'{x:.2f}' for x in I_sched)
        switch = ' / '.join(f'{v:.3f}' for v in V_sched + [C.V_MAX])
        print(f'\n===== 조건 {cid}/{len(conds)} [{subset}]: c_in={c_in:.0f} | '
              f'I {ladder} mA | 전환 {switch} V =====')
        C.set_conc(model, c_in, known)

        def charge_run(model, st, cyc, budget, I=I_sched, V=V_sched):
            tsw = []                                   # 확정된 전환 시각 [min]
            sol = None
            per_pass = max(budget / max(len(I), 1), 3.0)

            for k in range(len(I)):
                target = V[k] if k < len(V) else C.V_MAX
                t_lim = min(cap_time(I, tsw), MAX_CHARGE_MIN)
                C.set_its(model, its_expr(I, tsw))
                split, cg, fn = set_tlist_2stage(model, st, round(t_lim, 1))
                npts = int(split / cg + (t_lim - split) / fn)
                print(f'    pass{k+1}/{len(I)}  I={I[k]:.2f}mA -> {target:.3f}V '
                      f'(tlim {t_lim:.0f}분, 격자 {cg:.1f}→{split:.0f}분→{fn:.1f}, '
                      f'{npts}점) solving...')

                # 목표+마진에서 정지 -> 궤적이 목표를 확실히 통과 -> 보간이 정확
                sol = C.run_phase(model, st,
                                  f'comp1.Vcell>={target + STOP_MARGIN:.4f}[V]',
                                  timeout_min=per_pass)

                if k == len(I) - 1:                    # 마지막 단은 전환 없이 끝
                    break

                d = C.eval_series(model, sol)
                t, v = d['t'].to_numpy(float), d['Vcell'].to_numpy(float)
                hit = np.where(v >= target)[0]
                if len(hit):
                    j = int(hit[0])
                    t_k = float(t[0]) if j == 0 else float(
                        np.interp(target, [v[j-1], v[j]], [t[j-1], t[j]]))
                    how = '교차보간'
                elif target - v.max() <= STOP_TOL:
                    # 모델이 목표 바로 아래에서 top out 하는 경우가 있다. 전류를 더 못
                    # 밀어 넣는 지점이 곧 전환점이므로 최고점에서 전환한다.
                    t_k = float(t[int(np.argmax(v))])
                    how = f'근접전환({(target-v.max())*1000:.0f}mV 부족)'
                else:
                    print(f'      [미도달] {target:.3f}V 에 못 닿음 (최대 {v.max():.3f}) '
                          f'-> 남은 단 생략')
                    break

                tsw.append(t_k)
                print(f'      전환 t={t_k:.1f}분  ({how}, V끝 {v[-1]:.3f})')

            return sol, i_of_t(I, tsw), f'{len(tsw)+1}단 {ladder}mA'

        # 방전 전류는 사다리 1단(공칭 전류)으로 고정한다. 정전류(vlim) 는 충방전에
        # 같은 전류를 쓰므로, 다단도 '공칭 전류로 방전' 으로 맞춰야 순에너지 비교가
        # 전류 자체의 차이에 오염되지 않는다.
        dfs, ok, timed_out = C.run_condition(model, known, cid, c_in, plan, charge_run,
                                             dis_current=lambda cyc, i=I_sched[0]: i)
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
