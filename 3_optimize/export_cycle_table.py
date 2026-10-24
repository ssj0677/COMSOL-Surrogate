# -*- coding: utf-8 -*-
"""프로토콜별 · 사이클별 조건/전압/탈염량 표를 뽑는다 — 에너지를 직접 계산하려고.

    _O_result/cycle_table_surrogate.csv   대리모델 전선·기준선 (위상마다 한 줄)
    _O_result/cycle_table_comsol.csv      COMSOL 검증 집합 (위상마다 한 줄)

한 줄 = 한 조건의 한 위상(충전 또는 방전). 사이클 3개 x 2위상 = 조건당 6줄.

열
    source      surrogate | comsol
    protocol    전선/기준선 이름
    c_in        공급수 농도 [mol/m^3]
    cond_id     조건 번호 (대리모델은 파일 안의 해 순번)
    cycle       1~3
    phase       charge | discharge
    I_mA        그 위상의 전류 크기 [mA] (부호 없음)
    hours       그 위상의 길이 [h]
    Q_mAh       I_mA * hours
    V_start/V_end/V_mean   셀 전압 [V]
    E_mWh       적분 에너지 = sum(V * I * dt). 충전 +(넣은 것), 방전 -(뺀 것)
    c_mix_start/c_mix_end  혼합 농도 [mol/m^3]
    removed     그 위상에서의 농도 감소 = c_mix_start - c_mix_end
    removed_total  그 조건의 최종 탈염량 (마지막 충전 끝 기준, c_in - c_mix)

★ 탈염은 충전 중에만 일어난다
    방전은 새 해수에서 진행되므로 방전 구간의 농도 변화는 탈염 성능이 아니다.
    optimize_nsga2.rollout 이 탈염량을 '마지막 충전이 끝난 시점' 으로 재는 이유다.
    방전 줄의 removed 는 참고용이지 성능이 아니다.

★ COMSOL 쪽 방전 전하는 설계값 0.98 보다 조금 적다 (중앙 0.967, p5~p95 0.947~0.980)
    추출 러너가 방전 시간을 '충전 실측전하의 98%' 로 잡아 tlist 를 만드는데,
    솔버 출력이 그 시각 직전에서 끝난다 (로그: 98% = 514분 -> 완료 t=510분).
    1% 안팎의 차이고, 이 표는 그 실측을 그대로 적는다 — 0.98 로 되돌리지 않는다.
    Q_mAh = I_mA * hours 는 정확히 일치하도록 위상 길이를 t_cont 로 쟀다.

★ 회수율은 이렇게 낸다
    eta = -sum(E_mWh[방전]) / sum(E_mWh[충전])         (조건 단위로 묶어서)

    py 3_optimize/export_cycle_table.py [surrogate|comsol|all]
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
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '3_optimize'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = _ROOT
OUT = os.path.join(BASE, '_O_result')
BOM = chr(65279)
CHG = ['I_chg1', 'I_chg2', 'I_chg3']
DIS = ['I_dis1', 'I_dis2', 'I_dis3']

# 대리모델 쪽 — (파일, 표에 쓸 이름)
SURR = [
    ('pareto_front_chgonly_idis10_rmax224.csv', 'step_dis1.0'),
    ('baseline_fastdis_idis10_n60.csv',         'constant_dis1.0'),
    ('pareto_front_chgonly_rmax224.csv',        'step_dis1.5'),
    ('baseline_fastdis_n60.csv',                'constant_dis1.5'),
    ('pareto_front_mirror.csv',                 'step_dis=chg'),
    ('baseline_constant_n60.csv',               'constant_conventional'),
]
# COMSOL 쪽 — 폴더 꼬리표
COMSOL = ('f10uto', 'f10opt', 'f10base', 'f10base2',
          'f10uto_d10', 'f10opt_d10', 'f10base_d10', 'f10base_d10b',
          'f10base_d10c', 'f10lit10', 'f10lit20', 'f10uto3',
          'f2s_a', 'f2s_b', 'f2s_c', 'f2s_r', 'f2sref_a', 'f2sref_b', 'f2sref_c',
          'f2scc_a', 'f2scc_b', 'f2scc_c', 'f2sm_a', 'f2sm_b', 'f2sm_c', 'f2sm_r',
          'f2su_a', 'f2su_b', 'f2su_c', 'f2su_r',
          'f2se_a', 'f2se_b', 'f2se_c', 'f2st', 'f2se_r',
          'j3u_a', 'j3u_b', 'j3u_c', 'j3c_a', 'j3c_b', 'j3c_c', 'j3d_a', 'j3d_b', 'j3d_c',
          'j2k_a', 'j2k_b', 'j2k_c', 'j2e_a', 'j2e_b', 'j2e_c',
          'j2e5_a', 'j2e5_b', 'j2e5_c', 'j2e7_a', 'j2e7_b', 'j2e7_c', 'j2c_a', 'j2c_b', 'j2c_c',
          'j2p_a', 'j2p_b', 'j2p_c', 'j2p_r', 'j2pB_a', 'j2pB_b', 'j2pB_c', 'j2pB_r',
          'a3cc_a', 'a3cc_b', 'a3cc_c', 'd600_a', 'd600_b', 'd600_c', 'r600_a', 'r600_b', 'r600_c', 'r600g_a', 'r600b_a', 'r600b_b', 'p4v', 'p4v2', 'perm')


def rd(path):
    d = pd.read_csv(path, encoding='utf-8-sig')
    d.columns = [c.strip().lstrip(BOM) for c in d.columns]
    return d


# ─────────────────────────────────────────────── 대리모델
def surrogate_rows():
    import optimize_nsga2 as O
    m, scaler, v0 = O.get_surrogate()
    net = m.net
    rows, t0 = [], time.time()
    for fname, label in SURR:
        p = os.path.join(OUT, fname)
        if not os.path.exists(p):
            print('  [건너뜀] %s 없음' % fname)
            continue
        d = rd(p)
        n_ok = 0
        for sid, r in d.iterrows():
            if set(CHG).issubset(d.columns):
                x = np.empty(2 * O.N_CYCLES)
                x[0::2] = r[CHG].to_numpy(float)
                x[1::2] = r[DIS].to_numpy(float)
            else:                                   # 정전류 기준선
                x = np.empty(2 * O.N_CYCLES)
                x[0::2] = float(r['I'])
                x[1::2] = float(r['I_dis']) if 'I_dis' in d.columns else float(r['I'])
            c_in = float(r['c_in'])
            g = O.rollout(net, scaler, v0, x, c_in, full=True)
            if not g['ok']:
                continue
            n_ok += 1
            V, I, C, st = g['V'], g['I'], g['c_mix'], g['steps']
            i0 = 0
            for ph in range(2 * O.N_CYCLES):
                k = int(st[ph])
                sl = slice(i0, i0 + k)
                v, cur, cm = V[sl], I[sl], C[sl]
                rows.append(dict(
                    source='surrogate', protocol=label, c_in=c_in, cond_id=sid,
                    cycle=ph // 2 + 1,
                    phase='charge' if ph % 2 == 0 else 'discharge',
                    I_mA=abs(float(cur[0])), hours=k * O.DT_H,
                    Q_mAh=abs(float(cur[0])) * k * O.DT_H,
                    V_start=float(v[0]), V_end=float(v[-1]),
                    V_mean=float(v.mean()),
                    E_mWh=float(np.sum(v * cur) * O.DT_H),
                    c_mix_start=float(cm[0]), c_mix_end=float(cm[-1]),
                    removed=float(cm[0] - cm[-1]),
                    removed_total=float(g['removed'])))
                i0 += k
        print('  %-22s %5d해 -> %5d줄   (%.0fs)'
              % (label, n_ok, n_ok * 6, time.time() - t0))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────── COMSOL
def comsol_rows():
    rows = []
    for tag in COMSOL:
        fs = [p for p in glob.glob(os.path.join(BASE, '_E_seawater_%s_v49' % tag,
                                                '*.csv'))
              if 'all' in os.path.basename(p)]
        if not fs:
            print('  [건너뜀] %s 아직 없음' % tag)
            continue
        d = pd.read_csv(fs[0])
        for cid, g in d.groupby('cond_id'):
            # ★ 위상 길이는 t_cont(누적)로 잰다. 위상 안의 t 는 0 부터 다시
            #   세므로 (t.max - t.min) 은 마지막 출력점까지일 뿐 다음 위상이
            #   시작하는 시각이 아니다. 그 한 칸을 빼먹으면 짧은 위상(방전)에서
            #   손해가 커져 방전/충전 전하비가 설계값 0.98 아래로 내려간다.
            g = g.sort_values('t_cont')
            key = (g.cycle.astype(str) + '|' + g.phase.astype(str)).to_numpy()
            bnd = [0] + list(np.where(key[1:] != key[:-1])[0] + 1) + [len(g)]
            tc = g.t_cont.to_numpy(float) / 60.0
            last_chg_end = None
            per = []
            for a, b in zip(bnd[:-1], bnd[1:]):
                q = g.iloc[a:b]
                cy, ph = int(q.cycle.iloc[0]), str(q.phase.iloc[0])
                t_end = tc[b] if b < len(g) else (
                    tc[b - 1] + (tc[b - 1] - tc[b - 2] if b - 2 >= a else 0.0))
                tt = np.r_[tc[a:b], t_end]
                dt = np.diff(tt)                      # 합 = 위상의 실제 길이
                v = q.Vcell.to_numpy(float)
                cur = q.I_applied.to_numpy(float)
                cm = q.c_mix.to_numpy(float)
                per.append(dict(
                    source='comsol', protocol=tag, c_in=float(q.c_in.iloc[0]),
                    cond_id=int(cid), cycle=cy, phase=ph,
                    I_mA=float(np.abs(cur).mean()),
                    hours=float(t_end - tc[a]),
                    Q_mAh=float(np.sum(np.abs(cur) * dt)),
                    V_start=float(v[0]), V_end=float(v[-1]),
                    V_mean=float(v.mean()),
                    E_mWh=float(np.sum(v * cur * dt)),
                    c_mix_start=float(cm[0]), c_mix_end=float(cm[-1]),
                    removed=float(cm[0] - cm[-1])))
                if ph == 'charge':
                    last_chg_end = float(cm[-1])
            tot = (per[0]['c_in'] - last_chg_end) if last_chg_end is not None else np.nan
            for r in per:
                r['removed_total'] = tot
            rows += per
        print('  %-14s 조건 %3d -> %4d줄' % (tag, d.cond_id.nunique(), len(rows)))
    return pd.DataFrame(rows)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('surrogate', 'all'):
        print('[대리모델]')
        t = surrogate_rows()
        p = os.path.join(OUT, 'cycle_table_surrogate.csv')
        t.to_csv(p, index=False, encoding='utf-8-sig')
        print('  saved: %s  (%d줄)\n' % (p, len(t)))
        chk = t.groupby(['protocol', 'cond_id']).apply(
            lambda g: -g.loc[g.phase == 'discharge', 'E_mWh'].sum()
            / g.loc[g.phase == 'charge', 'E_mWh'].sum())
        print('  확인 — 표에서 다시 낸 eta 중앙값')
        print(chk.groupby('protocol').median().round(4).to_string())
    if what in ('comsol', 'all'):
        print('\n[COMSOL]')
        t = comsol_rows()
        if len(t):
            p = os.path.join(OUT, 'cycle_table_comsol.csv')
            t.to_csv(p, index=False, encoding='utf-8-sig')
            print('  saved: %s  (%d줄)' % (p, len(t)))


if __name__ == '__main__':
    main()
