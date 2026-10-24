# -*- coding: utf-8 -*-
"""파레토 전선과 '농도에 따른 최적 전류'를 그린다 (NSGA-II 재실행 없이).

optimize_nsga2.py 가 남긴 CSV 두 개만 읽는다.
    _O_result/pareto_front.csv        NSGA-II 가 찾은 해 (c_in 마다 여러 개)
    _O_result/baseline_constant.csv   같은 조건의 정전류 기준선

농도를 촘촘히(12.5 mol/m^3 간격) 훑었으므로 패널을 농도마다 두면 못 읽는다.
대신 색으로 농도를 나타내고, '농도가 바뀌면 최적 전류가 어떻게 바뀌는가'를
따로 곡선으로 뽑는다. 이쪽이 논문에서 말하고 싶은 것에 더 가깝다.

    그림 1 (pareto.png)
      (a) 파레토 전선 — 색 = 유입수 농도. 정전류 기준선은 회색 파선.
      (b) 같은 시간에서의 개선폭 vs 농도
      (c) 등탈염 기준 시간 단축 vs 농도

    그림 2 (pareto_currents.png)
      운전점(빠름/중간/많이)별로, 농도가 오를 때 최적 전류가 어떻게 움직이는지.
      사이클 1/2/3 의 충전 전류를 따로 그린다.

    py 3_optimize/plot_pareto.py
"""
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import plotstyle as _PS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

OUT_DIR = _os.path.join(_ROOT, '_O_result')
CHG = ['I_chg1', 'I_chg2', 'I_chg3']
TOL = 0.95        # 최대 탈염량의 이 비율 이상을 내는 해들을 '동등' 으로 본다
CBINS = [500.0, 550.0, 600.0, 650.0, 700.0]   # 농도를 50 간격으로 묶는다
# 260825: 그림 단위를 M(= mol/L)으로 통일한다. 1 M = 1000 mol/m^3.
#   내부 계산은 전부 mol/m^3 그대로 두고, 그리는 순간에만 곱한다.
MM = 1e-3
DIS = ['I_dis1', 'I_dis2', 'I_dis3']


def _style():
    _PS.apply()


def _interp_gain(f, b):
    """정전류 대비: (같은 시간에서 탈염 개선 %, 같은 탈염에서 시간 단축 %)."""
    f, b = f.sort_values('hours'), b.sort_values('hours')
    if len(f) < 2 or len(b) < 2:
        return np.nan, np.nan
    lo, hi = max(f.hours.min(), b.hours.min()), min(f.hours.max(), b.hours.max())
    g1 = np.nan
    if hi > lo:
        t = np.linspace(lo, hi, 40)
        gb = np.interp(t, b.hours, b.removed)
        gf = np.interp(t, f.hours, f.removed)
        g1 = float(np.median((gf - gb) / gb * 100))
    lo, hi = max(f.removed.min(), b.removed.min()), min(f.removed.max(), b.removed.max())
    g2 = np.nan
    if hi > lo:                       # removed 기준 보간은 단조 증가로 정렬해야 한다
        r = np.linspace(lo, hi, 40)
        hb = np.interp(r, b.sort_values('removed').removed, b.sort_values('removed').hours)
        hf = np.interp(r, f.sort_values('removed').removed, f.sort_values('removed').hours)
        g2 = float(np.median((hf - hb) / hb * 100))
    return g1, g2


def fig_pareto(front, base, levels):
    norm = colors.Normalize(min(levels) * MM, max(levels) * MM)
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')

    # 260825: 패널을 하나로 줄였다. 가운데(등시간 이득)·오른쪽(등탈염 시간단축)은
    # COMSOL 재검증판(verified_gain / gain_compare)이 같은 내용을 더 강하게 보여
    # 주므로 중복이다. 다만 g(농도별 이득표)는 main 이 표로 쓰므로 계산은 남긴다.
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for c_in in levels:
        f = front[front.c_in == c_in].sort_values('hours')
        b = base[base.c_in == c_in].sort_values('hours')
        c = sm.to_rgba(c_in * MM)
        if len(b):
            ax.plot(b.hours, b.removed * MM, '--', c=c, lw=.9, alpha=.45)
        if len(f):
            ax.plot(f.hours, f.removed * MM, '-', c=c, lw=1.8)
    ax.set_xlabel('Total operating time (h)')
    ax.set_ylabel('Total amount of salt removal (M)')
    ax.set_title('Pareto fronts')
    ax.grid(True, ls=':', alpha=.5)
    # 실선/파선이 무엇인지 밝힌다. 색은 농도를 나타내므로 범례는 회색 표본으로
    # 선 모양만 보여 준다 — 색까지 넣으면 특정 농도를 가리키는 것처럼 읽힌다.
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], c='0.35', ls='-', lw=1.8,
                              label='Step current (optimized)'),
                       Line2D([], [], c='0.35', ls='--', lw=1.2,
                              label='Constant current')],
              loc='lower right', fontsize=13, frameon=False)
    cb = fig.colorbar(sm, ax=ax, pad=.02)
    cb.set_label('Feed concentration (M)', fontsize=17, fontweight='bold')

    g = []
    for c_in in levels:
        a, b2 = _interp_gain(front[front.c_in == c_in], base[base.c_in == c_in])
        g.append(dict(c_in=c_in, gain=a, save=b2))
    g = pd.DataFrame(g)

    fig.tight_layout(w_pad=3.4)
    p = os.path.join(OUT_DIR, 'pareto.png')
    fig.savefig(p, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved:', p)
    return g


def fig_currents(front, levels):
    """농도가 오르면 최적 전류가 어떻게 움직이는가 — 'deep' 운전점만.

    운전점을 셋(빠름/중간/많이)으로 나눠 봤더니 '빠름'은 전류가 전부 상한(1.5)에
    붙어 노이즈만 보였다. 빨리 끝내려면 무조건 세게 미는 것이 답이라 당연하다.
    말이 되는 추세는 'deep'(가장 많이 빼는 해)에서만 나오므로 그것만 남긴다.
    """
    # ★ 두 가지를 바꾼다.
    #   ① 해 '하나' 가 아니라 '최적 근처 해들' 을 쓴다.
    #      탈염 최대 해 하나만 이으면 크게 흔들린다. 최적이 평평하기 때문이다 —
    #      탈염 상위 5해에서 탈염량은 0.17~2.51% 밖에 안 다른데 전류는 충전
    #      0.9~15.6%, 방전 1.4~32.6% 나 다르다. 즉 그 흔들림은 잡음이 아니라
    #      '동등한 해가 여럿' 이라는 사실이다. 최대 탈염량의 TOL 이상인 해를 쓴다.
    #   ② 농도를 50 간격으로 묶는다.
    #      원 농도 격자는 12.5 간격 17수준이라 점이 너무 촘촘하고, 농도 사이의
    #      미세한 차이는 NSGA-II 의 무작위성(농도당 60개체)이다. 500/550/600/
    #      650/700 다섯 지점으로 묶으면 지점당 해가 수십 개가 되어 안정된다.
    #      구간은 겹치지 않게 반열린 구간으로 자른다.
    rows = []
    for k, x in enumerate(CBINS):
        lo = x - 25.0 if k else x
        hi = x + 25.0
        last = (k == len(CBINS) - 1)
        sub = front[(front.c_in >= lo) &
                    ((front.c_in <= hi) if last else (front.c_in < hi))]
        if not len(sub):
            continue
        # 농도마다 그 농도의 최대 탈염량 기준으로 걸러 모은다
        keep = [g[g.removed >= g.removed.max() * TOL]
                for _, g in sub.groupby('c_in')]
        g = pd.concat(keep, ignore_index=True)
        r = {'c_in': x, 'n': len(g), 'n_lv': sub.c_in.nunique()}
        for kk in CHG + DIS:
            r[kk] = g[kk].median()
            r[kk + '_lo'] = g[kk].quantile(.25)
            r[kk + '_hi'] = g[kk].quantile(.75)
        rows.append(r)
    d = pd.DataFrame(rows).sort_values('c_in')
    print('농도 구간별 표본')
    for _, r in d.iterrows():
        print(f'  c_in {r.c_in:.0f}   농도 {int(r.n_lv)}수준, 해 {int(r.n)}개')

    # 충전·방전을 한 축에 함께 그린다. 값 범위가 겹치지 않아(충전 0.50~0.81,
    # 방전 1.32~1.49) 오히려 두 무리가 위아래로 확실히 갈린다.
    # 띠(사분위 범위)는 빼고 중앙값 선만 남긴다 — 6개 곡선에 띠까지 얹으면
    # 서로를 가린다. 흩어짐은 표(current_by_cycle.csv)와 캡션에 남긴다.
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    cyc_c = ['tab:blue', 'tab:orange', 'tab:green']
    # 방전은 '점선'. 파선(--)은 범례의 짧은 선분에서 실선과 구분이 잘 안 됐다.
    DOT = (0, (1.4, 1.6))
    for k, (col, c) in enumerate(zip(CHG, cyc_c), start=1):
        ax.plot(d.c_in * MM, d[col], 'o-', c=c, ms=6, lw=2.0,
                label=f'Charge, cycle {k}')
    for k, (col, c) in enumerate(zip(DIS, cyc_c), start=1):
        ax.plot(d.c_in * MM, d[col], 'o', ls=DOT, c=c, ms=6, lw=2.2,
                mfc='white', mew=1.6, label=f'Discharge, cycle {k}')
    ax.set_xlabel('Feed concentration (M)')
    ax.set_xticks([x * MM for x in CBINS])
    ax.grid(True, ls=':', alpha=.5)
    # 방전 전류는 실제로는 음수(-1.5 mA)지만 크기만 비교하는 그림이라 절댓값으로
    # 그린다. 축 라벨에 magnitude 임을 적어 부호 오해를 막는다.
    ax.set_ylabel('Optimal current magnitude (mA)')
    # handlelength 를 늘려야 범례 안에서 점선 무늬가 보인다(기본 2.0 은 짧다).
    ax.legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False,
              fontsize=13.5, handlelength=3.4, labelspacing=.8)
    # 제목은 짧게 둔다. 다만 그림에 그려진 것이 전선 전체가 아니라 최대탈염
    # 끝단 근처(탈염량 상위 TOL 이내)의 해들이라는 점은 캡션에 반드시 적을 것.
    # 사이클별 전류 감소는 그 끝단에서만 성립한다 (p00 에서는 정전류로 수렴).
    ax.set_title('Near-optimal protocol', fontsize=16.5)

    fig.tight_layout(w_pad=3.4)
    p = os.path.join(OUT_DIR, 'pareto_currents.png')
    fig.savefig(p, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved:', p)


def main():
    front = pd.read_csv(os.path.join(OUT_DIR, 'pareto_front.csv'))
    base = pd.read_csv(os.path.join(OUT_DIR, 'baseline_constant.csv'))
    levels = sorted(set(front.c_in))
    print(f'농도 {len(levels)}개  {min(levels):.0f} ~ {max(levels):.0f} mol/m3   '
          f'파레토 해 {len(front)}개')
    _style()
    g = fig_pareto(front, base, levels)
    if set(CHG) <= set(front.columns):
        fig_currents(front, levels)

    print('\n농도별 개선폭 (정전류 대비)')
    print(f'{"c_in":>8}{"같은 시간 탈염 +%":>20}{"같은 탈염 시간 -%":>20}')
    for _, r in g.iterrows():
        print(f'{r.c_in:>8.0f}{r.gain:>20.1f}{-r.save:>20.1f}')


if __name__ == '__main__':
    main()
