# -*- coding: utf-8 -*-
"""모든 그림의 공통 서식. 한 군데서 고치면 전부 반영되게 하려고 분리했다.

쓰는 쪽:
    import sys; sys.path.insert(0, '<repository root>')
    from plotstyle import apply
    plt = apply()

규칙
  - 글꼴은 Times New Roman, 수식은 STIX (논문 본문과 맞춘다)
  - 제목·축이름은 굵게
  - 제목에는 '무엇을 그렸는지'만 쓴다. 읽는 법(band = ..., 1 = best 같은 설명)은
    캡션에 쓸 내용이지 그림 제목에 넣을 것이 아니다.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 260824: 전 항목을 다시 2포인트씩 키웠다 (발표·인쇄에서 잘 보이도록).
#   이전 값 -> 지금 값 :  title 14->16,  label 13->15,  tick 11->13,
#                        legend 11->13,  figure title 15->17
RC = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    # 260922: 범례만 2포인트 내렸다 (사용자 요청) — 항목이 많은 그림에서
    #         범례가 본 그림 폭을 넘어 두 칸이 서로 침범했다.
    'legend.fontsize': 13,
    'figure.titlesize': 19,
    # 260826: 인쇄·투고용으로 600 dpi. 스크립트에서 savefig(dpi=...) 를 주면
    # 그쪽이 우선하므로, 개별 스크립트도 함께 600 으로 맞춰 두었다.
    'savefig.dpi': 600,
    'figure.dpi': 150,
    # 260826: (a)/(b) 패널 사이가 좁아 축 라벨과 패널 기호가 붙어 보였다.
    # tight_layout 을 쓰면 이 값이 무시되므로 W_PAD 를 함께 넘겨야 한다.
    'figure.subplot.wspace': 0.34,
}

# 가로 패널 간격. fig.tight_layout(w_pad=_PS.W_PAD) 로 넘긴다.
W_PAD = 3.4

# ─────────────────────────────────────────────────────────────────────────────
# 261004(3): 파스텔 톤으로 바꿨다 (사용자 요청).
#
#   slate - plum - sage - terracotta - ochre  (261004(2) 와 같은 색상 계열이되
#   채도를 낮추고 밝기를 올린 판)
#
# ★ 파스텔은 흑백 분리가 약해진다 — 알고 쓰는 것
#     파스텔로 만들면 다섯 색의 회색 명도가 0.58~0.83 사이로 모인다. 저채도
#     판(0.32~0.78)만큼 벌릴 수가 없다. 그래도 이웃 간 0.04 이상은 남기도록
#     밝기를 계단으로 잡았다:
#         0.58  0.62  0.66  0.70  0.83
#     흑백 인쇄가 필요하면 선 모양(실선/파선)이나 표식을 함께 써야 한다.
MODEL_COLORS = ['#6f9bc9',    # slate       0.58
                '#b98bc0',    # plum        0.62
                '#86be8b',    # sage        0.66
                '#efa076',    # terracotta  0.70
                '#f0d385']    # ochre       0.83

# 계열이 둘뿐인 그림(실험 vs 모형, 에너지 vs 시간 …)에 쓰는 두 색.
C_PRIMARY = '#6f9bc9'      # slate
C_SECOND = '#ef9b6e'       # terracotta — 두 색만 쓸 때는 조금 진하게
C_THIRD = '#86be8b'        # sage
C_FOURTH = '#b98bc0'       # plum
C_ACCENT = '#f0d385'       # ochre  — 강조점 하나에만
C_NEUTRAL = '#9aa2aa'      # 중립 회색 (기준선·정전류 등)

# 농도 구간처럼 **몇 개로 끊어 쓰는** 색. 연속 색지도에서 뽑으면 뒤의 두 색이
# 거의 같아지므로 목록에서 고른다 (261004 확인).
_BAND_SETS = {
    2: ['#6f9bc9', '#efa076'],
    3: ['#6f9bc9', '#86be8b', '#efa076'],
    4: ['#6f9bc9', '#86be8b', '#efa076', '#f0d385'],
    5: ['#6f9bc9', '#b98bc0', '#86be8b', '#efa076', '#f0d385'],
}


def bands(n, mono=False):
    """구간 n 개에 쓸 색.

    mono=True 면 **한 계열의 밝기 계단**으로 준다 (261005, 사용자 요청).
    농도처럼 순서가 있는 구간에 색상 넷을 쓰면 '색이 너무 다양하다' 는 인상을
    준다. 같은 파스텔 파랑을 옅은 쪽부터 진한 쪽으로 늘어놓으면 순서도 함께
    읽히고 그림이 차분해진다.
    """
    import numpy as _np
    if mono:
        # 색지도에서 뽑으면 옅은 쪽이 거의 흰색이라 흰 바탕의 점이 안 보인다.
        # 명도 0.80 - 0.45 사이의 네 단계를 직접 적어 둔다.
        _MONO = ["#b8d0e6", "#8fb4d8", "#6f9bc9", "#4d7aa8"]
        if n <= len(_MONO):
            idx = _np.linspace(0, len(_MONO) - 1, n).round().astype(int)
            return [_MONO[i] for i in idx]
        return [SEQ(x) for x in _np.linspace(.30, 1.0, n)]
    if n in _BAND_SETS:
        return list(_BAND_SETS[n])
    return [BAND(x) for x in _np.linspace(.04, .96, n)]


# 연속 색지도도 파스텔로. SEQ 는 히트맵처럼 '많고 적음' 에, BAND 는 구간에.
def _cmap(name, colors):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


SEQ = _cmap('swb_seq', ['#fbfdff', '#e2eef8', '#c3dbee', '#9fc2de', '#7aa5cb'])
SEQ_WARM = _cmap('swb_warm', ['#fffaf4', '#fbe7d4', '#f5cdaa', '#efa076', '#d98455'])
BAND = _cmap('swb_band', ['#6f9bc9', '#86be8b', '#efa076', '#f0d385'])

# 반투명 면(바이올린 등)에 쓸 때의 최소 alpha. 파스텔은 이미 옅으므로
# 저채도 판(0.75)보다 높게 잡아야 흰 바탕에 묻히지 않는다.
MODEL_FILL_ALPHA = 0.90

# 그림 안 글자(지표 상자, 막대 라벨 등)에 쓰는 크기. 함께 2포인트 키웠다.
FS_ANNOT = 15
FS_SMALL = 14
# 아래 _rescale_consts() 에서 FIGSCALE 만큼 다시 올린다.


# 패널 기호를 축 왼쪽 밖으로 얼마나 내보낼지 (축 너비 대비 비율).
# 0 이면 축 왼쪽 모서리, 음수면 그보다 왼쪽 = y축 제목 위쪽.
PANEL_DX = -0.16


def panel(ax, letter, dx=PANEL_DX, **kw):
    """패널 기호 (a), (b), ... 를 제목과 같은 줄에, y축 제목 위쯤에 놓는다.

    matplotlib 은 한 축에 loc='center'/'left'/'right' 제목을 동시에 가질 수 있다.
    본 제목은 그대로 두고 왼쪽 제목 자리에 기호만 넣으면 별도 좌표 계산 없이
    항상 제목과 같은 높이에 정렬된다. 가로 위치만 x 로 밀어 낸다.

    ★ dx 는 축 너비 대비 비율이라, 눈금 라벨이 길거나 축이 좁으면 y축 제목과
      어긋난다. 그럴 때는 호출부에서 dx 를 따로 준다.
    """
    ax.set_title(f'({letter})', loc='left', fontweight='bold',
                 fontsize=RC['axes.titlesize'], x=dx, **kw)
    return ax


def panels(fig, axes, letters=None, dy=0.0, align=False):
    """패널 기호를 'y축 제목과 같은 x, 제목과 같은 y' 에 놓는다.

    왜 set_title(loc='left') 로는 안 되는가
        그 방식은 x 를 축(axes) 좌표로만 줄 수 있는데, y축 제목의 가로 위치는
        눈금 라벨 길이에 따라 그림마다 달라진다. 고정 오프셋(dx)으로는 어떤
        그림에서는 맞고 어떤 그림에서는 어긋난다.

    그래서 한 번 그린 뒤 실제 배치를 재서 그 자리에 글자를 놓는다.
        x = y축 제목 상자의 왼쪽 끝,  y = 제목 상자의 아래쪽
    align=True 면 넘긴 축들의 기호 x 를 하나로 맞춘다(가장 왼쪽 기준).
        세로로 쌓인 두 축은 눈금 라벨 폭이 달라 y축 제목의 x 도 달라진다.
        그대로 두면 (b) 와 (c) 가 좌우로 어긋나 보인다.

    ★ 반드시 tight_layout 다음, savefig 앞에서 부를 것. 위치를 재야 하므로
      레이아웃이 확정된 뒤여야 한다.
    """
    import numpy as _np
    ax_list = [a for a in _np.ravel(_np.asarray(axes, dtype=object))]
    if letters is None:
        letters = [chr(ord('a') + i) for i in range(len(ax_list))]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    # 1) 축마다 기호 자리를 먼저 재고, 2) align 이면 가장 왼쪽으로 통일한다.
    #    예전에는 x 를 먼저 통일한 뒤 축별 보정을 해서, 보정이 걸린 칸과
    #    안 걸린 칸의 기호가 좌우로 어긋났다 (261004).
    boxes = [a.get_window_extent(renderer=r) for a in ax_list]

    def left_limit(i):
        """i 번째 칸의 기호를 왼쪽으로 얼마까지 뺄 수 있나 (픽셀).

        같은 높이에 왼쪽 이웃 칸이 있으면 그 칸의 오른쪽 변까지가 한계다.
        세로로 쌓인 그림에는 이웃이 없으므로 그림 왼쪽 끝까지 쓸 수 있다.
        이 구분을 안 하면, 가로로 나란한 그림에서 (b) 기호가 (a) 칸 위로
        넘어가 어느 칸의 것인지 알 수 없게 된다 (261004).
        """
        me = boxes[i]
        lim = 0.0
        for j, other in enumerate(boxes):
            if j == i or other.x1 > me.x0:
                continue
            if other.y1 > me.y0 and other.y0 < me.y1:      # 높이가 겹치는 이웃
                lim = max(lim, other.x1 + 0.02 * me.width)
        return lim

    spots = []
    for i, ax in enumerate(ax_list):
        ab = boxes[i]
        # y축 제목이 비어 있으면(이미지 패널 등) 그 상자는 폭 0 이라 자리를 못
        # 잡는다. 그럴 때는 축 왼쪽 모서리에서 조금 왼쪽을 쓴다.
        if ax.get_ylabel():
            yl = ax.yaxis.label.get_window_extent(renderer=r)
        else:
            yl = type('B', (), {'x0': ab.x0 - 0.030 * ab.width, 'y1': ab.y1})()
        ti = ax.title.get_window_extent(renderer=r)
        x_px = yl.x0
        if ti.height > 1:
            # 제목이 있으면 제목 아래쪽에 맞춘다.
            ytop = ti.y0
        elif yl.y1 > ab.y1 + 1:
            # 261004: 제목이 없고 y축 제목이 축보다 위로 뻗는 경우.
            #   글씨를 키우면 긴 세로 제목('Total salt removal (mM)')이 축
            #   높이를 넘어서, 예전처럼 yl.y1 에 기호를 놓으면 제목 글자
            #   한가운데에 찍혔다. 축 윗변 높이에 두고 제목 **왼쪽**으로 뺀다.
            ytop = ab.y1
            x_px = yl.x0 - 2.4 * RC['axes.titlesize'] * fig.dpi / 72.0
            lim = left_limit(i)
            if x_px < lim:
                # 왼쪽에 자리가 없다. 그럴 때는 제목 **위** 가 그나마 낫다 —
                # 다만 세로 제목이 축보다 높으면 거기도 글자와 겹친다. 그런
                # 그림은 호출부에서 제목을 두 줄로 쪼개 해결한다.
                x_px, ytop = yl.x0, max(ab.y1, yl.y1)
        else:
            ytop = max(ab.y1, yl.y1)
        spots.append((inv.transform((x_px, 0))[0], inv.transform((0, ytop))[1]))

    if align:
        x_min = min(x for x, _ in spots)
        spots = [(x_min, y) for _, y in spots]

    for (x, y), L in zip(spots, letters):
        fig.text(x, y + dy, f'({L})', ha='left', va='bottom',
                 fontsize=RC['axes.titlesize'], fontweight='bold')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 261004(2): 인쇄 크기 모드.  FIGPT=1 로 켠다 (기본 꺼짐 = 예전 그대로).
#
# 왜 '배율' 로는 안 되는가
#     종이 위 글자 크기는 (글자 포인트) x (그림을 넣는 폭) / (그림 캔버스 폭)
#     이다. 캔버스가 16 in 인 그림을 A4 에 11.5 cm 로 넣으면 축소율이 0.28 이라
#     24.6 pt 짜리 글자가 종이에서는 7 pt 가 된다. 배율을 아무리 올려도
#     캔버스가 넓으면 따라잡지 못한다.
#
#     그래서 **그림을 넣을 크기 그대로 그린다.** 캔버스 폭을 11.5 cm 로 잡고
#     글자를 13 pt 로 박으면, 100 % 로 넣었을 때 종이에서 정확히 13 pt 다.
#     대신 칸이 좁아지므로 가로로 나란하던 칸은 세로로 쌓아야 한다 —
#     그 재배치는 그림마다 호출부에서 한다 (_PS.size 참고).
#
#     사용자 요청(261004): A4 에 10~12 cm 로 넣고 종이 위 12~14 pt.
import os as _os

#   폭 등급이 둘이다.
#     PRINT_W      11.5 cm — 기본. 칸이 하나나 둘인 그림.
#     PRINT_W_WIDE 16.5 cm — A4 본문 전폭. 칸이 셋 이상이거나 3D·히트맵이
#                            들어가 좁은 폭에서는 세로로 쌓아도 A4 높이를
#                            넘기는 그림.
#   두 등급 모두 글자는 같은 12~14 pt 다. 넣을 때의 폭만 다르므로, 그림마다
#   어느 폭으로 넣어야 하는지 figure_list.csv 에 적어 둔다.
# 261005: 사용자가 원하는 최종 크기 — 가로 15~16 cm, 세로 14~15 cm.
#   Word 에 100 % 로 넣었을 때의 크기다. 그 안에서 글자는 14 pt.
#   ★ savefig(bbox_inches='tight') 가 흰 여백을 깎으므로, 코드의 figsize 보다
#     최종 파일이 조금 작다. 그래서 figsize 는 목표보다 살짝 크게 잡고
#     만들어진 png 를 재서 맞춘다 (check 는 final_v2/_src/collect.py 가 적는다).
PRINT_W_CM, PRINT_H_CM = 15.5, 14.5
PRINT_W = PRINT_W_CM / 2.54          # in  (= 6.10)
PRINT_H = PRINT_H_CM / 2.54          # in  (= 5.71)
# 예전 이름은 남겨 둔다 (아직 참조하는 그림이 있다).
PRINT_W_WIDE_CM = PRINT_W_CM
PRINT_W_WIDE = PRINT_W

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGPT = _os.environ.get('FIGPT') == '1'

# 인쇄용 절대 글자 크기 (요청 범위 12~14 pt 안).
# 261004(3): 전부 한 포인트씩 올렸다 (사용자 요청).
#   13 -> 14 pt. 종이 위에서도 그대로 14 pt 다 (캔버스를 넣을 크기로 잡으므로).
PRINT_RC = {
    'axes.titlesize': 15, 'axes.labelsize': 14,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 13, 'figure.titlesize': 15,
    # 좁은 캔버스에서는 선이 굵으면 자료를 덮는다. 기본값 근처로 되돌린다.
    'axes.linewidth': 0.9,
    'lines.linewidth': 1.5,
    'lines.markersize': 4.5,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
    'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
    'axes.labelpad': 3.0,
    'axes.titlepad': 5.0,
    'grid.linewidth': 0.6,
    'patch.linewidth': 0.8,
    'legend.handlelength': 1.5,
    'legend.borderpad': 0.3,
    'legend.labelspacing': 0.3,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.2,
    'figure.subplot.wspace': 0.30,
}

if FIGPT:
    RC.update(PRINT_RC)
    FS_ANNOT = 12.5
    FS_SMALL = 12.0
    W_PAD = 0.9
    PANEL_DX = -0.19


# 스크립트에 숫자로 박힌 선 굵기·표식 크기에 곱할 값.
#   좁은 캔버스에서는 같은 포인트 수의 선이 그림 폭의 3 배 비율을 차지한다.
#   비율을 그대로 맞추면(0.34) 1 pt 미만이 되어 인쇄에서 끊겨 보이므로,
#   눈으로 맞춘 0.60 을 쓴다.
LW_K = 0.60 if FIGPT else 1.0
MS_K = 0.70 if FIGPT else 1.0


def size(print_wh, legacy_wh):
    """FIGPT 면 인쇄용 크기, 아니면 예전 크기.

    그림마다 세로를 따로 잡아야 해서 (칸을 세로로 쌓으므로) 자동 계산이 아니라
    호출부에서 두 값을 같이 적는다. 예전 판을 다시 뽑을 길도 남는다.
    """
    return print_wh if FIGPT else legacy_wh


# ─────────────────────────────────────────────────────────────────────────────
# 261004: 글자 확대 배율. 환경변수 FIGSCALE 로만 켠다 (기본 1.0 = 예전 그대로).
#
# "그림 글씨가 작고 직관적이지 않다" 는 지적을 받아 전 그림을 다시 뽑을 때 쓴다.
# 그림마다 스크립트를 고치지 않고 한 군데서 올리려고 여기에 뒀다.
#     set FIGSCALE=1.45 && py 2_surrogate/plot_fig_models.py
#
# ★ RC·FS_ANNOT·FS_SMALL 을 모듈을 읽는 시점에 제자리에서 올린다.
#   panel()/panels() 가 RC['axes.titlesize'] 를 직접 보고, 개별 스크립트도
#   _PS.FS_ANNOT 을 그대로 쓰기 때문에, 여기서 올려야 전부 따라온다.
#
# 선 굵기·눈금 길이도 함께 키운다. 글자만 키우면 축선이 상대적으로 가늘어져
# 축소 인쇄에서 오히려 흐려 보인다. 글자와 같은 배율로 키우면 선이 너무 굵어
# 데이터를 덮으므로 제곱근 정도로 둔다.
try:
    FIGSCALE = float(_os.environ.get('FIGSCALE', '1'))
except ValueError:
    FIGSCALE = 1.0

if FIGSCALE != 1.0:
    for _k in ('axes.titlesize', 'axes.labelsize', 'xtick.labelsize',
               'ytick.labelsize', 'legend.fontsize', 'figure.titlesize'):
        RC[_k] = round(RC[_k] * FIGSCALE, 1)
    FS_ANNOT = round(FS_ANNOT * FIGSCALE, 1)
    FS_SMALL = round(FS_SMALL * FIGSCALE, 1)
    _w = FIGSCALE ** 0.5
    RC.update({
        'axes.linewidth': 0.8 * _w,
        'lines.linewidth': 1.5 * _w,
        'lines.markersize': 6.0 * _w,
        'xtick.major.width': 0.8 * _w, 'ytick.major.width': 0.8 * _w,
        'xtick.minor.width': 0.6 * _w, 'ytick.minor.width': 0.6 * _w,
        'xtick.major.size': 3.5 * _w, 'ytick.major.size': 3.5 * _w,
        'xtick.minor.size': 2.0 * _w, 'ytick.minor.size': 2.0 * _w,
        'grid.linewidth': 0.8 * _w,
        'patch.linewidth': 1.0 * _w,
        'legend.handlelength': 2.0 * _w,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.4,
    })
    # 글자가 커지면 패널 사이도 그만큼 벌어져야 축 제목이 옆 패널을 침범하지 않는다.
    W_PAD = W_PAD * FIGSCALE
    RC['figure.subplot.wspace'] = RC['figure.subplot.wspace'] * FIGSCALE


def apply(**over):
    plt.rcParams.update(RC)
    if over:
        plt.rcParams.update(over)
    return plt


# ─────────────────────────────────────────────────────────────────────────────
# 261004: FIGSVG=1 이면 png 를 저장할 때 같은 이름의 svg 도 함께 남긴다.
#
# 투고본은 벡터를 요구하는 경우가 많은데, 그림마다 스크립트를 고쳐
# savefig 를 두 번 부르게 하면 109 개를 다 건드려야 한다. 저장 자체를 한 번만
# 가로채는 편이 안전하다. 기본은 꺼짐이라 평소 동작은 그대로다.
if _os.environ.get('FIGSVG') == '1':
    from matplotlib.figure import Figure as _Figure

    _orig_savefig = _Figure.savefig

    def _savefig_also_svg(self, fname, *a, **kw):
        r = _orig_savefig(self, fname, *a, **kw)
        try:
            name = _os.fspath(fname)
        except TypeError:
            return r                      # 파일 객체 등 — 경로가 아니면 건너뛴다
        if name.lower().endswith('.png'):
            kw2 = dict(kw)
            kw2.pop('dpi', None)          # 벡터라 dpi 는 의미가 없다
            try:
                _orig_savefig(self, name[:-4] + '.svg', *a, **kw2)
            except Exception as e:
                print('  (svg 저장 실패: %s)' % e)
        return r

    _Figure.savefig = _savefig_also_svg
