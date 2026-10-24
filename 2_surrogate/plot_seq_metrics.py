# -*- coding: utf-8 -*-
"""시퀀스 모델 3종(LSTM / Transformer / TFT) 비교.  seed 3개.

그림 fig_seq_metrics.png (2 x 1, 세로로 쌓고 x축 공유)
    (a) 대표 조건의 전압 궤적     — (b) 의 봉우리가 충전-방전 전환점 자리임을
                                 눈으로 보이는 맥락 패널. 세로 점선이 두 칸을
                                 관통한다.
    (b) 정규화 시간에 따른 오차   — 세 모델(LSTM/Transformer/TFT) 비교.
                                 표로는 못 담는 것, 이 그림의 존재 이유다.
                                 띠 = seed 3개의 범위(조건 산포가 아니다).

★ 구간 수(NBIN)를 12 -> 24 로 올렸다. 결론이 하나 바뀐다
    12등분에서는 'Transformer 가 뒤로 갈수록 LSTM 보다 나빠진다(역전)' 로 읽혔다.
    구간 수를 바꿔 보면 그 역전이 사라진다 (뒤 15% 의 Transformer - LSTM):

        12등분 +0.67 mV   18등분 +0.32   24등분 +0.07   36등분 -0.69   48등분 -0.45

    12등분은 한 구간이 중앙 13 시간이라 충전 끝의 오차 봉우리가 구간을 통째로
    물들이고, 뒤 15% 창(구간 2개)의 절반을 그 봉우리가 차지한다. 구간을 잘게
    쪼개면 봉우리는 제자리에 또렷이 서고 창 평균은 희석된다. 즉 역전은 물리가
    아니라 구간 나누기의 산물이었다.

    구간 수를 바꿔도 남는 것 — 이쪽만 주장한다
      · TFT 가 전 구간에서 가장 낮다 (뒤 15% 에서 0.9~1.7 mV, 나머지 둘 2.3~4.8)
      · Transformer 는 자기 시작 대비 뒤가 2.5~3.1 배로 커진다. LSTM 은 1.0~1.2 배
        — 즉 '수평선이 길어질수록 나빠지는' 성질은 Transformer 쪽이다. 다만
        출발이 훨씬 좋아서 끝까지 LSTM 보다 나쁘지는 않다.
      · 오차는 충전 끝(4.2 V 직전)에 몰린다. 최적화 목적함수가 그 도달시각이므로
        표의 '도달시각 오차' 가 가장 중요한 값인 이유가 여기 있다.

    ※ (a) 는 조건 하나, (b) 는 37조건의 중앙값이다. 조건마다 충전/방전 시간
      비중이 달라 (a) 의 세로 점선과 (b) 의 봉우리는 정확히 겹치지 않는다.
      완전히 맞추려면 t/T 대신 '사이클 + 위상 내 진행률' 로 나눠야 한다.
표  seq_metrics.csv               모든 지표를 seed 평균 ± 표준편차로

★ 이 그림은 '어떤 구조를 쓸 것인가' 만 답한다. 고른 구조(TFT)를 '어떤 방식으로
  굴릴 것인가' — 자기회귀 vs 궤적예측 — 는 fig_ar_direct.png 로 뺐다
  (replot_autoregressive_new.py 가 그린다). 한때 두 질문을 2x2 한 장에 담았는데,
  윗줄은 세 모델·아랫줄은 TFT 하나라 비교 대상이 칸마다 달라 읽는 사람이
  범례를 두 번 다시 확인하게 됐다. 질문이 둘이면 그림도 둘이다.

★ RMSE/MAE 막대는 그리지 않는다 — 표에 이미 있는 값을 그림으로 한 번 더
  그리면 독자가 두 곳을 대조하게 되고, 그림이 '표의 삽화' 가 된다. 이 그림은
  표가 못 하는 것(시간에 따른 변화)만 맡는다.

★ 두 출력을 모두 본다
    최적화의 목적함수가 둘이다 — 운전시간은 전압에서, 탈염량은 농도에서
    나온다. 전압만 평가하면 절반만 본 것이다. 실제로 2·3위가 뒤집힌다.

★ 본문 표에 쓰는 다섯 항목
    RMSE, MAE   딥러닝 시계열 예측 논문의 사실상 표준. Informer(AAAI 2021),
                Autoformer(NeurIPS 2021), PatchTST(ICLR 2023), TFT(Lim et al.,
                Int. J. Forecasting 2021) 모두 이 둘로 보고한다.
    MAPE        무차원 상대오차. 전압 2.8~4.2 V, 농도 200~900 mM 로 0 에서
                멀어 MAPE 의 약점(0 근처 발산)이 여기서는 생기지 않는다.
    도달시각 오차 4.2 V 에 언제 닿는가. 최적화 목적함수가 운전시간이므로
                '전압을 얼마나 맞히나' 가 아니라 '최적화 결과를 얼마나
                틀리게 만드나' 를 직접 잰다. 이 표에서 가장 중요한 값이다.
    파라미터 수  구조로 정해지는 값이라 학습·기계 상태와 무관하다.
                (학습시간은 부하에 좌우돼 같은 모델이 287~626 s 로 흔들렸다.
                 벤치마크로 쓸 수 없어 뺐다.)

★ CSV 에만 남기는 것 — 심사 대응용
    MASE     Hyndman & Koehler (2006), IJF 22:679. 척도무관 지표지만 분모가
             '1스텝 naive 예보' 라, 입력을 다 받는 이 시뮬레이션 과제에는
             기준선이 허수아비다. 그리고 여기서는 MAPE 와 거의 비례한다
             (MASE/MAPE 가 세 모델에서 1.05~1.09 로 일정).
    R2_seg   위상별 R2. 전압은 0.990/0.993/0.9986 으로 갈리지만 농도는
             0.9985~0.9997 로 포화라 표에 넣으면 '다 똑같다' 로 읽힌다.
             R2_all 은 세 모델 전부 0.9997~1.0000 이라 아예 못 쓴다 —
             충전(3.3~4.2 V)과 방전(2.8~3.0 V)의 위상 차이가 분산의
             대부분이라 '충전이면 3.7 V' 만 맞혀도 0.999 가 나온다.

★ 표준 지표는 표로, 그림은 표가 못 하는 것만
    Informer/Autoformer/PatchTST 등도 지표는 전부 표로 싣는다.

    py 2_surrogate/plot_seq_metrics.py
"""
import os
import sys
import warnings

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, '2_surrogate'))
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import plotstyle as _PS
import surrogate as S

OUT_DIR = os.path.join(S.BASE, '_S_result')
CKPT = os.path.join(OUT_DIR, 'ar_models')
SEQ = ['LSTM', 'Transformer', 'TFT']
COLS = ['tab:orange', 'tab:green', 'tab:red']     # fig6 의 모델 색과 맞춘다
SEEDS = [0, 1, 2]
EPOCHS = 150
# (b) 정규화 시간 구간 수. 명령줄에서 nbin= 로 바꿀 수 있고, 기본값이 아니면
# 파일 이름에 꼬리표가 붙어 기존 그림을 덮지 않는다.
#   12 -> 구간당 중앙 13 스텝(=시간). 전환점 봉우리가 구간 안에 묻힌다.
#   24 -> 중앙 6.5, 최소 2.0. 봉우리가 또렷해지고 결론이 바뀐다(아래 참고).
NBIN = 24
_NSUF = ''
for _a in sys.argv[1:]:
    if _a.startswith('nbin='):
        NBIN = int(_a.split('=')[1])
        _NSUF = '_nbin%d' % NBIN


def wpath(name, seed):
    """★ 반드시 figures_{name}_s{seed}.pt 만 쓴다.

    예전에는 seed 0 에 한해 figures_{name}.pt 를 읽었는데, 그 파일은 08-20 에
    학습된 것이라 어제 학습한 seed 1/2 와 **같은 실행이 아니다**. 실제로
    같은 seed 인데 LSTM 전압 RMSE 가 8.32(구) vs 5.19(신) 로 갈렸다.
    그런 것을 섞어 평균·표준편차를 내면 안 된다. 세 seed 를 같은 조건에서
    다시 학습해(train_seq_seeds.py 0 1 2) 그 결과만 쓴다.

    figures_{name}.pt 는 Fig.6 / parity / SI 그림이 쓰므로 건드리지 않는다.
    """
    return os.path.join(CKPT, f'figures_{name}_s{seed}.pt')


def load():
    pool = S.load(list(S.USE), verbose=False)
    tr, va, te = S.split(pool, verbose=False)
    sc = S.Scaler().fit(tr)
    fit = {}
    for n in SEQ:
        for sd in SEEDS:
            p = wpath(n, sd)
            if not os.path.exists(p):
                print(f'  [{n} s{sd}] 가중치 없음 — 건너뜀')
                continue
            m = S.build(n, epochs=EPOCHS, seed=sd)
            m.net = S.TORCH_MODELS[n](**m.kwargs).to(S.DEVICE)
            m.net.load_state_dict(torch.load(p, map_location=S.DEVICE))
            m.net.eval()
            fit[(n, sd)] = m
        print(f'  [{n}] seed {[s for s in SEEDS if (n, s) in fit]}')
    return tr, te, sc, fit


def metrics(tr, te, sc, fit):
    """(모델, seed) 별 지표 + 정규화 시간 프로파일."""
    # MASE 분모 — 학습셋의 1스텝 naive 예측 MAE (Hyndman & Koehler 2006)
    naive = tuple(float(np.mean(np.concatenate(
        [np.abs(np.diff(r[c].to_numpy(float))) for r in tr.values()])))
        for c in ('Vcell', 'c_mix'))
    rows, prof = [], {}
    for (n, sd), m in fit.items():
        E, O, EC, OC, pr = [], [], [], [], []
        for k, r in te.items():
            p = S.to_c_mix(sc.back(m.predict(sc.forward(r))), r)
            o = r['Vcell'].to_numpy(float)
            e = p[:, 0] - o
            oc_ = r['c_mix'].to_numpy(float)
            E.append(e)
            O.append(o)
            EC.append(p[:, 1] - oc_)
            OC.append(oc_)
            # 조건마다 길이가 달라 t/T 로 맞춰 구간별 RMSE.
            # ★ 3사이클 조건만 넣는다. test 54조건 중 3사이클 37 / 2사이클 9 /
            #   1사이클 8 이라 전부 섞으면 t/T 정렬이 어긋나 전환점 봉우리가
            #   뭉개진다. 지표(표)는 54조건 전부를 쓴다 — 프로파일만 추린다.
            if r['cycle'].nunique() >= 3:
                T = len(e)
                b = np.clip((np.arange(T) / max(T - 1, 1) * NBIN).astype(int),
                            0, NBIN - 1)
                pr.append([np.sqrt(np.mean(e[b == i] ** 2)) * 1000
                           if (b == i).any() else np.nan for i in range(NBIN)])
        # seed 하나의 프로파일 = 조건들의 중앙값
        prof[(n, sd)] = np.nanmedian(np.array(pr, float), axis=0)
        e, o = np.concatenate(E), np.concatenate(O)
        ec, oc = np.concatenate(EC), np.concatenate(OC)
        # 4.2 V 도달시각 오차 — 최적화 목적함수가 운전시간이므로
        # '전압을 얼마나 맞히나' 가 아니라 '최적화를 얼마나 틀리게 만드나' 다.
        cm, ch = [], []
        for k, r in te.items():
            pr = S.to_c_mix(sc.back(m.predict(sc.forward(r))), r)
            a, b = S.ceiling_error(pr, r)
            cm += a
            ch += b
        sm_ = S.split_metrics(m, te, sc)
        rows.append(dict(
            model=n, seed=sd,
            V_RMSE_mV=np.sqrt((e ** 2).mean()) * 1000,
            V_MAE_mV=np.abs(e).mean() * 1000,
            V_MAPE_pct=np.mean(np.abs(e / o)) * 100,
            c_RMSE_mM=np.sqrt((ec ** 2).mean()),
            c_MAE_mM=np.abs(ec).mean(),
            c_MAPE_pct=np.mean(np.abs(ec / oc)) * 100,
            ceil_h=float(np.median(ch)),
            # 아래는 본문 표에 안 쓰지만 CSV 에 남긴다 (심사 대응용)
            V_MASE=np.abs(e).mean() / naive[0],
            c_MASE=np.abs(ec).mean() / naive[1],
            V_R2seg=sm_['Vcell']['r2_seg'],
            c_R2seg=sm_['c_mix']['r2_seg']))
    return pd.DataFrame(rows), prof, naive


def save_csv(d, name):
    """엑셀에서 열려 잠겨 있으면 _new 로 우회 — 그림은 계속 그린다.

    ★ 둘 다 잠기면 조용히 지나가면 안 된다. 예전 판이 그래서 CSV 가 이틀 전
      값 그대로인데 아무 말도 없었다. 실패하면 크게 알린다.
    """
    p = os.path.join(OUT_DIR, name)
    for path in (p, p.replace('.csv', '_new.csv')):
        try:
            d.to_csv(path, encoding='utf-8-sig')
            print(f'  saved: {os.path.basename(path)}'
                  + ('   (원래 이름이 잠겨 우회)' if path != p else ''))
            return
        except PermissionError:
            continue
    print(f'  [!!] {name} 저장 실패 — 원본과 _new 둘 다 잠겨 있다. '
          f'엑셀에서 닫고 다시 돌릴 것.')


def main():
    print(f'\n시퀀스 모델 {len(SEQ)}종 x seed {SEEDS}')
    tr, te, sc, fit = load()
    raw, prof, naive = metrics(tr, te, sc, fit)

    n_par = {n: sum(p.numel() for p in fit[(n, SEEDS[0])].net.parameters()
                    if p.requires_grad) / 1e3 for n in SEQ}
    # 본문 표 다섯 항목: RMSE, MAE, MAPE (출력 2개) + 도달시각 오차 + 파라미터
    MAIN = ['V_RMSE_mV', 'V_MAE_mV', 'V_MAPE_pct',
            'c_RMSE_mM', 'c_MAE_mM', 'c_MAPE_pct', 'ceil_h']
    cols = [c for c in raw.columns if c not in ('model', 'seed')]
    mu = raw.groupby('model')[cols].mean().loc[SEQ]
    sd = raw.groupby('model')[cols].std().loc[SEQ]
    mu['params_k'] = [n_par[n] for n in SEQ]

    pd.set_option('display.width', 240)
    print(f'\nMASE 분모(naive 1-step MAE, train)'
          f'   전압 {naive[0] * 1000:.2f} mV   농도 {naive[1]:.2f} mM')
    print(f'테스트 {len(te)}조건, seed {sorted(raw.seed.unique())}'
          f'  (mM = mol/m^3)\n')
    print('[본문 표]')
    print(f'{"":13}' + ''.join(f'{c:>20}' for c in MAIN))
    for n in SEQ:
        print(f'{n:13}' + ''.join(
            f'{mu.loc[n, c]:>12.3f} ±{sd.loc[n, c]:6.3f}' for c in MAIN))
    extra = [c for c in cols if c not in MAIN]
    print('\n[CSV 에만 — 심사 대응용]')
    print(f'{"":13}' + ''.join(f'{c:>20}' for c in extra))
    for n in SEQ:
        print(f'{n:13}' + ''.join(
            f'{mu.loc[n, c]:>12.4f} ±{sd.loc[n, c]:6.4f}' for c in extra))
    print(f'\n파라미터 (천개): ' +
          '   '.join(f'{n} {n_par[n]:.0f}' for n in SEQ))

    save_csv(raw.set_index(['model', 'seed']), 'seq_metrics_by_seed.csv')
    # ★ 평균/표준편차를 pd.concat 으로 붙이면 헤더가 mean, mean.1, std.1 ...
    #   로 나와 엑셀에서 못 읽는다. 열 이름을 직접 만든다.
    flat = pd.DataFrame(index=mu.index)
    for c in cols:
        flat[c + '_mean'] = mu[c]
        flat[c + '_sd'] = sd[c]
    flat['params_k'] = mu['params_k']
    save_csv(flat, 'seq_metrics.csv')

    # seed 마다 순위가 같은가 — 이 비교의 신뢰도를 좌우한다
    for col, lab in (('V_RMSE_mV', '전압 RMSE'), ('c_RMSE_mM', '농도 RMSE')):
        piv = raw.pivot_table(index='seed', columns='model', values=col)
        order = [list(piv.loc[s].sort_values().index) for s in piv.index]
        print(f'  {lab} 순위 seed 전부 동일: {all(o == order[0] for o in order)}'
              f'   ({" < ".join(order[0])})')

    # ── 그림 ────────────────────────────────────────────────────────────
    # 대표 조건 — 3사이클 중 c_in 이 600 에 가장 가까운 것. ar_horizon.png 와
    # 같은 조건을 쓴다(두 그림을 나란히 놓고 읽게 된다).
    te3 = {k: r for k, r in te.items() if r['cycle'].nunique() >= 3}
    ref = te3[min(te3, key=lambda k: abs(float(te3[k]['c_in'].iloc[0]) - 600.))]
    ref_x = np.arange(len(ref)) / (len(ref) - 1)

    _PS.apply()
    # 세로로 쌓고 x축(t/T)을 공유한다 — 위 칸의 어느 지점이 아래 칸의 어느
    # 봉우리인지를 눈이 세로로 따라가게 하려는 것이다. 그래서 위 칸의 x 눈금
    # 라벨은 지운다(아래 칸 것이 두 칸 모두의 눈금이다).
    fig = plt.figure(figsize=(8.8, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45])
    ax = [fig.add_subplot(gs[0, 0]), None]
    ax[1] = fig.add_subplot(gs[1, 0], sharex=ax[0])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    xb = (np.arange(NBIN) + .5) / NBIN

    # ── (a) 대표 궤적 ───────────────────────────────────────────────────
    # 예측을 겹쳐 그리지 않는다 — 전압 진폭 1310 mV 에 RMSE 3~9 mV 라 세 모델의
    # 선이 관측선 굵기 안에서 완전히 겹친다. 이 칸의 역할은 정량 비교가 아니라
    # (b) 의 봉우리가 어디인지 보이는 것이다.
    ax[0].plot(ref_x, ref['Vcell'].to_numpy(float), '-', c='black', lw=2.4)
    ax[0].set_ylabel('Cell voltage (V)')
    ax[0].set_title('Representative condition  '
                    f'($c_{{in}}$ = {ref["c_in"].iloc[0] * 1e-3:.3f} M)')
    ax[0].grid(True, ls=':', alpha=.5)

    # ── (b) 정규화 시간에 따른 오차 ─────────────────────────────────────
    # 선 = seed 평균,  띠 = seed 최소~최대. 조건 산포가 아니라 **seed 산포**다.
    # 이 그림의 주장(Transformer 가 뒤로 갈수록 무너진다)이 seed 하나의
    # 우연인지 아닌지가 여기서 판정된다.
    for n, c in zip(SEQ, COLS):
        P = np.array([prof[(n, s_)] for s_ in SEEDS if (n, s_) in prof], float)
        ax[1].fill_between(xb, P.min(axis=0), P.max(axis=0), color=c,
                           alpha=.18, lw=0)
        ax[1].plot(xb, P.mean(axis=0), '-o', c=c, lw=2.2, ms=6, label=n)
    ax[1].set_xlabel('Normalized time, $t/T$')
    ax[1].set_ylabel('Voltage RMSE (mV)')
    ax[1].set_yscale('log')
    ax[1].grid(True, ls=':', alpha=.5)
    ax[1].legend(frameon=False, loc='upper left', ncol=3)

    # 전환점 세로선 — 대표 궤적에서 위상이 바뀌는 자리
    ph = ref['phase'].to_numpy()
    for i in np.where(ph[1:] != ph[:-1])[0]:
        for a_ in ax:
            a_.axvline(ref_x[i], c='0.55', ls=(0, (3, 3)), lw=1.0, zorder=0)

    fig.tight_layout(w_pad=_PS.W_PAD)
    _PS.panels(fig, ax, letters=['a', 'b'], dy=.006, align=True)
    p = os.path.join(OUT_DIR, 'fig_seq_metrics%s.png' % _NSUF)
    fig.savefig(p, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('\nsaved:', p)


if __name__ == '__main__':
    main()
