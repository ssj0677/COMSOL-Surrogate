# -*- coding: utf-8 -*-
"""모델 5종 비교표 — 전부 '오늘 학습한 한 실행' 에서 뽑는다.

왜 새로 만드는가
    _S_result/compare_all_runs.csv (08-19) 는 지표만 남기고 **가중치를 저장하지
    않았다**. 그래서 거기 없는 지표(MAPE 등)를 추가하려면 재학습해야 하고,
    재학습하면 같은 모델의 숫자가 두 벌이 되어 표가 서로 어긋난다.
    실제로 그 일이 벌어졌다 — LSTM 전압 RMSE 가 7.50(08-19) vs 5.95(재학습).

    그래서 5모델 x 3 seed 를 **가중치까지 남긴 채** 한 번에 정리한다.
    앞으로 지표를 더 넣어도 재학습이 필요 없다.

    가중치   figures_{모델}_s{seed}.pt   (train_seq_seeds.py 가 만든다)
    XGBoost  torch 모델이 아니고 4초면 학습되므로 여기서 그때그때 학습한다.

지표
    RMSE, MAE   전압[mV] / 농도[mM]. 딥러닝 시계열 예측의 표준 보고 지표
                (Informer, Autoformer, PatchTST, TFT 원논문).
    MAPE        무차원 상대오차. 전압 2.8~4.2 V, 농도 200~900 mM 로 0 에서
                멀어 MAPE 의 약점(0 근처 발산)이 여기서는 생기지 않는다.
    파라미터 수  구조로 정해지는 값. 학습·기계 상태와 무관하다.
    (R2_seg, MASE, 4.2 V 도달시각 오차도 CSV 에 함께 남긴다.)

★ R2_seg 는 중앙값만으로는 모델을 못 가른다 — 꼬리를 함께 낸다
    노이즈 없는 COMSOL 출력을 맞히는 문제라 R2_seg 중앙값이 1 에 붙는다.
    농도는 LSTM 0.9982 / Transformer 0.9994 / TFT 0.9996 으로 셋이 0.002 안에
    들어오는데, 같은 데이터의 RMSE 는 1.84 / 1.56 / 0.69 mM 로 2.7 배 차이다.
    R2 가 1 근처에서 눌려 차이를 못 보여 주는 것이다.

    차이는 꼬리에 있다. 그래서 구간별 R2 분포에서 두 값을 더 낸다.
        *_R2seg_p10       하위 10 퍼센타일
        *_R2seg_lt99_pct  R2 < 0.99 인 구간의 비율 (%)
    전압에서 특히 갈린다 — 중앙값은 0.9903/0.9934/0.9985 로 붙어 있지만
    0.99 미만 구간이 50.1% / 40.9% / 6.6% 다. LSTM 은 구간 절반이 0.99 아래인데
    중앙값만 보면 안 보인다.

    ※ 기존 *_R2seg 열(중앙값)은 S.split_metrics 가 내던 값 그대로 둔다.
      새 열만 이 파일에서 직접 구간을 갈라 계산한다.

★ 조건별 전압 RMSE(바이올린 그림용)도 여기서 낸다
    예전에는 figures_model.py 가 per_condition_rmse.csv 를 따로 만들었는데,
    그쪽은 figures_{모델}.pt(08-20 학습분, seed 하나)를 읽었다. 그래서 SI 그림
    (c) 와 이 표가 **서로 다른 학습 결과**를 보고 있었고, 실제로 LSTM 과
    Transformer 의 순위가 그림과 표에서 뒤집혔다. 표를 만드는 이 자리에서
    같이 내면 둘이 갈릴 수가 없다.

        per_condition_rmse.csv          조건별, seed 3개 평균 (54행/모델)
        per_condition_rmse_byseed.csv   조건별 x seed (162행/모델)

    py 2_surrogate/make_model_table.py
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

import numpy as np
import pandas as pd
import torch

import surrogate as S

OUT_DIR = S.RESULT_DIR      # SWB_RESULT_DIR 로 갈아끼운다 (260922)
CKPT = os.path.join(OUT_DIR, 'ar_models')
ORDER = ['MLP', 'XGBoost', 'LSTM', 'Transformer', 'TFT']
SEEDS = [0, 1, 2]
EPOCHS = 150


def get_model(name, seed, train, val, scaler):
    """저장된 가중치를 읽는다. XGBoost 도 부스터를 저장해 두고 다시 쓴다.

    XGBoost 도 부스터를 남긴다 — 매번 새로 학습하면 같은 seed 인데도 값이
    실행마다 달라진다(이유는 S.xgb_cached 에).
    """
    if name == 'XGBoost':
        return S.xgb_cached(seed, train, val, scaler, CKPT)
    p = os.path.join(CKPT, f'figures_{name}_s{seed}.pt')
    if not os.path.exists(p):
        return None
    m = S.build(name, epochs=EPOCHS, seed=seed)
    m.net = S.TORCH_MODELS[name](**m.kwargs).to(S.DEVICE)
    m.net.load_state_dict(torch.load(p, map_location=S.DEVICE))
    m.net.eval()
    return m


def main():
    pool = S.load(list(S.USE))
    train, val, te = S.split(pool)
    sc = S.Scaler().fit(train)
    naive = tuple(float(np.mean(np.concatenate(
        [np.abs(np.diff(r[c].to_numpy(float))) for r in train.values()])))
        for c in ('Vcell', 'c_mix'))

    rows, per = [], []
    for name in ORDER:
        for seed in SEEDS:
            m = get_model(name, seed, train, val, sc)
            if m is None:
                print(f'  [{name} s{seed}] 가중치 없음 — 건너뜀')
                continue
            E, O, EC, OC, ch = [], [], [], [], []
            seg = {0: [], 1: []}          # 구간별 R2 (0 전압, 1 농도)
            for k, r in te.items():
                p = S.to_c_mix(sc.back(m.predict(sc.forward(r))), r)
                o = r['Vcell'].to_numpy(float)
                oc_ = r['c_mix'].to_numpy(float)
                true2 = np.c_[o, oc_]
                for _, gg in r.groupby(['cycle', 'phase']):
                    ii = gg.index.to_numpy()
                    for jj in (0, 1):
                        vv = np.var(true2[ii, jj])
                        if vv < 1e-12:
                            continue
                        seg[jj].append(
                            1 - np.mean((p[ii, jj] - true2[ii, jj]) ** 2) / vv)
                E.append(p[:, 0] - o)
                O.append(o)
                EC.append(p[:, 1] - oc_)
                OC.append(oc_)
                ch += S.ceiling_error(p, r)[1]
                per.append(dict(model=name, seed=seed, cond=k,
                                v_rmse_mV=float(np.sqrt(np.mean(
                                    (p[:, 0] - o) ** 2))) * 1000))
            e, o = np.concatenate(E), np.concatenate(O)
            ec, oc = np.concatenate(EC), np.concatenate(OC)
            sm = S.split_metrics(m, te, sc)
            rows.append(dict(
                model=name, seed=seed,
                V_RMSE_mV=np.sqrt((e ** 2).mean()) * 1000,
                V_MAE_mV=np.abs(e).mean() * 1000,
                V_MAPE_pct=np.mean(np.abs(e / o)) * 100,
                c_RMSE_mM=np.sqrt((ec ** 2).mean()),
                c_MAE_mM=np.abs(ec).mean(),
                c_MAPE_pct=np.mean(np.abs(ec / oc)) * 100,
                V_R2seg=sm['Vcell']['r2_seg'], c_R2seg=sm['c_mix']['r2_seg'],
                ceil_h=float(np.median(ch)),
                V_MASE=np.abs(e).mean() / naive[0],
                c_MASE=np.abs(ec).mean() / naive[1],
                V_R2seg_p10=float(np.percentile(seg[0], 10)),
                c_R2seg_p10=float(np.percentile(seg[1], 10)),
                V_R2seg_lt99_pct=100.0 * float(np.mean(np.array(seg[0]) < 0.99)),
                c_R2seg_lt99_pct=100.0 * float(np.mean(np.array(seg[1]) < 0.99))))
            print(f'  [{name} s{seed}] V RMSE {rows[-1]["V_RMSE_mV"]:7.3f} mV'
                  f'   c RMSE {rows[-1]["c_RMSE_mM"]:7.3f} mM', flush=True)
    raw = pd.DataFrame(rows)

    # 파라미터 수 — 구조에서 직접 센다 (XGBoost 는 신경망이 아니라 해당 없음)
    par = {}
    for n in ORDER:
        if n == 'XGBoost':
            par[n] = np.nan
            continue
        m = S.build(n, epochs=1, seed=0)
        par[n] = sum(p.numel() for p in S.TORCH_MODELS[n](**m.kwargs)
                     .parameters() if p.requires_grad) / 1e3

    cols = [c for c in raw.columns if c not in ('model', 'seed')]
    mu = raw.groupby('model')[cols].mean().loc[ORDER]
    sd = raw.groupby('model')[cols].std().loc[ORDER]

    MAIN = [('V_RMSE_mV', 'V RMSE (mV)', 2), ('V_MAE_mV', 'V MAE (mV)', 2),
            ('V_MAPE_pct', 'V MAPE (%)', 3), ('c_RMSE_mM', 'c RMSE (mM)', 2),
            ('c_MAE_mM', 'c MAE (mM)', 2), ('c_MAPE_pct', 'c MAPE (%)', 3)]
    print(f'\n테스트 {len(te)}조건, seed {SEEDS} 평균 ± 표준편차\n')
    print(f'{"model":13}' + ''.join(f'{h:>21}' for _, h, _ in MAIN)
          + f'{"params (k)":>12}')
    for n in ORDER:
        line = ''.join(f'{mu.loc[n, c]:>13.{d}f} ±{sd.loc[n, c]:6.{d}f}'
                       for c, _, d in MAIN)
        p_ = f'{par[n]:>12.0f}' if not np.isnan(par[n]) else f'{"-":>12}'
        print(f'{n:13}{line}{p_}')

    TAIL = [('V_R2seg', 'V R2seg'), ('c_R2seg', 'c R2seg')]
    print(f'\n[R2_seg 꼬리]  {"":6}' + ''.join(
        f'{h + " 중앙":>16}{h + " p10":>16}{h + " <0.99%":>16}' for _, h in TAIL))
    for n in ORDER:
        line = ''.join(
            f'{mu.loc[n, k]:>16.4f}{mu.loc[n, k + "_p10"]:>16.4f}'
            f'{mu.loc[n, k + "_lt99_pct"]:>15.1f}%' for k, _ in TAIL)
        print(f'{n:13}{line}')

    out = pd.DataFrame(index=mu.index)
    for c in cols:
        out[c + '_mean'] = mu[c]
        out[c + '_sd'] = sd[c]
    out['params_k'] = [par[n] for n in ORDER]

    # 조건별 RMSE — 바이올린 그림(SI (c), performance_summary)이 읽는다.
    # 그림 쪽은 열 이름 v_rmse_mV 를 그대로 쓰므로 seed 평균을 그 이름에 담고,
    # seed 별 원자료는 _byseed 로 따로 남긴다.
    pc = pd.DataFrame(per)
    if len(pc):
        pcm = (pc.groupby(['model', 'cond'], sort=False).v_rmse_mV.mean()
               .reset_index())
        for nm, d_ in (('per_condition_rmse.csv', pcm),
                       ('per_condition_rmse_byseed.csv', pc)):
            try:
                d_.to_csv(os.path.join(OUT_DIR, nm), index=False,
                          encoding='utf-8-sig')
                print(f'saved: {nm}')
            except PermissionError:
                print(f'[!!] {nm} 저장 실패 — 엑셀에서 닫고 다시 돌릴 것')

    for path in (os.path.join(OUT_DIR, 'model_table_3seed.csv'),
                 os.path.join(OUT_DIR, 'model_table_3seed_new.csv')):
        try:
            out.to_csv(path, encoding='utf-8-sig')
            raw.to_csv(path.replace('.csv', '_byseed.csv'), index=False,
                       encoding='utf-8-sig')
            print(f'\nsaved: {os.path.basename(path)}')
            return
        except PermissionError:
            continue
    print('\n[!!] CSV 저장 실패 — 엑셀에서 닫고 다시 돌릴 것')


if __name__ == '__main__':
    main()
