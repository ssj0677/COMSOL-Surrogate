# COMSOL-Surrogate

Code for physics-assisted surrogate modeling and dynamic optimization of seawater battery desalination (SWB-D).

Pipeline: COMSOL simulations -> training data -> TFT surrogate model -> NSGA-II optimization of step-current protocols.

Only the code is provided here. The COMSOL model (`.mph`), the simulation dataset and trained weights are not included.

## Layout

| Folder / file | Role |
|---|---|
| `1_extract/` | Generates operating conditions and runs the COMSOL model to collect charge-discharge data (needs COMSOL and the `.mph`). `comsol_setup.py` holds the settings; `_D_swb_core.py` is the COMSOL driver. |
| `2_surrogate/` | `surrogate.py` (data loading, MLP/XGBoost/LSTM/Transformer/TFT, training, metrics), `model_tft_re.py` (direct TFT), `model_tft_ar.py` (autoregressive TFT), `train_seq_seeds.py`, `make_model_table.py` (Table 1), `plot_seq_metrics.py` (Table 2), `compare_ar_direct.py`, `dump_training_set.py`. |
| `3_optimize/` | `reopt600.py` (3-objective NSGA-II: salt removal, operating time, net energy; 6 current magnitudes over 3 cycles), `term_corr.py` (end-of-charge and final-concentration correction), `optimize_joint3.py`, `optimize_nsga2.py`, `export_cycle_table.py`, `plot_pareto.py`. |
| `plotstyle.py` | Shared matplotlib style. |

## Setup

```
pip install -r requirements.txt
```

Paths are relative to the repository root. Datasets are expected in `_E_<water>_<protocol>_v49/` folders under the root, and results are written to `_S_result/` and `_O_result/`. Set `SWB_MPH` to the COMSOL model file and `SWB_RESULT_DIR` to change the result folder.

## Usage

```
python 2_surrogate/train_seq_seeds.py
python 2_surrogate/make_model_table.py
python 2_surrogate/compare_ar_direct.py
python 3_optimize/reopt600.py base opt cmp
```

Each script's docstring (Korean) describes its inputs, outputs and options.

## Data

The simulation dataset is not distributed with the code. Please contact the authors.

## License

MIT (see `LICENSE`).
