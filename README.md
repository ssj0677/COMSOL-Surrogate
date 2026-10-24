# COMSOL-Surrogate

Code for physics-assisted surrogate modeling and dynamic optimization of seawater battery desalination (SWB-D).

Pipeline: COMSOL simulations -> training data -> TFT surrogate model -> NSGA-II optimization of step-current protocols.

The repository holds the code and the COMSOL-generated datasets (`data/`). The COMSOL model (`.mph`) and trained weights are not included.

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

Paths are relative to the repository root. Training data are read from `_E_regrid/_E_seawater_<set>_v49_grid5/` under the root, and results are written to `_S_result/` and `_O_result/`. Set `SWB_MPH` to the COMSOL model file and `SWB_RESULT_DIR` to change the result folder.

## Usage

```
python 2_surrogate/train_seq_seeds.py
python 2_surrogate/make_model_table.py
python 2_surrogate/compare_ar_direct.py
python 3_optimize/reopt600.py base opt cmp
```

Each script's docstring (Korean) describes its inputs, outputs and options.

## Data

The training data and result tables are in `data/` (see `data/README.md`). Unzip the four `_E_seawater_*_v49_grid5.zip` archives in the repository root before running the scripts in `2_surrogate/`. The `.mph` model and trained weights are not distributed; please contact the authors.

## License

MIT (see `LICENSE`).
