# Data

COMSOL simulation data used to train and evaluate the surrogate models, and the optimization results reported in the manuscript.

## Training data (COMSOL, 5-min grid)

| File | Content |
|---|---|
| `_E_seawater_lhs6_v49_grid5.zip` | Latin-hypercube conditions; current differs in every cycle (201 files) |
| `_E_seawater_vstart_v49_grid5.zip` | Conditions starting from a partially desalinated state (201 files) |
| `_E_seawater_ratio_v49_grid5.zip` | High discharge-to-charge current ratio (20 files) |
| `_E_seawater_ratio2_v49_grid5.zip` | Revised high-ratio set (121 files) |

Each zip holds one `cond_XXX.csv` per operating condition, with columns `t` (min), `t_cont`, `cycle`, `phase` (charge/discharge), `cond_id`, `c_in` (feed concentration, mol/m3), `I_applied` (mA), `Q_cum`, `Vcell` (V), `cNa_C2`, `cCl_C2`, `nNa`, `c_tank`, `c_mix` (mol/m3).

Unzip all four archives in the repository root. They extract to `_E_regrid/_E_seawater_<set>_v49_grid5/`, which is where `2_surrogate/surrogate.py` reads them.

The four sets contain 539 conditions in total; 355 were retained after the 4.08 V screening.

## Tables

| File | Content |
|---|---|
| `dataset_conditions.csv` | All 539 conditions: set, feed concentration, currents, end-of-charge voltages, and whether the condition was used for training (`used`) |
| `dataset_phases.csv` | Per-phase (charge/discharge) summary of the 539 conditions |
| `training_set_conditions.csv` | The 355 retained conditions |
| `step_current_protocols_229.csv` | 229 COMSOL-simulated step-current protocols compared with constant-current operation (Fig. 7, Fig. S8) |
| `optimized_protocols_17conc.csv` | Minimum-time, minimum-energy and selected protocols at the 17 feed concentrations, with changes relative to constant-current operation (Figs. 9-10, Table S5) |

Concentrations are in mol/m3 (= mM), currents in mA, energies in mWh and times in h.
