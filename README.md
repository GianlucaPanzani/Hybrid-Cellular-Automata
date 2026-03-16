# Hybrid Cellular Automata for Avascular Tumour Growth

## 1. What this project is about
This project implements a hybrid **Cellular Automaton (CA) + PDE** model for avascular tumour growth.
It combines:
- a discrete lattice of tumour cells (cell states and local decisions),
- a continuous oxygen field updated by reaction-diffusion dynamics,
- cell-level decision logic based on a small feed-forward Artificial Neural Network (ANN).

The repository includes simulation code, ANN training/grid-search notebooks, reproducible experiments, and a presentation discussing both:
- Gerlee & Anderson (2007): *An evolutionary hybrid cellular automaton model of solid tumour growth*,
- Messina et al. (2023): *Automated hybrid cell model* (used mainly for conceptual comparison in presentation material).

## 2. Project goal
Main goal: **replicate and study the oxygen-driven growth/evolution dynamics** from Gerlee & Anderson (2007), then build a clean, inspectable implementation for experiments.

Concretely, the project aims to:
- reproduce growth trends under different oxygen boundary conditions (`c0`, `c0/2.5`, `c0/10`),
- model clonal evolution through ANN mutation at cell division,
- measure growth and morphology proxies (total cells, invasive distance, necrotic fraction),
- evaluate clonal heterogeneity (Shannon index),
- run both single-run and multi-seed simulations (with mean/std aggregation).

## 3. Data structures used
The implementation is mostly NumPy-based for performance and clarity.

- `Params` (`dataclass`): all simulation hyperparameters (lattice size, time step, oxygen parameters, mutation rates, thresholds, RNG seed, etc.).
- Lattice/state arrays (`np.ndarray`):
  - `state: (N, N), uint8` for cell state labels (`EMPTY`, `PROLIFERATING`, `QUIESCENT`, `NECROTIC`, `DEAD`),
  - `c: (N, N), float32` oxygen field,
  - `age_hours`, `prolif_age_hours: (N, N), float32` per-cell timing variables.
- ANN genotype container:
  - `ann: (N, N), dtype=object`, where each alive site stores an `ArtificialNN` object.
- Per-step helper maps:
  - `action_map`, `F_map`, uptake fields (`alpha`) as NumPy arrays.
- Time-series outputs:
  - dictionaries of 1D arrays (`alive`, `proliferating`, `necrotic`, `dead`, `invasive`, `shannon`, etc.).
- Experiment summaries:
  - `pandas.DataFrame` in notebooks for aggregated metrics and comparison tables.

## 4. Code implementation
### Core modules
- `simulation/model.py`
  - State constants and PDE helpers:
    - `laplacian_5pt`, `oxygen_step_explicit`, boundary handling.
  - `Params` dataclass: centralized simulation configuration.
  - `ArtificialNN`:
    - forward pass (`[P, Q, A]` outputs),
    - optional supervised fitting,
    - mutation operators (`mutate_inplace`, `mutated_copy`) for inherited variability.
  - `SimulationModel`:
    - initializes the lattice and tumour seed,
    - applies stochastic CA updates each step (`step`),
    - couples cell actions to oxygen uptake and PDE update,
    - collects trajectories (`run`), including invasiveness and Shannon diversity.

- `simulation/utils.py`
  - `.env` parsing and typed parameter loading,
  - ANN parameter loader from `.env_ann_params`,
  - synthetic dataset generator for ANN bootstrapping,
  - plotting and animation helpers (`plot_stability_curve`, `create_animation`),
  - utility metrics (`eval_simulation`).

### Notebooks
- `simulation/ann_training.ipynb`
  - bootstraps ANN behavior from rule-based labels,
  - evaluates predictions (metrics/confusion matrix/ROC),
  - exports ANN parameters.
- `simulation/grid_search_ann_params.ipynb`
  - scans ANN parameter combinations,
  - stores scores in `ann_grid_search_results.csv`.
- `simulation/sim.ipynb`
  - single-run simulation with dynamics and qualitative plots.
- `simulation/sim_multiple_seed.ipynb`
  - repeated runs across multiple random seeds,
  - aggregated curves and error bars (mean ± std), aligned with paper-style stochastic evaluation.

### Configuration files
- `simulation/.env_*`: predefined tumour aggressiveness setups.
- `simulation/.env_ann_params`: ANN matrices/vectors (`w`, `W`, `theta`, `phi`, output order).

## 5. Results (current state)
The implemented model reproduces the expected qualitative behavior reported in the 2007 paper:
- lower oxygen conditions produce earlier hypoxia/necrosis and more irregular (fingered) growth fronts,
- higher oxygen supports larger, more compact tumours with faster overall growth,
- mutations introduce evolutionary variability and affect clone diversity over time.

From the simulation notebooks:
- growth dynamics are tracked through invasive distance and total tumour cell counts,
- tumour size vs oxygen at fixed time can be compared with variability bars,
- Shannon index captures temporal changes in clonal heterogeneity,
- multi-seed runs improve robustness by separating trend from stochastic noise.

## Repository structure (main files)
```text
Hybrid-Cellular-Automata/
├── papers/
│   ├── hybrid-cellular-automaton-for-tumour-growth.pdf
│   └── cancers-15-05660.pdf
├── simulation/
│   ├── model.py
│   ├── utils.py
│   ├── ann_training.ipynb
│   ├── grid_search_ann_params.ipynb
│   ├── sim.ipynb
│   ├── sim_multiple_seed.ipynb
│   ├── ann_grid_search_results.csv
│   └── .env_ann_params
└── presentation/
    └── presentation.tex
```

## 6. How to run the code
### Prerequisites
- Python 3.10+ (tested in a modern CPython environment)
- `pip`
- Jupyter Notebook or JupyterLab

### Environment setup
From the project root:

```bash
cd Hybrid-Cellular-Automata
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyter ipython
```

### Run notebooks (recommended workflow)
```bash
cd simulation
jupyter lab
```

Then execute notebooks in this order:
1. `ann_training.ipynb`
   - trains/validates the ANN behavior and can be used to update ANN parameters.
2. `grid_search_ann_params.ipynb`
   - explores ANN parameter combinations and saves scores to `ann_grid_search_results.csv`.
3. `sim.ipynb`
   - runs a single-seed simulation and plots growth/oxygen/diversity trajectories based on the parameters stored in `.env_ann_params`.
4. `sim_multiple_seed.ipynb`
   - runs multiple seeds (paper-style stochastic averaging) and reports mean ± std.

### Configuration notes
- ANN parameters are loaded from `simulation/.env_ann_params`.
- Alternative tumour parameter presets are available in:
  - `.env_low-aggressive-tumour`
  - `.env_medium-aggressive-tumour`
  - `.env_highly-aggressive-tumour`
