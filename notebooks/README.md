# GRNO research notebooks

These notebooks implement the same GRNO operator blueprint with a common trainable core within each spatial rank. Each task is trained separately; the task adapters compute the parameter-free carrier, routing map, boundary handling, and output projection.

| Notebook | Task keys |
| --- | --- |
| [GRNO_2D_NS_SQG_VP.ipynb](GRNO_2D_NS_SQG_VP.ipynb) | `kolmogorov_ns`, `dissipative_sqg`, `vlasov_poisson` |
| [GRNO_3D_KellerSegel_PorousConvection.ipynb](GRNO_3D_KellerSegel_PorousConvection.ipynb) | `keller_segel3d`, `porous_convection3d` |

The release preserves the original code cells and their defaults, while removing saved outputs and execution counts. It includes training, validation, rollout evaluation, and same-weight route interventions. It does not include baseline notebooks, MHD experiments, or trained checkpoints. Dataset generation is documented separately in [data/README.md](../data/README.md).

## Environment

Use Python 3.10 or newer with PyTorch 2.x and the packages in [requirements.txt](../requirements.txt). Install a PyTorch build compatible with your CUDA setup, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

The dependency file lists imported packages rather than a validated, pinned reproduction environment. CUDA is required by `PROFILE="pilot"` and `PROFILE="paper"`. Both use bfloat16 autocast, so select a GPU and PyTorch build that support it. The 3D notebook uses activation checkpointing and micro-batch accumulation for volumetric data. CPU use is limited to `PROFILE="smoke"`, which still requires valid datasets and can be slow.

The first 3D code cell retains `os.environ['CUDA_VISIBLE_DEVICES'] = '1'` from the research environment. Change this to a GPU available on your machine, or remove it to honor your launcher or scheduler's device allocation, **before importing PyTorch**. Restart the kernel after changing GPU visibility.

## Configure before running

Open one notebook and edit its first code cell before running the remaining sections in order. By default, it launches the `paper` profile for every supported task and seed `0`, with training, testing, and all route interventions enabled.

| Setting | Purpose |
| --- | --- |
| `PROFILE` | `smoke`, `pilot`, or `paper`; selects the corresponding `RunConfig`. |
| `TASKS_TO_RUN` | Tuple containing one or more supported task keys. |
| `SEEDS` | Default `(0,)`; use `(0, 1, 2)` for three independently trained seeds. |
| `DATA_ROOT_OVERRIDES` | Maps each task key to its canonical dataset directory. |
| `RUN_TRAINING` | Train when `True`; otherwise load an existing `best.pt` at the computed path. |
| `RUN_TESTING` | Run test rollout evaluation when `True`. |
| `RUN_ROUTE_INTERVENTIONS` | Evaluate `correct`, `zero`, `reverse`, and `shuffle`; `False` evaluates only `correct`. |
| `RESUME_IF_AVAILABLE` | Resume a compatible existing `last.pt`. |

For a small setup check, set `PROFILE="smoke"`, choose one task, and disable route interventions. Smoke mode changes model size, training duration, and rollout horizon; use it to check the pipeline, not to report scientific performance.

For example, configure the 2D notebook with an existing NS dataset:

```python
PROFILE = "smoke"
TASKS_TO_RUN = ("kolmogorov_ns",)
SEEDS = (0,)
DATA_ROOT_OVERRIDES = {
    "kolmogorov_ns": "/absolute/path/to/kolmogorov_ns_paper_dataset",
}
RUN_ROUTE_INTERVENTIONS = False
```

The notebooks infer `ROOT` from their working directory and recognized legacy data directories. If datasets are stored elsewhere, use the overrides and assign `ROOT = Path("/absolute/path/to/GRNO")` immediately after `ROOT = locate_project_root()` in the first code cell. This keeps outputs in the intended project directory even when Jupyter starts in `notebooks/`.

Each selected task needs its **train, val, and test** splits, including the checked manifests and normalization files. The notebook validates canonical PDE parameters even in smoke mode. It does not create dummy trajectories or automatically download data. Paper and pilot testing require at least 41 frames per trajectory for the 40-step forecast.

## Preserved training profiles

The table documents the shipped code, which is the shared-core training realization. Check the manuscript's experiment-specific protocol before treating these defaults as an exact reproduction of every reported comparison.

| Notebook/profile | Epochs | Samples/epoch | Batch | Width | Train/val/test rollout |
| --- | ---: | ---: | --- | ---: | --- |
| 2D `paper` | 100 | 2048 | 4 | 48 | 4 / 10 / 40 |
| 2D `pilot` | 20 | 512 | 4 | 48 | 4 / 10 / 40 |
| 2D `smoke` | 2 | 16 | 2 | 16 | 2 / 3 / 5 |
| 3D `paper` | 100 | 512 | micro 1, accumulation 4 | 24 | 4 / 10 / 40 |
| 3D `pilot` | 20 | 128 | micro 1, accumulation 4 | 24 | 4 / 10 / 40 |
| 3D `smoke` | 2 | 4 | micro 1, accumulation 2 | 8 | 2 / 3 / 5 |

Paper/pilot profiles use AdamW with learning rate `3e-4`, weight decay `1e-4`, cosine scheduling, normalized state MSE plus gradient and spectral terms, and autoregressive four-step training. The full settings remain visible in each notebook's `RunConfig`.

## Outputs and interpretation

The main output directories are:

- `ROOT/outputs/shared_grno2d/<profile>/shared-grno2d-direct-route-v1/`
- `ROOT/outputs/shared_grno3d/<profile>/shared-grno3d-direct-route-v1/`

| File/location | Contents |
| --- | --- |
| `<task>/seed_<seed>/<signature>/best.pt` | Checkpoint selected by mean validation relative L2 over the validation rollout. |
| `<task>/seed_<seed>/<signature>/last.pt` | Latest checkpoint, optimizer/scheduler state, and RNG state for resumption. |
| `<task>/seed_<seed>/<signature>/history.csv` | Per-epoch training loss, validation errors, learning rate, and elapsed time. |
| `all_step_results.csv` | Per-trajectory, per-step metrics for each task, seed, and route mode. |
| `final_per_seed.csv` | Final-step metrics averaged over test trajectories within each seed. |
| `final_aggregate_across_seeds.csv` | Mean and sample standard deviation of the per-seed final-step averages. |
| `ROOT/manifests/shared_grno{2d,3d}_K4_v1/<task>/windows_seed_<seed>_<signature>.npz` | Deterministic training-window references. |

Use `route_mode="correct"` for normal prediction results. Relative L2, relative H1, and spectral error are lower-is-better; correlation is higher-is-better. `skill_vs_persistence` compares prediction against repeating the initial field. The route interventions reuse the same trained model and test the effect of changing the departure map.

The compact diagnostic plots display the mean relative L2 curve across trajectories and seeds. The final CSV summaries describe the last rollout step, rather than the mean error over the complete horizon. With the default single seed, the across-seed sample standard deviation is undefined (`NaN`). Full-field 2D samples and center-plane 3D samples are also available in memory as `SAMPLE_ROLLOUTS`.

Checkpoint signatures include configuration, data-contract information, normalization sources, and dataset paths. Moving data can therefore change the computed checkpoint directory. For evaluation without training, ensure a matching checkpoint exists at the path produced by the current configuration.

## Optional 2D figure export

The final nonempty cell of the 2D notebook exports a seed-0 NS rollout to `kolmogorov_grno_seed0.npz`. It retains the original absolute `PANEL_B_CACHE` path; edit it to a writable directory before running this cell. It assumes NS is included in `TASKS_TO_RUN`, seed `0` is included in `SEEDS`, a compatible trained checkpoint exists, and the test trajectory has at least 41 frames. Skip this cell for a non-NS or smoke-only run. Its baseline branch expects a separate baseline notebook, which is not included in this release.

The release was checked with nbformat validation, Python AST parsing, and exact comparison against the original code cells. Training and dataset-dependent evaluation were not executed during release preparation.
