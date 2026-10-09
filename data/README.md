# Dataset generators

This folder contains data-only extracts of the archived GRNO experiment notebooks. The generated arrays, checkpoints, and simulation results are not included in the repository. These notebooks generate targets independently of the neural operator.

| Notebook | Dataset | Original saved default | Original reference solver |
|---|---|---|---|
| [generate_ns_sqg.ipynb](generate_ns_sqg.ipynb) | Kolmogorov Navier–Stokes and dissipative SQG | `pilot` | 2/3-dealiased pseudo-spectral ETDRK2 |
| [generate_vlasov_poisson.ipynb](generate_vlasov_poisson.ipynb) | 1D1V Vlasov–Poisson | `paper` | Split-spectral phase-space shifts and Fourier Poisson field |
| [generate_keller_segel3d.ipynb](generate_keller_segel3d.ipynb) | 3D screened Keller–Segel | `smoke` | 2/3-dealiased full-complex Fourier ETD2 |
| [generate_porous_convection3d.ipynb](generate_porous_convection3d.ipynb) | 3D one-sided Darcy convection | `paper` | Finite-volume advection, slab Green map, and Strang/RK2 integration |

## Generate the paper-format data

1. Install the repository dependencies in a Python kernel with a compatible PyTorch build. The generators do not install packages automatically. The original `pilot` and `paper` profiles require CUDA; the source runs were intended for one A100.
2. Start Jupyter from the repository, or set `GRNO_ROOT` to an absolute project path before starting the kernel. Each generator prints its chosen root. Data locations are derived from this root; the numerical settings and fingerprints are unchanged.
3. Select `PROFILE = "paper"` in each generator for the split/grid contract below. Restart the kernel and run from the top. The saved defaults in the table above are retained from the source and do not all generate paper-sized data.
4. Keep `RUN_DATA_GENERATION = True` for generation or resume. Set it to `False` only when all required split arrays are already complete; the simulator audits and normalization cells still run.
5. Read the printed preflight and generation diagnostics before using the data. A completed execution of these extracts has not been performed in this release environment; the release was checked statically.

The `paper` branches match the manuscript's Table 6 grid, split counts, stored-frame counts, and frame intervals. This confirms the code contract; it is not a claim that the released files reproduce trained checkpoints or reported errors.

| Dataset | Stored grid | Train: trajectories × stored frames | Validation | Test | Frame interval | Approximate stored payload |
|---|---:|---:|---:|---:|---:|---:|
| Kolmogorov NS | 128 × 128 | 64 × 51 | 8 × 101 | 12 × 101 | 0.1 | 0.328 GiB |
| Dissipative SQG | 128 × 128 | 64 × 51 | 8 × 101 | 12 × 101 | 0.1 | 0.328 GiB |
| Vlasov–Poisson | 128 × 128 in `(v,x)` | 64 × 101 | 8 × 121 | 12 × 161 | 0.1 | 0.572 GiB |
| Keller–Segel | 64 × 64 × 64 | 64 × 51 | 8 × 101 | 12 × 101 | 0.25 | 5.160 GiB |
| Porous convection | 64 × 64 × 64 | 64 × 101 | 8 × 101 | 12 × 101 | 0.1 | 8.285 GiB |

Payloads use `float32` and exclude file headers, JSON, CSV diagnostics, intermediate memory, and checkpoints. NS/SQG totals include their fixed-forcing arrays; VP includes the small regime-label arrays. The five target datasets together need about **14.672 GiB** of stored payload. Generation additionally requires solver memory and working space. Porous convection uses a **128³ reference grid** and volume-averages each frame to **64³**; 64³ is the stored/model grid, not the reference grid.

**Frame-count convention:** NS/SQG's `train_frames=50`, `val_frames=100`, and `test_frames=100` count transitions; the generator saves the initial state too. Keep those values to obtain 51/101/101 states. Keller–Segel and porous convection explicitly use `*_transitions` and also save the initial state. VP's `*_frames` counts stored states directly, so its paper values are 101/121/161.

## Files produced and reader compatibility

| Dataset | Root relative to the generator's `ROOT` | Produced files required by the released training reader |
|---|---|---|
| NS/SQG | `grno_ns_sqg_data/<task>_paper_<fingerprint>/` | `train/val/test_states.npy`, `train/val/test_forcing.npy`, `manifest.json`, `normalization.json` |
| VP | `grno_mhd_vp_data/vlasov_poisson_paper_<fingerprint>/` | `train/val/test_states.npy`, `manifest.json`; `train/val/test_regime.npy` also supplied |
| Keller–Segel | `data/keller_segel3d/paper/` | `train/val/test/trajectory_XXXX.npy`, `metadata.json`, one `training_stats_<signature>.json` |
| Porous convection | `data/porous_convection3d/paper/<signature>/` | `train/val/test/trajectory_XXXX.npy`, `dataset_manifest.json`, `train_statistics.json`; trajectory diagnostics and `dataset_readiness.csv` also supplied |

NS/SQG arrays have shape `[trajectory,time,y,x]`, with forcing `[trajectory,y,x]`. VP arrays have shape `[trajectory,time,1,v,x]`; regime 0 is Landau and regime 1 is two-stream. Each 3D trajectory is stored separately with shape `[time,1,z,y,x]`.

The generators retain dataset version identifiers checked by the released training notebooks:

- NS/SQG: `active_scalar_etdrk2_v3`.
- VP: `grno_mhd_vp_custom_pilot_v2` (the legacy version string is retained although this extract generates only VP).
- Keller–Segel: `screened-keller-segel3d-etd2-v1`.
- Porous convection: `porous-convection3d-fv-strang-rk2-continuum-ic-v2`.

The released [2D training notebook](../notebooks/GRNO_2D_NS_SQG_VP.ipynb) consumes NS/SQG normalization JSON and estimates VP normalization deterministically from training arrays with seed 17. The released [3D training notebook](../notebooks/GRNO_3D_KellerSegel_PorousConvection.ipynb) consumes the generated state/delta statistics and estimates its carrier scale from training data.

Those training notebooks preserve their original root-discovery code. Set their `ROOT` explicitly to the same absolute location printed by the generators. If automatic discovery finds multiple datasets, paste the exact dataset directories into `DATA_ROOT_OVERRIDES` in the training configuration cell. These examples show all tasks; replace every placeholder with the printed path/hash:

```python
# In GRNO_2D_NS_SQG_VP.ipynb, set ROOT before the data-discovery cells.
ROOT = Path("/absolute/path/to/GRNO")
DATA_ROOT_OVERRIDES = {
    "kolmogorov_ns": ROOT / "grno_ns_sqg_data/kolmogorov_ns_paper_<fingerprint>",
    "dissipative_sqg": ROOT / "grno_ns_sqg_data/dissipative_sqg_paper_<fingerprint>",
    "vlasov_poisson": ROOT / "grno_mhd_vp_data/vlasov_poisson_paper_<fingerprint>",
}
```

```python
# In GRNO_3D_KellerSegel_PorousConvection.ipynb, set ROOT before discovery.
ROOT = Path("/absolute/path/to/GRNO")
DATA_ROOT_OVERRIDES = {
    "keller_segel3d": ROOT / "data/keller_segel3d/paper",
    "porous_convection3d": ROOT / "data/porous_convection3d/paper/<signature>",
}
```

Run generator and training configuration from the top after editing roots. Generated paths/configuration are part of recorded contracts, so do not rename or mix files from different profiles. Keller statistics discovery expects exactly one `training_stats_*.json` in the chosen directory.

## What was extracted

The release preserves the solver algorithms, physical configurations, split seeds, primary simulator preflights, array layouts, and needed statistics code. Neural models, optimization, checkpoint evaluation, training-window manifest creation, and model-result plotting are omitted. Legacy `RunConfig` classes retain some inactive training fields: NS/VP write these fields into the source manifest; porous data audits also use the original rollout horizon. No training function remains in these extracts.

| Released notebook | Archived source | Selected original code cells (zero-based) |
|---|---|---|
| `generate_ns_sqg.ipynb` | `GRNO_Kolmogorov_NS_and_SQG_128(1).ipynb` | 3, 4, 6, 7, 8, 11, 12 |
| `generate_vlasov_poisson.ipynb` | `GRNO_MHD_and_VlasovPoisson_128 (1).ipynb` | 3, 4, 8, 10, 12 |
| `generate_keller_segel3d.ipynb` | `GRNO_KellerSegel3D_64.ipynb` | 4, 5, 7, 9, 11, 13 |
| `generate_porous_convection3d.ipynb` | `GRNO_PorousConvection3D_Ra1000_64 (3).ipynb` | 4, 5, 7, 9, 11, 13, 15 |

Selection used Python AST boundaries, preserving whole solver functions and class bodies. VP's original shared simulator-audit, generation, cache-validation, and manifest statements were narrowed to their existing VP branches. The original unconditional MHD generation was removed; simply setting `TASKS` in that source would not have done this. No MHD arrays or MHD solver code are bundled. The source VP solver performs Fourier velocity shifts while recording the finite velocity boundary as nonperiodic; its existing boundary-tail audit is retained, without changing the solver's boundary algorithm.

The portable dependency/root cell replaces the original imports and package-install cells. It inherits GPU visibility from the running kernel and does not hardcode the original VP GPU index. Data directories are relocated under `ROOT`; paths are excluded from the retained numerical fingerprints. All notebook outputs and execution counts are cleared. Source hashes, selected cells, and detailed extraction notes are recorded in [extraction_manifest.json](extraction_manifest.json) and notebook metadata.

The porous source's later optional spatial/temporal refinement audit cell 29 is not included in this compact extract. Its original primary physics preflight and saved-dataset readiness gate are included. This release does not claim that the omitted optional audits were run.

## Source behavior when resuming

These extracts retain the original cache and acceptance logic. NS/SQG reuse cached arrays based on file presence, so reusing externally modified or partial files requires checking their shape and completeness. Keller–Segel performs its final mass-drift gate after atomically renaming the trajectory; if generation fails at that gate, inspect the rejected file before resuming, since its reuse check validates schema. Use a new data location when changing physical settings, and do not treat a failed generation or audit as an accepted dataset.

## Release validation

All four notebooks parse as valid JSON/Python; static symbol-table checks find no unresolved global dependencies. Retained solver classes, generation functions, normalization functions, and primary audit functions match their source ASTs, except the explicitly scoped VP-only audit. No simulations were executed for this release because the validation environment lacked PyTorch. Numerical runtime, GPU memory, and full-dataset acceptance remain to be checked in a suitable kernel.
