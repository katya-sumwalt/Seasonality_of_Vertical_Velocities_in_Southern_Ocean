# Seasonality_of_Vertical_Velocities_in_Southern_Ocean
[![DOI](https://zenodo.org/badge/1316508562.svg)](https://doi.org/10.5281/zenodo.22030505)

Code for Sumwalt, K. & Siegelman, L., *Seasonality of deep-reaching submesoscale vertical velocities and vertical heat
transport in the Southern Ocean* (Geophysical Research Letters, revised version).

Submesoscale vertical velocities w are diagnosed from southern elephant seal CTD profiles with a 2-D quasi-geostrophic
omega equation forced by the along-track buoyancy gradient and the NeurOST surface strain. All results use the upper
boundary condition w(15 m) = w_m, the mixing-driven vertical velocity of Ponte et al. (2013, eq. 5).

## Data

| Data | Source | Used for |
|---|---|---|
| Seal CTD profiles | MEOP-SMS submesoscale database, http://www.meop.net/database/meop-databases/meop-sms-database-submesoscale.html | all |
| Sea surface height, strain and vorticity | NeurOST SSH-SST L4 v2024.0, https://doi.org/10.5067/NEURO-STV24 | forcing of the omega equation, Fig. 1a, Rossby number |
| Bathymetry | RTopo-2.0.1 30 arc-second bedrock topography (Schaffer et al. 2016, https://doi.org/10.1594/PANGAEA.856844) | Fig. 1 |

All paths are set in [`config.py`](config.py). By default inputs are read from `data/` and figures are written to
`figures/` inside the repository; each path can be overridden with an environment variable
(`SES_DATA_DIR`, `SES_SEASONAL_DIR`, `SES_DW_WM_DIR`, `SES_DW_STRAIN_DIR`, `SES_RAW_SEAL_FILE`, `SES_ALL_PROFILES_FILE`,
`SES_NEUROST_DIR`, `SES_STRAIN_CACHE`, `SES_RTOPO_FILE`, `SES_MLD_FILE`, `SES_OUT_DIR`).

## Processing pipeline

1. `computing_vertical_velocities/cleanup_seal_data.py`, `compute_w.py`, `compute_w_main.py`: mask large |b_x|, extend
   the profiles to 700 m, trim the track ends and solve the omega equation with w(15 m) = 0 on 1 km sections. This
   produces the seasonal files `seal_data_with_vertical_velocity_east_<season>_*.nc` (`config.SEASON_FILES`).
2. `computing_vertical_velocities/compute_w_surface_bc.py`: omega solver with a prescribed w at 15 m, and w_m.
3. `python -m computing_vertical_velocities.compute_dw_mixing_bc <season>`: by linearity, w with w(15 m) = w_m is
   w(15 m = 0) + dw, where dw solves the unforced equation with w(15 m) = w_m (writes `dw_<season>.nc`).
4. `python -m computing_vertical_velocities.compute_dw_strain_profile <A|B|C> <season>`: dw for a depth-dependent strain
   σ(d) = σ(0) F(d), cases A, B, C (writes `dw_<case>_<season>.nc`; sensitivity tests only).
5. `python -m analysis.mld_comparison`: de Boyer Montégut (2004) and Holte & Talley (2009) mixed layer depths along the
   Figure 2 section (writes `config.MLD_FILE`, used in Figs. 2, S2, S4 and S8).
6. Statistics and figures (below). Shared code: `analysis/statistics.py` (confidence intervals, PDFs, conditional means,
   climatology, L-kurtosis), `analysis/bootstrap.py` (block bootstrap and downsampling to the fall sampling),
   `analysis/joint_pdf.py` (joint PDF and frequency/strength decomposition), `plotting_figures/plot_utils.py`.

## Figures and tables

| Paper | Script (`python -m ...`) | Output in `figures/` |
|---|---|---|
| Fig. 1 | `plotting_figures.fig1_study_region` | `fig1_study_region` |
| Fig. 2 | `plotting_figures.fig2_section` | `fig2_section` |
| Fig. 3 | `plotting_figures.fig3_seasonal_statistics` | `fig3_seasonal_statistics` |
| Fig. 4 | `plotting_figures.fig4_joint_pdf` | `fig4_joint_pdf` |
| Fig. S1 | `plotting_figures.figS1_lkurtosis` | `figS1_lkurtosis` |
| Fig. S2 | `plotting_figures.figS2_upper_bc_section` | `figS2_upper_bc_section` |
| Fig. S3 | `plotting_figures.fig3_seasonal_statistics` | `figS3_upper_bc_difference` |
| Fig. S4 | `plotting_figures.figS4_strain_section` | `figS4_strain_section` |
| Fig. S5 | `plotting_figures.figS5_strain_statistics` | `figS5_strain_statistics` |
| Fig. S6 | `plotting_figures.figS6_S7_bootstrap` | `figS6_obs_vs_depth` |
| Fig. S7 | `plotting_figures.figS6_S7_bootstrap` | `figS7_bootstrap_statistics` |
| Table S2 | `plotting_figures.figS6_S7_bootstrap` (or `analysis.bootstrap`) | printed, and `tableS2_bootstrap_layers.npz` |
| Fig. S8 | `plotting_figures.figS8_salinity_section` (also prints the b' = a CT' + c SA' fit) | `figS8_salinity_section` |
| Fig. S9 | `plotting_figures.fig3_seasonal_statistics` | `figS9_conditional_means` |
| MLD comparison | `analysis.mld_comparison` | `mld_comparison` |
| Rossby number | `analysis.rossby_number` | printed |
| Deep fall rms w | `analysis.deep_fall_rms` | printed |

Figures are saved as `.png` and `.pdf`. Options: `--bc zero` (w(15 m) = 0) for Figs. 2, 4, S4, S6/S7;
`--case A|B|C` (depth-dependent strain) for Figs. 4, S6/S7; `--n-iter N` (default 500 resamples) for Figs. S1, S6/S7.

## Running

```bash
conda env create -f environment.yml
conda activate ses_w
pip install -e .                      # or run from the repository root
export SES_DATA_DIR=/path/to/data     # see config.py
python -m plotting_figures.fig3_seasonal_statistics
```

## License

MIT, see [LICENSE](LICENSE).
