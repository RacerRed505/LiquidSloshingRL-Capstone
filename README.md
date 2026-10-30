# Liquid Sloshing in a Pivoted Tank: SPH Simulation and Active Control

**Capstone project, Department of Mechanical Engineering, MPSTME (SVKM's NMIMS), Mumbai Campus**

Team: A003 Hetansh Jain · A008 Aryaman Mavani · A052 Rahil Chawla
Supervisor: Dr. Sunny Nanade

A partly filled rectangular tank rotates about a single pivot at its base, driven by a motor. When it moves,
the liquid inside sloshes and pushes back on the tank. This project builds the tank entirely in simulation,
models the liquid in two ways (a fast equivalent-pendulum model and a detailed particle simulation), and
compares three controllers that try to keep the liquid calm.

The tank (0.40 × 0.25 × 0.30 m, 0.20 m water) is a 1:5 Froude-scale model of a road-tanker compartment
(about 2.0 × 1.25 × 1.5 m).

## Main results

| Result | Value |
|---|---|
| SPH grid convergence (particle spacing 10 → 5 → 3 → 2 mm) | change between sizes 1.40 → 0.70 → 0.35 mm; observed order ≈ 1.12; GCI 6.4 % |
| SPH vs exact hydrostatic answer (tank tilted and held) | 3.1 % error at 2 mm |
| Sloshing frequency (SPH, free oscillation) | 1.422 Hz; +4.2 % vs potential theory, +5.9 % vs Housner |
| Two-way coupled control on SPH water (6 s test) | PPO: 15× less sloshing energy than PID, but stable for ~5 s; PID: stable for the full test; LQR (designed on the pendulum model): failed after 0.7 s |
| Fine-tuning the learned controller on the full multibody tank | 50,000 steps improved its score 3.8×; longer fine-tuning made it worse |

Full details are in [`docs/paper/LiquidSloshingRL_CFD_Paper.pdf`](docs/paper/LiquidSloshingRL_CFD_Paper.pdf).

## Repository layout

| Folder | Contents |
|---|---|
| `builders/` | Builds the tank, pivot and motor in PyChrono from simple primitives (no CAD) |
| `physics/` | Liquid models: Housner equivalent pendulum (fast) and Chrono::FSI SPH (detailed), plus the two-way coupled SPH tank |
| `environment/` | Gymnasium environments the controllers act in |
| `control/` | PID and LQR controllers |
| `training/`, `evaluation/` | PPO training and evaluation (stable-baselines3) |
| `models/` | Trained controllers (see below) |
| `benchmarks/` | Passive / PID / LQR / PPO comparison |
| `scripts/` | SPH runs, comparisons, convergence study and figure generation |
| `tests/` | Automated tests |
| `config/`, `utilities/` | Configuration dataclasses, Housner formulas, geometry and inertia helpers |
| `notebooks/` | Google Colab notebook for the SPH viscosity and frequency checks |
| `results/` | Key result data (CSV) and rendered SPH figures and animations |
| `figures/` | Free-body diagram, control block diagram, PPO flowchart, code flowcharts |
| `interactive/` | Two browser pages: a 3D sloshing lab and an interactive report (open the HTML files directly; no internet needed for the 3D lab) |
| `docs/` | Research paper, Appendix A flowcharts, hand-calculation vs simulation spreadsheet, final presentation, and Chrono::FSI build guides |

### Trained models

| File | What it is |
|---|---|
| `models/ppo_sloshing` | PPO trained on the fast surrogate model |
| `models/ppo_sloshing_realphysics` | The same, fine-tuned for 50,000 steps on the full multibody tank |
| `models/ppo_sloshing_observable` | PPO trained with wall-pressure observations instead of the liquid's position |
| `models/ppo_sloshing_closed_loop_sph_16k_supervised_514560_steps` | Best controller fine-tuned in the two-way coupled SPH tank (used for the closed-loop results) |

Each model has a matching `*_vecnormalize.pkl` file holding its observation normalisation.

## Running it

Python 3.10 with the packages in `requirements.txt`, plus PyChrono:

```bash
pip install -r requirements.txt
conda install -c conda-forge -c projectchrono pychrono

python -m pytest tests/                                   # tests (SPH tests skip without Chrono::FSI)
python -m training.train_ppo                              # train PPO on the fast model
python -m evaluation.evaluate_policy models/ppo_sloshing --n-episodes 20
python -m benchmarks.run_controller_benchmarks --ppo-model models/ppo_sloshing
python main.py --fluid-viz                                # 3D demo window (needs pychrono.irrlicht)
```

The SPH (particle) simulations need Chrono::FSI, which requires an NVIDIA GPU with CUDA and a from-source
Chrono build. See [`docs/build_guides/chrono_fsi_build_guide.md`](docs/build_guides/chrono_fsi_build_guide.md).
Runs in this project used an RTX 3090 and an RTX 5060.

Scripts write their output files to the repository root by default; the copies used in the paper are in
`results/`.

## Notes

- Code comments sometimes cite "Entry N" of the development log or "the project notes". These refer to the
  team's internal working log, which is kept in a separate private repository.
- The raw particle captures for the 3 mm and 2 mm runs (1.1 GB and 3.7 GB) are not included because of their
  size; their results and rendered figures are.

## Key references

- G. W. Housner, "The dynamic behavior of water tanks," *Bull. Seismol. Soc. Am.*, 53(2), 1963.
- ACI Committee 350, *ACI 350.3-06*, American Concrete Institute, 2006.
- A. Tasora et al., "Chrono: An open source multi-physics dynamics engine," Springer LNCS, 2016.
- S. Adami et al., "A generalized wall boundary condition for SPH," *J. Comput. Phys.*, 231, 2012.
- I. B. Celik et al., "Procedure for estimation and reporting of uncertainty due to discretization in CFD,"
  *J. Fluids Eng.*, 130, 2008.
- J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

The full list of references is in the paper.
