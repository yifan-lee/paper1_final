# MCMC for IRT with Power Links

Running MCMC samplings on Item Response Theory (IRT) models with power-link and spike-and-slab priors. The repository provides simulation module, MCMC module, recovery metrics module, and model-selection module.


## Execution

1. Clone the repository:
2. Install the project dependences
3. Run `simulate_parameters.py` to generate simulation parameters.
4. Configure and run `simulate_data.py` to create synthetic datasets.
5. Configure and run `mcmc.py` to run the MCMC sampler and collect posterior samples.
6. Configure and run `metrics.py` to compute recovery statistics from the MCMC output.
7. Configure and run `model_selection.py` to compute model-selection criteria (DIC, LPML).

Each script is located under the `src/` package.

## Main Arguments / Options

- `sim_link`: link used for data simulation. Options: `cloglog`, `loglog`, `plogit`, `splogit`.
- `r`: power/link parameter. Typical values: `0.0`, `0.25`, `0.5`, `1.0`, `2.0`, `4.0`.
- `est_link`: link used during estimation. Options include: `logit`, `cloglog`, `loglog`, `plogit`, `splogit`, `lpe`, `grg`, `skewprobit`, `rh`, `glogit`.
