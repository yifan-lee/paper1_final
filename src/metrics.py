from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import json

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class MCMCMetrics:

    bias: np.ndarray
    mse: np.ndarray
    sd: np.ndarray
    se: np.ndarray
    cp: np.ndarray
    rhat: np.ndarray

    def as_dict(self) -> Mapping[str, np.ndarray]:
        return {
            "bias": self.bias,
            "mse": self.mse,
            "sd": self.sd,
            "se": self.se,
            "cp": self.cp,
            "rhat": self.rhat,
        }


def compute_mcmc_metrics(
    replications: Sequence[ArrayLike],
    true_value: ArrayLike | float,
    credible_interval: float = 0.95,
    rhat_values: ArrayLike | None = None,
) -> MCMCMetrics:
    draws = _stack_replications(replications)
    R, M = draws.shape[:2]
    if M < 2:
        raise ValueError("Each replication must contain at least two draws.")

    param_shape = draws.shape[2:]
    true_arr = np.asarray(true_value, dtype=float)
    if true_arr.shape not in (param_shape, ()):
        try:
            true_arr = np.broadcast_to(true_arr, param_shape)
        except ValueError as err:
            raise ValueError(
                f"true_value shape {true_arr.shape} is not broadcastable to {param_shape}"
            ) from err

    draws = draws.astype(float, copy=False)
    true_arr = true_arr.astype(float, copy=False)

    posterior_means = draws.mean(axis=1)  
    bias = (posterior_means - true_arr).mean(axis=0)
    mse = ((posterior_means - true_arr) ** 2).mean(axis=0)

    within_sd = draws.std(axis=1, ddof=1)
    sd = within_sd.mean(axis=0)

    if R >= 2:
        se = posterior_means.std(axis=0, ddof=1)
    else:
        se = np.full(param_shape, np.nan)

    if R >= 2:
        cp = _coverage_probability(draws, true_arr, credible_interval)
    else:
        cp = np.full(param_shape, np.nan)

    if rhat_values is not None:
        try:
            rhat_stacked = np.stack([np.asarray(rh) for rh in rhat_values], axis=0)
            rhat = rhat_stacked.mean(axis=0)
        except Exception:
            rhat = np.full(param_shape, np.nan)
    else:
        rhat = np.full(param_shape, np.nan)

    return MCMCMetrics(
        bias=np.asarray(bias),
        mse=np.asarray(mse),
        sd=np.asarray(sd),
        se=np.asarray(se),
        cp=np.asarray(cp),
        rhat=np.asarray(rhat),
    )

def _stack_replications(replications: Sequence[ArrayLike]) -> np.ndarray:
    if not replications:
        raise ValueError("replications must contain at least one element.")

    rep_arrays = [np.asarray(rep) for rep in replications]
    first_shape = rep_arrays[0].shape
    if len(first_shape) == 0:
        raise ValueError("Each replication must contain posterior draws (at least 1-D).")

    for idx, rep in enumerate(rep_arrays):
        if rep.shape != first_shape:
            raise ValueError(
                f"Replication {idx} has shape {rep.shape}, expected {first_shape}."
            )

    stacked = np.stack(rep_arrays, axis=0)
    if stacked.ndim < 2:
        raise ValueError("Replications must provide at least two dimensions (R, M).")
    return stacked


def _coverage_probability(
    draws: np.ndarray,
    true_value: np.ndarray,
    credible_interval: float,
) -> np.ndarray:
    if not (0.0 < credible_interval < 1.0):
        raise ValueError("credible_interval must be between 0 and 1.")

    alpha = 1.0 - credible_interval
    R = draws.shape[0]
    flat_draws = draws.reshape(R, draws.shape[1], -1)  # (R, M, n_params)
    flat_truth = true_value.reshape(-1) if true_value.shape else np.array([true_value])

    if flat_truth.size != flat_draws.shape[-1]:
        raise ValueError(
            f"true_value has {flat_truth.size} elements, expected {flat_draws.shape[-1]}."
        )

    coverage_counts = np.zeros(flat_draws.shape[-1], dtype=int)

    for r in range(R):
        replication = flat_draws[r]
        for p in range(flat_draws.shape[-1]):
            lower, upper = _highest_density_interval(replication[:, p], alpha)
            if lower <= flat_truth[p] <= upper:
                coverage_counts[p] += 1

    cp = coverage_counts / R
    return cp.reshape(true_value.shape if true_value.shape else ())


def _highest_density_interval(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    sorted_draws = np.sort(np.asarray(draws, dtype=float))
    n = sorted_draws.size
    if n == 0:
        raise ValueError("Cannot compute HPD interval from an empty sample.")

    mass = 1.0 - alpha
    interval_size = int(np.floor(mass * n))
    interval_size = min(max(interval_size, 1), n - 1)

    widths = sorted_draws[interval_size:] - sorted_draws[: n - interval_size]
    min_idx = int(np.argmin(widths))
    low = float(sorted_draws[min_idx])
    high = float(sorted_draws[min_idx + interval_size])
    return low, high


__all__ = ["MCMCMetrics", "compute_mcmc_metrics"]



import os
import pandas as pd
import arviz as az
from simulate_parameters import Params

def generate_all_metrics(
    data_type: str,
    sim_links: list[str] | None,
    sim_rs: list[float] | None,
    est_links: list[str],
    project_root: str,
):
    """
    Main orchestrator to load MCMC results and compute recovery metrics.
    """
    out_dir = os.path.join(project_root, "outputs", data_type)
    samplings_dir = os.path.join(out_dir, "samplings")
    
    true_params = None
    if data_type == "simulation":
        params_path = os.path.join(project_root, "data", "parameters.npz")
        true_params = Params.load(params_path)
        print(f"Loaded true parameters from {params_path}")
    
    actual_sim_links = sim_links if (data_type == "simulation" and sim_links is not None) else ["none"]
    actual_sim_rs = sim_rs if (data_type == "simulation" and sim_rs is not None) else [0.0]

    for sim_link in actual_sim_links:
        for sim_r in actual_sim_rs:
            all_metrics_for_summary = []
            last_draws = None

            for est_link in est_links:
                if data_type == "simulation":
                    print(f"\nLoading MCMC results for estimation link: '{est_link}', sim_link: '{sim_link}', sim_r: {sim_r}...")
                    config_filename = f"config_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}.json"
                else:
                    print(f"\nLoading MCMC results for estimation link: '{est_link}'...")
                    config_filename = f"config_{data_type}_estlink_{est_link}.json"
                
                config_path = os.path.join(out_dir, "config", config_filename)
                
                if not os.path.exists(config_path):
                    print(f"Config file not found: {config_path}. Skipping.")
                    continue
                    
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                draws = config['draws']
                N_reps = config['n_reps']
                last_draws = draws

                idatas = []
                for rep_idx in range(N_reps):
                    if data_type == "simulation":
                        folder_name = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}"
                        filename = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
                    else:
                        folder_name = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}"
                        filename = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
                        
                    file_path = os.path.join(samplings_dir, folder_name, filename)
                    
                    if not os.path.exists(file_path):
                        continue
                        
                    data = np.load(file_path)
                    idata = az.from_dict(posterior=dict(data))
                    idatas.append(idata)
                    
                if not idatas:
                    if data_type == "simulation":
                        print(f"No results found for {est_link} (sim_link={sim_link}, sim_r={sim_r}). Skipping metrics computation.")
                    else:
                        print(f"No results found for {est_link}. Skipping metrics computation.")
                    continue
                    
                print(f"Computing metrics for '{est_link}' across {len(idatas)} replications...")
                df = _compute_and_save_recovery_metrics(
                    idatas=idatas,
                    true_params=true_params,
                    data_type=data_type,
                    sim_link=sim_link,
                    sim_r=sim_r,
                    est_link=est_link,
                    draws=draws,
                    project_root=project_root
                )

                if df is not None and not df.empty:
                    df.insert(0, 'est_link', est_link)
                    all_metrics_for_summary.append(df)

            if all_metrics_for_summary:
                summary_df = pd.concat(all_metrics_for_summary)
                summary_out_dir = os.path.join(project_root, "outputs", data_type, "metrics")
                os.makedirs(summary_out_dir, exist_ok=True)
                
                if data_type == "simulation":
                    summary_filename = f"metrics_summary_{data_type}_simlink_{sim_link}_r_{sim_r}_draw_{last_draws}.csv"
                else:
                    summary_filename = f"metrics_summary_{data_type}_draw_{last_draws}.csv"
                    
                summary_path = os.path.join(summary_out_dir, summary_filename)
                summary_df.to_csv(summary_path, index=False)
                print(f"\n========================================")
                print(f"Saved Metrics Summary to {summary_path}")
                print(f"========================================")


def _compute_and_save_recovery_metrics(
    idatas: list[az.InferenceData],
    true_params: Params,
    data_type: str,
    sim_link: str,
    sim_r: float,
    est_link: str,
    draws: int,
    project_root: str
):
    out_dir = os.path.join(project_root, "outputs", data_type, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    filename = f"metrics_{data_type}_simlink_{sim_link}_r_{sim_r}_estlink_{est_link}_draw_{draws}.csv"
    save_path = os.path.join(out_dir, filename)

    a_draws = [idata.posterior["a"].values.reshape(-1, idata.posterior["a"].shape[-1]) for idata in idatas]
    b_draws = [idata.posterior["b"].values.reshape(-1, idata.posterior["b"].shape[-1]) for idata in idatas]
    mu_draws = [idata.posterior["mu"].values.reshape(-1, idata.posterior["mu"].shape[-1]) for idata in idatas]
    
    a_rhat = [az.rhat(idata)["a"].values for idata in idatas]
    b_rhat = [az.rhat(idata)["b"].values for idata in idatas]
    mu_rhat = [az.rhat(idata)["mu"].values for idata in idatas]

    metrics_a = compute_mcmc_metrics(a_draws, true_params.a, rhat_values=a_rhat)
    metrics_b = compute_mcmc_metrics(b_draws, true_params.b, rhat_values=b_rhat)
    metrics_mu = compute_mcmc_metrics(mu_draws, true_params.mu, rhat_values=mu_rhat)
    
    def extract_summary(m: MCMCMetrics, name: str):
        return {
            "Parameter": name,
            "Bias_Mean": float(np.nanmean(m.bias)),
            "Bias_2.5%": float(np.nanpercentile(m.bias, 2.5)),
            "Bias_97.5%": float(np.nanpercentile(m.bias, 97.5)),
            "MSE_Mean": float(np.nanmean(m.mse)),
            "MSE_2.5%": float(np.nanpercentile(m.mse, 2.5)),
            "MSE_97.5%": float(np.nanpercentile(m.mse, 97.5)),
            "SD_Mean": float(np.nanmean(m.sd)),
            "SD_2.5%": float(np.nanpercentile(m.sd, 2.5)),
            "SD_97.5%": float(np.nanpercentile(m.sd, 97.5)),
            "SE_Mean": float(np.nanmean(m.se)),
            "SE_2.5%": float(np.nanpercentile(m.se, 2.5)),
            "SE_97.5%": float(np.nanpercentile(m.se, 97.5)),
            "CP_Mean": float(np.nanmean(m.cp)),
            "CP_2.5%": float(np.nanpercentile(m.cp, 2.5)),
            "CP_97.5%": float(np.nanpercentile(m.cp, 97.5)),
            "Rhat_Mean": float(np.nanmean(m.rhat)),
            "Rhat_2.5%": float(np.nanpercentile(m.rhat, 2.5)),
            "Rhat_97.5%": float(np.nanpercentile(m.rhat, 97.5))
        }

    records = [
        extract_summary(metrics_a, "a (discrimination)"),
        extract_summary(metrics_b, "b (difficulty)"),
        extract_summary(metrics_mu, "mu (ability)")
    ]

    # Additional parameters
    if est_link in {"plogit", "splogit"}:
        r_draws = [idata.posterior["r"].values.reshape(-1) for idata in idatas]
        r_rhat = [az.rhat(idata)["r"].values for idata in idatas]
        r_true_val = sim_r if data_type == "simulation" else true_params.r
        metrics_r = compute_mcmc_metrics(r_draws, r_true_val, rhat_values=r_rhat)
        records.append(extract_summary(metrics_r, "r"))
        
    if est_link == "splogit":
        pe_draws = [idata.posterior["p_epsilon"].values.reshape(-1) for idata in idatas]
        pe_rhat = [az.rhat(idata)["p_epsilon"].values for idata in idatas]
        metrics_pe = compute_mcmc_metrics(pe_draws, true_params.p_epsilon, rhat_values=pe_rhat)
        records.append(extract_summary(metrics_pe, "p_epsilon"))
        
        te_draws = [idata.posterior["tau_epsilon"].values.reshape(-1) for idata in idatas]
        te_rhat = [az.rhat(idata)["tau_epsilon"].values for idata in idatas]
        metrics_te = compute_mcmc_metrics(te_draws, true_params.hyper_params.tau_epsilon, rhat_values=te_rhat)
        records.append(extract_summary(metrics_te, "tau_epsilon"))

    elif est_link == "lpe":
        xi_draws = [idata.posterior["xi"].values.reshape(-1) for idata in idatas]
        xi_rhat = [az.rhat(idata)["xi"].values for idata in idatas]
        metrics_xi = compute_mcmc_metrics(xi_draws, true_params.xi, rhat_values=xi_rhat)
        records.append(extract_summary(metrics_xi, "xi"))
        
    elif est_link == "skewprobit":
        ls_draws = [idata.posterior["lambda_skew"].values.reshape(-1) for idata in idatas]
        ls_rhat = [az.rhat(idata)["lambda_skew"].values for idata in idatas]
        metrics_ls = compute_mcmc_metrics(ls_draws, true_params.lambda_skew, rhat_values=ls_rhat)
        records.append(extract_summary(metrics_ls, "lambda_skew"))
        
    elif est_link == "rh":
        grh_draws = [idata.posterior["gamma_rh"].values.reshape(-1) for idata in idatas]
        grh_rhat = [az.rhat(idata)["gamma_rh"].values for idata in idatas]
        metrics_grh = compute_mcmc_metrics(grh_draws, true_params.gamma_rh, rhat_values=grh_rhat)
        records.append(extract_summary(metrics_grh, "gamma_rh"))

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"\n========================================\nSaved Metrics to {save_path}\n========================================")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_type = "simulation"
    sim_links = ["splogit"]
    sim_rs = [0.25, 0.5, 1.0, 2.0, 4.0]
    # sim_rs = [2.0, 4.0]
    est_links = ["splogit"]

    generate_all_metrics(
        data_type=data_type,
        sim_links=sim_links,
        sim_rs=sim_rs,
        est_links=est_links,
        project_root=project_root,
    )

    os.system('say "模型分析程序跑完了"')