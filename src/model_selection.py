from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd
import os
import json

import arviz as az
from scipy import stats
from scipy.special import logsumexp
from tqdm import tqdm


from simulate_data import (
    inv_logit,
    inv_cloglog,
    inv_loglog,
    inv_plogit,
    inv_splogit,
    inv_lpe,
    inv_grg,
    inv_skewprobit,
    inv_rh,
    inv_glogit,
    LinkType,
)


LINK_FUNCTIONS = {
    "logit": inv_logit,
    "cloglog": inv_cloglog,
    "loglog": inv_loglog,
    "plogit": inv_plogit,
    "splogit": inv_splogit,
    "lpe": inv_lpe,
    "grg": inv_grg,
    "skewprobit": inv_skewprobit,
    "rh": inv_rh,
    "glogit": inv_glogit,
}

def safe_inv_glogit(eta: np.ndarray, alpha1: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    eps = 1e-10
    res_pos = np.zeros_like(eta, dtype=float)
    a1_safe = np.where(np.abs(alpha1) < eps, eps, alpha1)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        pos_alpha_pos = (np.exp(a1_safe * eta) - 1.0) / a1_safe
        a1_neg = np.minimum(a1_safe, -eps)
        max_eta1 = (1.0 / np.abs(a1_neg)) - eps
        eta_clipped1 = np.clip(eta, None, max_eta1)
        pos_alpha_neg = -np.log(np.clip(1.0 - a1_safe * eta_clipped1, 1e-10, None)) / a1_safe
    cond_zero1 = np.abs(alpha1) < eps
    cond_pos1 = alpha1 >= eps
    h_eta_pos = np.where(cond_zero1, eta, np.where(cond_pos1, pos_alpha_pos, pos_alpha_neg))

    eta_abs = np.abs(eta)
    a2_safe = np.where(np.abs(alpha2) < eps, eps, alpha2)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        neg_alpha_pos = -(np.exp(a2_safe * eta_abs) - 1.0) / a2_safe
        a2_neg = np.minimum(a2_safe, -eps)
        max_eta_abs = (1.0 / np.abs(a2_neg)) - eps
        eta_abs_clipped = np.clip(eta_abs, None, max_eta_abs)
        neg_alpha_neg = np.log(np.clip(1.0 - a2_safe * eta_abs_clipped, 1e-10, None)) / a2_safe
    cond_zero2 = np.abs(alpha2) < eps
    cond_pos2 = alpha2 >= eps
    h_eta_neg = np.where(cond_zero2, eta, np.where(cond_pos2, neg_alpha_pos, neg_alpha_neg))
    
    h_eta = np.where(eta > 0, h_eta_pos, h_eta_neg)
    return np.where(h_eta >= 0, 1 / (1 + np.exp(-h_eta)), np.exp(h_eta) / (1 + np.exp(h_eta)))

def compute_log_likelihood(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    link: LinkType,
    r: Optional[float] = None,
    xi: Optional[float] = None,
    lambda_skew: Optional[float] = None,
    gamma_rh: Optional[float] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
    p_epsilon: Optional[float] = None,
    tau_epsilon: Optional[float] = None,
) -> float:
    N, I = y.shape

    eta = (mu[:, None] - b[None, :]) * a[None, :]

    inv_link = LINK_FUNCTIONS[link.lower()]

    if link.lower() == "plogit":
        if r is None:
            raise ValueError(f"{link} requires r parameter")
        p = inv_link(eta, r)
    elif link.lower() == "splogit":
        if r is None or p_epsilon is None or tau_epsilon is None:
            raise ValueError(f"{link} requires r, p_epsilon, tau_epsilon parameters")
        
        sigma_eps = np.sqrt(1.0 / tau_epsilon)
        p0 = inv_link(eta, r)
        loglik0 = y * np.log(np.clip(p0, 1e-10, 1 - 1e-10)) + (1 - y) * np.log(np.clip(1 - p0, 1e-10, 1 - 1e-10))
        loglik0 = loglik0 + np.log(np.clip(1 - p_epsilon, 1e-10, 1.0))
        
        n_quad = 2
        gh_points_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_quad)
        gh_points_scaled = gh_points_np * np.sqrt(2.0)
        gh_weights_norm = gh_weights_np / np.sqrt(np.pi)
        
        loglik1_components = []
        for i in range(n_quad):
            eps_val = gh_points_scaled[i] * sigma_eps
            eta_skew = eta + eps_val
            p1 = inv_link(eta_skew, r)
            loglik_i = y * np.log(np.clip(p1, 1e-10, 1 - 1e-10)) + (1 - y) * np.log(np.clip(1 - p1, 1e-10, 1 - 1e-10))
            loglik_i = loglik_i + np.log(gh_weights_norm[i])
            loglik1_components.append(loglik_i)
            
        loglik1 = logsumexp(np.stack(loglik1_components, axis=-1), axis=-1)
        loglik1 = loglik1 + np.log(np.clip(p_epsilon, 1e-10, 1.0))
        
        log_lik_total = logsumexp(np.stack([loglik0, loglik1], axis=-1), axis=-1)
        return float(np.sum(log_lik_total))
    elif link.lower() == "lpe":
        if xi is None:
            raise ValueError("lpe requires xi parameter")
        p = inv_link(eta, xi)
    elif link.lower() == "skewprobit":
        if lambda_skew is None:
            raise ValueError("skewprobit requires lambda_skew parameter")
        p = inv_link(eta, lambda_skew)
    elif link.lower() == "rh":
        if gamma_rh is None:
            raise ValueError("rh requires gamma_rh parameter")
        p = inv_link(eta, gamma_rh, mu)
    elif link.lower() == "glogit":
        if alpha1 is None or alpha2 is None:
            raise ValueError("glogit requires alpha1 and alpha2 parameters")
        p = inv_link(eta, alpha1, alpha2)
    else:
        p = inv_link(eta)

    p = np.clip(p, 1e-10, 1 - 1e-10)

    log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(log_lik)


def compute_log_likelihood_matrix(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
    thin: int = 1,
) -> np.ndarray:
    post = idata.posterior

    a_samples = post["a"].values.reshape(-1, post["a"].shape[-1]) 
    b_samples = post["b"].values.reshape(-1, post["b"].shape[-1]) 
    mu_samples = post["mu"].values.reshape(-1, post["mu"].shape[-1]) 

    if thin > 1:
        a_samples = a_samples[::thin]
        b_samples = b_samples[::thin]
        mu_samples = mu_samples[::thin]

    M = a_samples.shape[0]
    N, I = y.shape

    r_samples = None
    xi_samples = None
    lambda_skew_samples = None
    gamma_rh_samples = None
    alpha1_samples = None
    alpha2_samples = None

    if link.lower() == "plogit":
        if "r" not in post:
            raise ValueError(f"Parameter 'r' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        if thin > 1:
            r_samples = r_samples[::thin]
        if np.any(np.isnan(r_samples)) or np.any(np.isinf(r_samples)):
            raise ValueError(f"NaN or Inf values found in r posterior samples for {link} model")
            
    elif link.lower() == "splogit":
        if "r" not in post or "p_epsilon" not in post or "tau_epsilon" not in post:
            raise ValueError(f"Parameters 'r', 'p_epsilon', or 'tau_epsilon' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        p_epsilon_samples = post["p_epsilon"].values.reshape(-1)
        tau_epsilon_samples = post["tau_epsilon"].values.reshape(-1)
        if thin > 1:
            r_samples = r_samples[::thin]
            p_epsilon_samples = p_epsilon_samples[::thin]
            tau_epsilon_samples = tau_epsilon_samples[::thin]
        if np.any(np.isnan(r_samples)) or np.any(np.isinf(r_samples)):
            raise ValueError(f"NaN or Inf values found in r posterior samples for {link} model")

    elif link.lower() == "lpe":
        xi_samples = post["xi"].values.reshape(-1)
        if thin > 1:
            xi_samples = xi_samples[::thin]

    elif link.lower() == "skewprobit":
        lambda_skew_samples = post["lambda_skew"].values.reshape(-1)
        if thin > 1:
            lambda_skew_samples = lambda_skew_samples[::thin]

    elif link.lower() == "rh":
        gamma_rh_samples = post["gamma_rh"].values.reshape(-1)
        if thin > 1:
            gamma_rh_samples = gamma_rh_samples[::thin]

    elif link.lower() == "glogit":
        alpha1_samples = post["alpha1"].values.reshape(-1)
        alpha2_samples = post["alpha2"].values.reshape(-1)
        if thin > 1:
            alpha1_samples = alpha1_samples[::thin]
            alpha2_samples = alpha2_samples[::thin]

    # Fully Vectorized Log-Likelihood Computation over M, N, I
    inv_link = LINK_FUNCTIONS[link.lower()]
    
    # eta shape: (M, N, I)
    eta = (mu_samples[:, :, None] - b_samples[:, None, :]) * a_samples[:, None, :]

    if link.lower() == "plogit":
        # reshape r to (M, 1, 1) to broadcast with (M, N, I)
        r_exp = r_samples.reshape(-1, 1, 1)
        p = inv_link(eta, r_exp)
        
    elif link.lower() == "splogit":
        r_exp = r_samples.reshape(-1, 1, 1)
        p_eps_exp = p_epsilon_samples.reshape(-1, 1, 1)
        tau_eps_exp = tau_epsilon_samples.reshape(-1, 1, 1)
        sigma_eps_exp = np.sqrt(1.0 / tau_eps_exp)
        
        p0 = inv_link(eta, r_exp)
        loglik0 = y * np.log(np.clip(p0, 1e-10, 1 - 1e-10)) + (1 - y) * np.log(np.clip(1 - p0, 1e-10, 1 - 1e-10))
        loglik0 += np.log(np.clip(1 - p_eps_exp, 1e-10, 1.0))
        
        n_quad = 2
        gh_points_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_quad)
        gh_points_scaled = gh_points_np * np.sqrt(2.0)
        gh_weights_norm = gh_weights_np / np.sqrt(np.pi)
        
        loglik1_components = []
        for i in range(n_quad):
            eps_val = gh_points_scaled[i] * sigma_eps_exp
            eta_skew = eta + eps_val
            p1 = inv_link(eta_skew, r_exp)
            loglik_i = y * np.log(np.clip(p1, 1e-10, 1 - 1e-10)) + (1 - y) * np.log(np.clip(1 - p1, 1e-10, 1 - 1e-10))
            loglik_i += np.log(gh_weights_norm[i])
            loglik1_components.append(loglik_i)
            
        # loglik1_components is a list of (M, N, I) arrays
        loglik1 = logsumexp(np.stack(loglik1_components, axis=-1), axis=-1)
        loglik1 += np.log(np.clip(p_eps_exp, 1e-10, 1.0))
        
        log_lik_matrix = logsumexp(np.stack([loglik0, loglik1], axis=-1), axis=-1)
        return log_lik_matrix
        
    elif link.lower() == "lpe":
        xi_exp = xi_samples.reshape(-1, 1, 1)
        p = inv_link(eta, xi_exp)
        
    elif link.lower() == "skewprobit":
        ls_exp = lambda_skew_samples.reshape(-1, 1, 1)
        p = inv_link(eta, ls_exp)
        
    elif link.lower() == "rh":
        grh_exp = gamma_rh_samples.reshape(-1, 1, 1)
        # mu_samples is (M, N). We expand to (M, N, 1)
        mu_exp = mu_samples[:, :, None]
        p = inv_link(eta, grh_exp, mu_exp)
        
    elif link.lower() == "glogit":
        a1_exp = alpha1_samples.reshape(-1, 1, 1)
        a2_exp = alpha2_samples.reshape(-1, 1, 1)
        p = safe_inv_glogit(eta, a1_exp, a2_exp)
        
    else:
        p = inv_link(eta)

    p = np.clip(p, 1e-10, 1 - 1e-10)
    log_lik_matrix = y * np.log(p) + (1 - y) * np.log(1 - p)

    return log_lik_matrix



def compute_dic(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
) -> dict[str, float]:
    post = idata.posterior

    a_samples = post["a"].values.reshape(-1, post["a"].shape[-1])
    b_samples = post["b"].values.reshape(-1, post["b"].shape[-1])
    mu_samples = post["mu"].values.reshape(-1, post["mu"].shape[-1])

    M = a_samples.shape[0]

    if np.any(np.isnan(a_samples)) or np.any(np.isinf(a_samples)):
        raise ValueError(f"NaN or Inf values found in 'a' posterior samples")
    if np.any(np.isnan(b_samples)) or np.any(np.isinf(b_samples)):
        raise ValueError(f"NaN or Inf values found in 'b' posterior samples")
    if np.any(np.isnan(mu_samples)) or np.any(np.isinf(mu_samples)):
        raise ValueError(f"NaN or Inf values found in 'mu' posterior samples")

    a_mean = a_samples.mean(axis=0)
    b_mean = b_samples.mean(axis=0)
    mu_mean = mu_samples.mean(axis=0)

    r_mean = None
    xi_mean = None
    lambda_skew_mean = None
    gamma_rh_mean = None
    alpha1_mean = None
    alpha2_mean = None

    p_epsilon_mean = None
    tau_epsilon_mean = None

    if link.lower() == "plogit":
        if "r" not in post:
            raise ValueError(f"Parameter 'r' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        if np.any(np.isnan(r_samples)) or np.any(np.isinf(r_samples)):
            raise ValueError(f"NaN or Inf values found in r posterior samples for {link} model")
        r_mean = float(r_samples.mean())
        
    elif link.lower() == "splogit":
        if "r" not in post or "p_epsilon" not in post or "tau_epsilon" not in post:
            raise ValueError(f"Parameters 'r', 'p_epsilon', or 'tau_epsilon' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        p_epsilon_samples = post["p_epsilon"].values.reshape(-1)
        tau_epsilon_samples = post["tau_epsilon"].values.reshape(-1)
        r_mean = float(r_samples.mean())
        p_epsilon_mean = float(p_epsilon_samples.mean())
        tau_epsilon_mean = float(tau_epsilon_samples.mean())

    elif link.lower() == "lpe":
        xi_samples = post["xi"].values.reshape(-1)
        xi_mean = float(xi_samples.mean())

    elif link.lower() == "skewprobit":
        lambda_skew_samples = post["lambda_skew"].values.reshape(-1)
        lambda_skew_mean = float(lambda_skew_samples.mean())

    elif link.lower() == "rh":
        gamma_rh_samples = post["gamma_rh"].values.reshape(-1)
        gamma_rh_mean = float(gamma_rh_samples.mean())

    elif link.lower() == "glogit":
        alpha1_samples = post["alpha1"].values.reshape(-1)
        alpha2_samples = post["alpha2"].values.reshape(-1)
        alpha1_mean = float(alpha1_samples.mean())
        alpha2_mean = float(alpha2_samples.mean())

    log_lik_at_mean = compute_log_likelihood(
        y, a_mean, b_mean, mu_mean, link,
        r=r_mean, xi=xi_mean, lambda_skew=lambda_skew_mean,
        gamma_rh=gamma_rh_mean,
        alpha1=alpha1_mean, alpha2=alpha2_mean,
        p_epsilon=p_epsilon_mean, tau_epsilon=tau_epsilon_mean
    )
    deviance_at_mean = -2.0 * log_lik_at_mean

    deviances = []
    for m in range(M):
        r_m = None
        xi_m = None
        lambda_skew_m = None
        gamma_rh_m = None
        alpha1_m = None
        alpha2_m = None

        p_epsilon_m = None
        tau_epsilon_m = None

        if link.lower() == "plogit":
            r_m = float(r_samples[m])
        elif link.lower() == "splogit":
            r_m = float(r_samples[m])
            p_epsilon_m = float(p_epsilon_samples[m])
            tau_epsilon_m = float(tau_epsilon_samples[m])
        elif link.lower() == "lpe":
            xi_m = float(xi_samples[m])
        elif link.lower() == "skewprobit":
            lambda_skew_m = float(lambda_skew_samples[m])
        elif link.lower() == "rh":
            gamma_rh_m = float(gamma_rh_samples[m])
        elif link.lower() == "glogit":
            alpha1_m = float(alpha1_samples[m])
            alpha2_m = float(alpha2_samples[m])

        log_lik_m = compute_log_likelihood(
            y, a_samples[m], b_samples[m], mu_samples[m], link,
            r=r_m, xi=xi_m, lambda_skew=lambda_skew_m,
            gamma_rh=gamma_rh_m,
            alpha1=alpha1_m, alpha2=alpha2_m,
            p_epsilon=p_epsilon_m, tau_epsilon=tau_epsilon_m
        )
        deviances.append(-2.0 * log_lik_m)

    deviance_mean = float(np.mean(deviances))

    pd = deviance_mean - deviance_at_mean

    dic = deviance_mean + pd

    return {
        "dic": dic,
        "pd": pd,
        "deviance_mean": deviance_mean,
        "deviance_at_mean": deviance_at_mean,
    }


def compute_lpml(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
    thin: int = 1,
) -> dict[str, float]:
    log_lik_matrix = compute_log_likelihood_matrix(y, idata, link, thin=thin)

    M, N, I = log_lik_matrix.shape

    cpo_log = np.zeros((N, I))

    for i in range(N):
        for j in range(I):
            log_lik_ij = log_lik_matrix[:, i, j] 

            cpo_log[i, j] = -logsumexp(-log_lik_ij) + np.log(M)

    lpml = float(np.sum(cpo_log))

    cpo = np.exp(cpo_log)

    return {
        "lpml": lpml,
        "cpo_mean": float(cpo.mean()),
        "cpo_min": float(cpo.min()),
        "cpo_max": float(cpo.max()),
        "cpo_matrix": cpo,  
    }


def compute_model_selection_criteria(
    y: np.ndarray,
    idatas: list[az.InferenceData],
    links: list[LinkType],
    thin: int = 1,
) -> pd.DataFrame:
    """
    Computes DIC and LPML for multiple estimation links on the same dataset.
    Returns a pandas DataFrame comparing the criteria across all provided links.
    """
    records = []
    for idata, link in zip(idatas, links):
        dic_res = compute_dic(y, idata, link)
        lpml_res = compute_lpml(y, idata, link, thin=thin)
        records.append({
            "Link": link,
            "DIC": float(dic_res["dic"]),
            "pD": float(dic_res["pd"]),
            "Deviance_Mean": float(dic_res["deviance_mean"]),
            "LPML": float(lpml_res["lpml"])
        })
    return pd.DataFrame(records)

# =============================================================================
# Pipeline Orchestrator & Save Logic
# =============================================================================

def compute_and_save_model_selection(
    df: pd.DataFrame,
    data_type: Literal["simulation", "real_data"],
    sim_link: str,
    sim_r: float,
    est_link: str,
    draws: int,
    rep: int,
    base_dir: str = "outputs"
):
    """
    Saves the computed Model Selection DataFrame to a CSV file.
    """
    out_dir = os.path.join(base_dir, data_type, "DIC")
    
    r_str = f"{sim_r}" if sim_r is not None else "None"
    data_prefix = "simulation_data" if data_type == "simulation" else "real_data"
    
    if data_type == "simulation":
        folder_name = f"DIC_{data_prefix}_{sim_link}_r_{r_str}_estlink_{est_link}_draw_{draws}"
        filename = f"DIC_{data_prefix}_{sim_link}_r_{r_str}_estlink_{est_link}_draw_{draws}_rep_{rep}.csv"
    else:
        folder_name = f"DIC_{data_prefix}_estlink_{est_link}_draw_{draws}"
        filename = f"DIC_{data_prefix}_estlink_{est_link}_draw_{draws}_rep_{rep}.csv"
        
    out_dir = os.path.join(out_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, filename)

    df.to_csv(save_path, index=False)
    # print(f"Saved Model Selection Criteria to {save_path}")


def compute_single_dic_lpml(
    data_type: str,
    sim_link: str,
    sim_r: float,
    est_link: str,
    project_root: str,
):
    """
    1. 计算单独sim_link, sim_r, est_link 的 DIC 和 LPML 并保存每个 rep 的结果。
    """
    out_dir = os.path.join(project_root, "outputs", data_type)
    samplings_dir = os.path.join(out_dir, "samplings")
    
    if data_type == "simulation":
        config_filename = f"config_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}.json"
    else:
        config_filename = f"config_{data_type}_estlink_{est_link}.json"
    
    config_path = os.path.join(out_dir, "config", config_filename)
    if not os.path.exists(config_path):
        print(f"Warning: Master config file not found at {config_path}. Skipping.")
        return
        
    with open(config_path, 'r') as f:
        master_config = json.load(f)
    N_reps = master_config['n_reps']
    draws = master_config['draws']

    y_real = None
    if data_type != "simulation":
        real_data_path = os.path.join(project_root, "data/real_data/real_data.npz")
        y_real = np.load(real_data_path)['y']

    for rep_idx in tqdm(range(N_reps), desc=f"Computing DIC for {est_link}", unit="rep"):
        # Check if DIC result is already saved, skip if it exists
        dic_base_dir = os.path.join(project_root, "outputs", data_type, "DIC")
        r_str = f"{sim_r}" if sim_r is not None else "None"
        data_prefix = "simulation_data" if data_type == "simulation" else "real_data"
        
        if data_type == "simulation":
            dic_folder = f"DIC_{data_prefix}_{sim_link}_r_{r_str}_estlink_{est_link}_draw_{draws}"
            dic_file = f"DIC_{data_prefix}_{sim_link}_r_{r_str}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.csv"
        else:
            dic_folder = f"DIC_{data_prefix}_estlink_{est_link}_draw_{draws}"
            dic_file = f"DIC_{data_prefix}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.csv"
            
        expected_dic_path = os.path.join(dic_base_dir, dic_folder, dic_file)
        if os.path.exists(expected_dic_path):
            continue

        if data_type == "simulation":
            sim_data_path = os.path.join(
                project_root, 
                "data", "simulation", 
                f"simulation_data_{sim_link}_r_{sim_r}",
                f"simulation_data_{sim_link}_r_{sim_r}_rep_{rep_idx}.npz"
            )
            if not os.path.exists(sim_data_path):
                print(f"Warning: Simulation data not found at {sim_data_path}. Skipping rep {rep_idx}.")
                continue
            y = np.load(sim_data_path)['y']
        else:
            y = y_real

        if data_type == "simulation":
            folder_name = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}"
            filename = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
        else:
            folder_name = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}"
            filename = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
            
        file_path = os.path.join(samplings_dir, folder_name, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: Missing data for est_link='{est_link}', rep={rep_idx}.")
            continue

        data = np.load(file_path)
        idata = az.from_dict(posterior=dict(data))
        
        df = compute_model_selection_criteria(y, [idata], [est_link])
        
        compute_and_save_model_selection(
            df=df,
            data_type=data_type,
            sim_link=sim_link,
            sim_r=sim_r,
            est_link=est_link,
            draws=draws,
            rep=rep_idx,
            base_dir=os.path.join(project_root, "outputs")
        )

def compute_all_dic_lpml(
    data_type: str,
    sim_links: list[str] | None,
    sim_rs: list[float] | None,
    est_links: list[str],
    project_root: str,
):
    """
    2. 用第一个函数对list的sim_link, sim_r, est_link的每一种组合计算DIC和LPML并且保存
    """
    if not est_links:
        print("No estimation links provided.")
        return

    actual_sim_links = sim_links if (data_type == "simulation" and sim_links is not None) else ["none"]
    actual_sim_rs = sim_rs if (data_type == "simulation" and sim_rs is not None) else [0.0]

    for sim_link in actual_sim_links:
        for sim_r in actual_sim_rs:
            for est_link in est_links:
                compute_single_dic_lpml(
                    data_type=data_type,
                    sim_link=sim_link,
                    sim_r=sim_r,
                    est_link=est_link,
                    project_root=project_root
                )

def summarize_model_selections(
    data_type: str,
    sim_link: str,
    sim_r: float,
    est_links: list[str],
    project_root: str,
):
    """
    3. 给定sim_link, sim_r, 和一个list的est_link，用第二步的结果合并所有的DIC和LPML，每一行是一个est_link，并且保存
    """
    if not est_links:
        return
        
    out_dir = os.path.join(project_root, "outputs", data_type)
    
    first_est_link = est_links[0]
    if data_type == "simulation":
        config_filename = f"config_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{first_est_link}.json"
    else:
        config_filename = f"config_{data_type}_estlink_{first_est_link}.json"
        
    config_path = os.path.join(out_dir, "config", config_filename)
    if not os.path.exists(config_path):
        print(f"Warning: Master config file not found at {config_path}. Cannot determine draws for summary.")
        return
        
    with open(config_path, 'r') as f:
        master_config = json.load(f)
    draws = master_config['draws']
    
    dic_out_dir = os.path.join(out_dir, "DIC")
    r_str = f"{sim_r}" if sim_r is not None else "None"
    data_prefix = "simulation_data" if data_type == "simulation" else "real_data"
    
    all_rep_results_for_summary = []
    
    for est_link in est_links:
        if data_type == "simulation":
            folder_name = f"DIC_{data_prefix}_{sim_link}_r_{r_str}_estlink_{est_link}_draw_{draws}"
        else:
            folder_name = f"DIC_{data_prefix}_estlink_{est_link}_draw_{draws}"
            
        folder_path = os.path.join(dic_out_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: No DIC results folder found at {folder_path}")
            continue
            
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(folder_path, csv_file))
            all_rep_results_for_summary.append(df)
            
    if all_rep_results_for_summary:
        master_df = pd.concat(all_rep_results_for_summary)
        summary_df = master_df.groupby("Link").mean().reset_index()
        
        os.makedirs(dic_out_dir, exist_ok=True)
        
        if data_type == "simulation":
            summary_filename = f"DIC_summary_{data_prefix}_{sim_link}_r_{r_str}_draw_{draws}.csv"
        else:
            summary_filename = f"DIC_summary_{data_prefix}_draw_{draws}.csv"
            
        summary_path = os.path.join(dic_out_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ All-replication summary saved to: {summary_path}")
        print(summary_df.to_string(index=False))
    else:
        print(f"\n❌ No results found to summarize for sim_link={sim_link}, sim_r={sim_r}.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_type = "simulation"
    sim_links = ["splogit"]
    sim_rs = [0.25, 0.5, 1.0, 2.0, 4.0]
    est_links = ["plogit", "splogit"]

    # data_type = "real_data"
    # sim_links = None
    # sim_rs = None
    # est_links = ["logit", "cloglog", "loglog", "plogit", "splogit", "lpe", "grg", "skewprobit", "rh", "glogit"]

    print("Step 1 & 2: Computing DIC/LPML for all combinations...")
    compute_all_dic_lpml(
        data_type=data_type,
        sim_links=sim_links,
        sim_rs=sim_rs,
        est_links=est_links,
        project_root=project_root,
    )

    data_type = "simulation"
    sim_links = ["splogit"]
    sim_rs = [0.25, 0.5, 1.0, 2.0, 4.0]
    est_links = ["plogit", "splogit"]

    # data_type = "real_data"
    # sim_links = None
    # sim_rs = None
    # est_links = ["logit", "cloglog", "loglog", "plogit", "splogit", "lpe", "grg", "skewprobit", "rh", "glogit"]
    
    print("\nStep 3: Summarizing results...")
    actual_sim_links = sim_links if (data_type == "simulation" and sim_links is not None) else ["none"]
    actual_sim_rs = sim_rs if (data_type == "simulation" and sim_rs is not None) else [0.0]
    
    for sim_link in actual_sim_links:
        for sim_r in actual_sim_rs:
            summarize_model_selections(
                data_type=data_type,
                sim_link=sim_link,
                sim_r=sim_r,
                est_links=est_links,
                project_root=project_root,
            )