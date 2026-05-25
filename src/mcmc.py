from __future__ import annotations
from typing import Optional, Literal, Any
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import os
import sys
import multiprocessing
import json
import time
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from utils import generate_seed

LinkType = Literal["logit", "cloglog", "loglog", "plogit", "splogit",
                    "lpe", "grg", "skewprobit", "rh", "glogit"]
SamplerType = Literal["nuts", "hmc", "gibbs", "slice", "mh", "metropolis"]


def _sigmoid(x: pt.TensorVariable) -> pt.TensorVariable:
    return pt.sigmoid(x)


def inv_logit_pt(eta: pt.TensorVariable) -> pt.TensorVariable:
    return _sigmoid(eta)


def inv_cloglog_pt(eta: pt.TensorVariable) -> pt.TensorVariable:
    eta_safe = pt.clip(eta, -30.0, 30.0)
    return 1.0 - pt.exp(-pt.exp(eta_safe))


def inv_loglog_pt(eta: pt.TensorVariable) -> pt.TensorVariable:
    eta_safe = pt.clip(eta, -30.0, 30.0)
    return pt.exp(-pt.exp(-eta_safe))


def inv_plogit_pt(eta: pt.TensorVariable, r: pt.TensorVariable) -> pt.TensorVariable:
    r_safe = pt.clip(r, 1e-7, np.inf)
    return pt.switch(
        pt.le(r, 1.0),
        pt.clip(_sigmoid(eta / r_safe), 1e-10, 1.0) ** r,
        1.0 - pt.clip(_sigmoid(-r * eta), 1e-10, 1.0) ** (1.0 / r_safe),
    )


def inv_splogit_pt(eta: pt.TensorVariable, r: pt.TensorVariable) -> pt.TensorVariable:
    return inv_plogit_pt(eta, r)


def inv_lpe_pt(eta: pt.TensorVariable, xi: pt.TensorVariable) -> pt.TensorVariable:
    logit_p = pt.clip(_sigmoid(eta), 1e-10, 1.0)
    return pt.power(logit_p, xi)


def inv_grg_pt(eta: pt.TensorVariable) -> pt.TensorVariable:
    eta_safe = pt.clip(eta, -30.0, 30.0)
    gumbel_max = pt.exp(-pt.exp(-eta_safe))
    reverse_gumbel = 1.0 - pt.exp(-pt.exp(eta_safe))
    return 0.5 * (gumbel_max + reverse_gumbel)


def inv_skewprobit_pt(eta: pt.TensorVariable, lambda_skew: pt.TensorVariable) -> pt.TensorVariable:
    phi_eta = 0.5 * pt.erfc(-eta / pt.sqrt(2.0))
    owens_t_val = pt.math.owens_t(eta, lambda_skew)
    return phi_eta - 2.0 * owens_t_val


def inv_rh_pt(eta: pt.TensorVariable, gamma_rh: pt.TensorVariable, mu: pt.TensorVariable) -> pt.TensorVariable:
    log_scale = pt.clip(gamma_rh * mu[:, None], -30.0, 30.0)
    scaled_eta = eta * pt.exp(-0.5 * log_scale)
    return 0.5 * pt.erfc(-scaled_eta / pt.sqrt(2.0))

def _h_alpha_pt(eta: pt.TensorVariable, alpha: pt.TensorVariable, positive: bool = True) -> pt.TensorVariable:
    eps = 1e-10

    if positive:
        alpha_eta = pt.clip(alpha * eta, -30.0, 30.0)
        h_pos = (pt.exp(alpha_eta) - 1.0) / (alpha + eps)
        h_zero = eta
        h_neg = -pt.log(pt.clip(1.0 - alpha_eta, eps, 1e10)) / (alpha - eps)

        result = pt.switch(
            pt.gt(alpha, eps),
            h_pos,
            pt.switch(
                pt.lt(alpha, -eps),
                h_neg,
                h_zero
            )
        )
    else:
        eta_abs = pt.abs(eta)
        alpha_eta_abs = pt.clip(alpha * eta_abs, -30.0, 30.0)
        h_pos = -(pt.exp(alpha_eta_abs) - 1.0) / (alpha + eps)
        h_zero = eta
        h_neg = pt.log(pt.clip(1.0 - alpha_eta_abs, eps, 1e10)) / (alpha - eps)

        result = pt.switch(
            pt.gt(alpha, eps),
            h_pos,
            pt.switch(
                pt.lt(alpha, -eps),
                h_neg,
                h_zero
            )
        )

    return result


def inv_glogit_pt(eta: pt.TensorVariable, alpha1: pt.TensorVariable, alpha2: pt.TensorVariable) -> pt.TensorVariable:
    h_eta_positive = _h_alpha_pt(eta, alpha1, positive=True)
    h_eta_negative = _h_alpha_pt(eta, alpha2, positive=False)
    h_eta = pt.switch(pt.gt(eta, 0.0), h_eta_positive, h_eta_negative)
    return _sigmoid(h_eta)


def get_inverse_link_pt(
    link: str,
    r: Optional[pt.TensorVariable] = None,
    xi: Optional[pt.TensorVariable] = None,
    lambda_skew: Optional[pt.TensorVariable] = None,
    gamma_rh: Optional[pt.TensorVariable] = None,
    alpha1: Optional[pt.TensorVariable] = None,
    alpha2: Optional[pt.TensorVariable] = None,
    mu: Optional[pt.TensorVariable] = None,
):
    link_lower = link.lower()

    if link_lower == "logit":
        return lambda eta: inv_logit_pt(eta)
    elif link_lower == "cloglog":
        return lambda eta: inv_cloglog_pt(eta)
    elif link_lower == "loglog":
        return lambda eta: inv_loglog_pt(eta)
    elif link_lower == "plogit":
        if r is None:
            raise ValueError("plogit requires r parameter")
        return lambda eta: inv_plogit_pt(eta, r)
    elif link_lower == "splogit":
        if r is None:
            raise ValueError("splogit requires r parameter")
        return lambda eta: inv_splogit_pt(eta, r)
    elif link_lower == "lpe":
        if xi is None:
            raise ValueError("lpe requires xi parameter")
        return lambda eta: inv_lpe_pt(eta, xi)
    elif link_lower == "grg":
        return lambda eta: inv_grg_pt(eta)
    elif link_lower == "skewprobit":
        if lambda_skew is None:
            raise ValueError("skewprobit requires lambda_skew parameter")
        return lambda eta: inv_skewprobit_pt(eta, lambda_skew)
    elif link_lower == "rh":
        if gamma_rh is None or mu is None:
            raise ValueError("rh requires gamma_rh and mu parameters")
        return lambda eta: inv_rh_pt(eta, gamma_rh, mu)
    elif link_lower == "glogit":
        if alpha1 is None or alpha2 is None:
            raise ValueError("glogit requires alpha1 and alpha2 parameters")
        return lambda eta: inv_glogit_pt(eta, alpha1, alpha2)
    else:
        raise ValueError(f"Unknown link '{link}'. Choose from: logit, cloglog, loglog, plogit, splogit, lpe, grg, skewprobit, rh, glogit")


def build_irt_model(
    y: np.ndarray,
    link: LinkType,
    mu_loga: float = np.log(0.5),
    tau_loga: float = 25.0,
    lower_a: float = 0.25,
    upper_a: float = 0.75,
    c_b1: float = 0.01,          
    c_b2: float = 0.01,
    mu_b_ratio: float = 1.0,    
    fix_mu_b_zero: bool = False,
    mu_theta: float = 0.0,
    tau_theta: float = 1.0,
    fix_mu_mean: bool = True,
    fix_mu_sd: bool = True,
    mu_log_r: float = 0.0,
    tau_log_r: float = 1.0,
    c_epsilon1: float = 0.01,  
    c_epsilon2: float = 0.01,
    infer_p_epsilon: bool = True,
    fixed_p_epsilon: Optional[float] = None,
    alpha_p_epsilon: float = 1.0,  
    beta_p_epsilon: float = 1.0,  
    marginalize_skew: bool = False,  
    alpha_xi: float = 2.0,      
    beta_xi: float = 1.0,
    mu_lambda: float = 0.0,    
    tau_lambda: float = 1.0,
    hermite_order: int = 2,      
    mu_alpha1: float = 0.0,     
    tau_alpha1: float = 1.0,
    mu_alpha2: float = 0.0,      
    tau_alpha2: float = 1.0,
) -> tuple[pm.Model, dict]:
    assert y.ndim == 2, "y must be a 2D array"
    assert y.dtype in (np.int8, np.int16, np.int32, np.int64), "y must contain integers"
    
    N, I = y.shape
    coords = {"person": np.arange(N), "item": np.arange(I)}
    
    with pm.Model(coords=coords) as model:
        loga = pm.TruncatedNormal(
            "loga",
            mu=mu_loga,
            sigma=pt.sqrt(1.0 / tau_loga),
            lower=np.log(lower_a + 1e-6),
            upper=np.log(upper_a - 1e-6),
            dims="item",
        )
        a = pm.Deterministic("a", pt.exp(loga), dims="item")
        
        tau_b = pm.Gamma("tau_b", alpha=c_b1, beta=c_b2)
        sigma_b = pt.sqrt(1.0 / tau_b)
        
        if fix_mu_b_zero:
            mu_b = pm.Deterministic("mu_b", pt.as_tensor_variable(0.0))
        else:
            mu_b = pm.Normal("mu_b", mu=0.0, sigma=pt.sqrt(mu_b_ratio) * sigma_b)
            
        # Non-Centered Parameterization (NCP) for b
        b_offset = pm.Normal("b_offset", mu=0.0, sigma=1.0, dims="item")
        b = pm.Deterministic("b", mu_b + b_offset * sigma_b, dims="item")
        
        if fix_mu_mean or fix_mu_sd:
            mu_raw = pm.Normal(
                "mu_raw",
                mu=mu_theta,
                sigma=pt.sqrt(1.0 / tau_theta),
                dims="person"
            )
            mu_centered = mu_raw - pt.mean(mu_raw)
            
            if fix_mu_sd:
                mu_scale = pt.sqrt(pt.var(mu_centered) + 1e-8)
                mu_id = mu_centered / mu_scale
            else:
                mu_id = mu_centered
            
            mu = pm.Deterministic("mu", mu_id, dims="person")
        else:
            mu = pm.Normal(
                "mu",
                mu=mu_theta,
                sigma=pt.sqrt(1.0 / tau_theta),
                dims="person"
            )
        
        uses_r = link.lower() in {"plogit", "splogit"}
        if uses_r:
            log_r = pm.Normal("log_r", mu=mu_log_r, sigma=pt.sqrt(1.0 / tau_log_r))
            r_rv = pm.Deterministic("r", pt.exp(log_r))
        else:
            r_rv = None

        xi_rv = None
        lambda_skew_rv = None
        hermite_coeffs_rv = None
        alpha1_rv = None
        alpha2_rv = None

        if link.lower() == "lpe":
            xi_rv = pm.Gamma("xi", alpha=alpha_xi, beta=beta_xi)

        elif link.lower() == "skewprobit":
            lambda_skew_rv = pm.Normal("lambda_skew", mu=mu_lambda, sigma=pt.sqrt(1.0 / tau_lambda))

        elif link.lower() == "rh":
            gamma_rh_rv = pm.Normal("gamma_rh", mu=0.0, sigma=0.25)

        elif link.lower() == "glogit":
            alpha1_rv = pm.Normal("alpha1", mu=mu_alpha1, sigma=pt.sqrt(1.0 / tau_alpha1))
            alpha2_rv = pm.Normal("alpha2", mu=mu_alpha2, sigma=pt.sqrt(1.0 / tau_alpha2))

        eta_base = (mu[:, None] - b[None, :]) * a[None, :]

        if link.lower() == "splogit":
            tau_epsilon = pm.Gamma("tau_epsilon", alpha=c_epsilon1, beta=c_epsilon2)

            if infer_p_epsilon:
                p_epsilon = pm.Beta("p_epsilon", alpha=alpha_p_epsilon, beta=beta_p_epsilon)
            else:
                if fixed_p_epsilon is None:
                    raise ValueError("fixed_p_epsilon must be provided when infer_p_epsilon=False")
                p_epsilon = pm.Deterministic("p_epsilon", pt.as_tensor_variable(fixed_p_epsilon))

            if marginalize_skew:

                inv_link = get_inverse_link_pt(link, r=r_rv)
                sigma_eps = pt.sqrt(1.0 / tau_epsilon)

                p0 = inv_link(eta_base)
                loglik0 = y * pt.log(pt.clip(p0, 1e-10, 1 - 1e-10)) + (1 - y) * pt.log(pt.clip(1 - p0, 1e-10, 1 - 1e-10))
                loglik0 = loglik0 + pt.log(pt.clip(1 - p_epsilon, 1e-10, 1.0))

                n_quad = 2
                gh_points_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_quad)
                gh_points_scaled = pt.as_tensor_variable(gh_points_np * np.sqrt(2.0))
                gh_weights_norm = pt.as_tensor_variable(gh_weights_np / np.sqrt(np.pi))
                
                eps_val = gh_points_scaled[:, None, None] * sigma_eps
                eta_skew = eta_base[None, :, :] + eps_val
                p1 = inv_link(eta_skew)
                
                y_expand = pt.shape_padleft(y)
                loglik_i = y_expand * pt.log(pt.clip(p1, 1e-10, 1 - 1e-10)) + (1 - y_expand) * pt.log(pt.clip(1 - p1, 1e-10, 1 - 1e-10))
                loglik_i = loglik_i + pt.log(gh_weights_norm[:, None, None])
                
                loglik1 = pt.logsumexp(loglik_i, axis=0)
                loglik1 = loglik1 + pt.log(pt.clip(p_epsilon, 1e-10, 1.0))
                loglik_total = pt.logsumexp(pt.stack([loglik0, loglik1], axis=-1), axis=-1)
                pm.Potential("y_loglik", pt.sum(loglik_total))
            else:
                z = pm.Bernoulli("z", p=p_epsilon, dims=("person", "item"))
                epsilon = pm.Normal(
                    "epsilon",
                    mu=0.0,
                    sigma=pt.sqrt(1.0 / tau_epsilon),
                    dims=("person", "item")
                )
                eta = eta_base + z * epsilon
                inv_link = get_inverse_link_pt(link, r=r_rv)
                p = inv_link(eta)
                p = pt.clip(p, 1e-10, 1.0 - 1e-10)
                pm.Bernoulli("y", p=p, observed=y, dims=("person", "item"))
        else:
            inv_link = get_inverse_link_pt(link, r=r_rv, xi=xi_rv,
                                          lambda_skew=lambda_skew_rv,
                                          gamma_rh=gamma_rh_rv if link.lower() == "rh" else None,
                                          alpha1=alpha1_rv, alpha2=alpha2_rv, mu=mu)
            p = inv_link(eta_base)
            p = pt.clip(p, 1e-10, 1.0 - 1e-10)
            pm.Bernoulli("y", p=p, observed=y, dims=("person", "item"))
    
    return model, coords


# ============================================================================
# Step Method Construction
# ============================================================================

def _build_step(
    model: pm.Model,
    sampler: SamplerType,
    target_accept: float = 0.9,
    step_kwargs: Optional[dict[str, Any]] = None
) -> pm.step_methods.arraystep.BlockedStep | list:
    sampler_lower = sampler.lower()
    step_kwargs = step_kwargs or {}
    nv = model.named_vars
    steps = []
    
    if sampler_lower in {"nuts", "hmc"}:
        nuts_kwargs = {k: v for k, v in step_kwargs.items() if k != "target_accept"}
        steps.append(pm.NUTS(target_accept=target_accept, **nuts_kwargs))
        
        if "z" in nv:
            steps.append(pm.BinaryGibbsMetropolis(vars=[nv["z"]]))
    
    elif sampler_lower in {"gibbs", "slice"}:
        blocks = []
        
        def add_block(var_names: list[str]):
            block = []
            for name in var_names:
                if name == "mu" and "mu_raw" in nv:
                    continue
                if name in nv and nv[name] in model.free_RVs:
                    block.append(nv[name])
            if block:
                blocks.append(block)
        
        add_block(["loga"])         
        add_block(["b"])            
        add_block(["mu_raw"])       
        add_block(["log_r"])       
        add_block(["epsilon"])      
        add_block(["tau_b", "tau_epsilon", "mu_b"])  
        
        default_widths = {
            0: 1.0,    
            1: 1.0,    
            2: 1.0,    
            3: 0.03,   
            4: 0.25,   
            5: 1.0,    
        }
        
        for i, block in enumerate(blocks):
                width_key = f"w_block_{i}"
                width = step_kwargs.get(width_key, default_widths.get(i, 1.0))
                steps.append(pm.Slice(vars=block, w=width))
        
        if "z" in nv:
            steps.append(pm.BinaryGibbsMetropolis(vars=[nv["z"]]))
    
    elif sampler_lower in {"mh", "metropolis"}:
        cont_vars = []
        var_order = ["loga", "tau_b", "mu_b", "b", "mu_raw", "mu", "tau_epsilon", "log_r", "epsilon"]
        
        for name in var_order:
            if name == "mu" and "mu_raw" in nv:
                continue  
            if name in nv and nv[name] in model.free_RVs:
                cont_vars.append(nv[name])
        
        if cont_vars:
            steps.append(pm.Metropolis(vars=cont_vars, **step_kwargs))

        if "z" in nv:
            steps.append(pm.BinaryGibbsMetropolis(vars=[nv["z"]]))
    
    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. "
            f"Choose from: nuts, hmc, gibbs, slice, mh, metropolis"
        )
    
    if not steps:
        raise ValueError("No step methods were created")
    
    return steps if len(steps) > 1 else steps[0]


def run_mcmc(
    y: np.ndarray,
    link: LinkType,
    sampler: SamplerType = "nuts",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.95,
    random_seed: int = 42,
    step_kwargs: Optional[dict[str, Any]] = None,
    progressbar: bool = False,
    silence_output: bool = False,
    **model_kwargs
) -> pm.backends.base.MultiTrace:
    import logging
    import warnings

    model, _ = build_irt_model(y, link, **model_kwargs)

    if silence_output:
        import sys
        import io

        old_pymc_level = logging.getLogger("pymc").level
        old_arviz_level = logging.getLogger("arviz").level
        old_stderr = sys.stderr
        old_stdout = sys.stdout

        logging.getLogger("pymc").setLevel(logging.CRITICAL)
        logging.getLogger("arviz").setLevel(logging.CRITICAL)

        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with model:
                step = _build_step(model, sampler, target_accept, step_kwargs)
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    random_seed=random_seed,
                    step=step,
                    progressbar=progressbar,
                )

        sys.stderr = old_stderr
        sys.stdout = old_stdout
        logging.getLogger("pymc").setLevel(old_pymc_level)
        logging.getLogger("arviz").setLevel(old_arviz_level)
    else:
        with model:
            step = _build_step(model, sampler, target_accept, step_kwargs)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                step=step,
                progressbar=progressbar,
            )

    return idata


def sample_nuts(
    y: np.ndarray,
    link: LinkType,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.95,
    random_seed: int = 42,
    **model_kwargs
) -> pm.backends.base.MultiTrace:
    return run_mcmc(
        y, link, sampler="nuts",
        draws=draws, tune=tune, chains=chains, cores=cores,
        target_accept=target_accept, random_seed=random_seed,
        **model_kwargs
    )


def sample_gibbs(
    y: np.ndarray,
    link: LinkType,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    step_kwargs: Optional[dict[str, Any]] = None,
    **model_kwargs
) -> pm.backends.base.MultiTrace:
    
    return run_mcmc(
        y, link, sampler="gibbs",
        draws=draws, tune=tune, chains=chains, cores=cores,
        random_seed=random_seed, step_kwargs=step_kwargs,
        **model_kwargs
    )


def sample_mh(
    y: np.ndarray,
    link: LinkType,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    step_kwargs: Optional[dict[str, Any]] = None,
    **model_kwargs
) -> pm.backends.base.MultiTrace:
    return run_mcmc(
        y, link, sampler="mh",
        draws=draws, tune=tune, chains=chains, cores=cores,
        random_seed=random_seed, step_kwargs=step_kwargs,
        **model_kwargs
    )


def save_mcmc_posterior(
    idata: az.InferenceData,
    save_path: str,
    save_full: bool = False
) -> None:
    if save_full:
        idata.to_netcdf(save_path)
    else:
        posterior_dict = {}
        for var_name in idata.posterior.data_vars:
            posterior_dict[var_name] = idata.posterior[var_name].values

        np.savez_compressed(save_path, **posterior_dict)


def save_mcmc_config(
    config: dict[str, Any],
    save_path: str
) -> None:
    """
    Save MCMC configuration parameters to a JSON file.
    """
    with open(save_path, 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, type(None), list, dict)):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        json.dump(serializable_config, f, indent=4)
    print(f"Saved MCMC configuration to {save_path}")





def _run_single_mcmc_worker(args):
    i, key, y_i, seed_i, est_link, sampler, draws, tune, chains, mcmc_kwargs, save_path = args
    if save_path is not None and Path(save_path).exists():
        return {'i': i, 'key': key, 'success': True, 'idata': None, 'skipped': True, 'error': None}
        
    worker_id = multiprocessing.current_process().name
    
    if save_path is not None:
        log_dir = Path(save_path).parent.parent / "worker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{worker_id}.log"
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)
        f = open(log_file, "a")
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    else:
        f = None
        
    try:
        if f is not None:
            print(f"\n{'='*50}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {key} rep {i}")
            sys.stdout.flush()
            
        idata = run_mcmc(
            y=y_i,
            link=est_link,
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,  # Always 1 in parallel mode
            random_seed=seed_i,
            progressbar=False,
            silence_output=True,
            **mcmc_kwargs,
        )
        if save_path is not None:
            save_mcmc_posterior(idata, save_path, save_full=False)
            
        if f is not None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISHED: {key} rep {i}\n{'='*50}\n")
            sys.stdout.flush()
            
        return {'i': i, 'key': key, 'success': True, 'idata': idata, 'skipped': False, 'error': None}
    except Exception as e:
        if f is not None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR in {key} rep {i}: {e}\n{'='*50}\n")
            sys.stdout.flush()
        return {'i': i, 'key': key, 'success': False, 'idata': None, 'skipped': False, 'error': str(e)}
    finally:
        if f is not None:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            f.close()


def run_mcmc_pipeline(
    N_reps: int,
    est_links: list[str],
    sim_links: list[str] | None = None,
    sim_rs: list[float] | None = None,
    data_type: Literal["simulation", "real_data"] = "simulation",
    sampler: str = "nuts",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    n_jobs: int = 1,
    root_dir: Optional[str] = None,
    **mcmc_kwargs
) -> list[az.InferenceData]:
    simulation_dir = os.path.join(root_dir, 'data', 'simulation')
    
    actual_sim_links = sim_links if (data_type == "simulation" and sim_links is not None) else ["none"]
    actual_sim_rs = sim_rs if (data_type == "simulation" and sim_rs is not None) else [0.0]

    all_idatas = {}
    all_tasks = []

    for est_link in est_links:
        for sim_link in actual_sim_links:
            for sim_r in actual_sim_rs:
                if data_type == "simulation":
                    print(f"[{data_type}] Prepping MCMC sampling tasks for est_link: {est_link}, sim_link: {sim_link}, sim_r: {sim_r}")
                else:
                    print(f"[{data_type}] Prepping MCMC sampling tasks for est_link: {est_link}")

                # Load data for each replication
                y_list = []
                if data_type == "simulation":
                    for i in range(N_reps):
                        data_folder = f"simulation_data_{sim_link}_r_{sim_r}"
                        data_file = f"simulation_data_{sim_link}_r_{sim_r}_rep_{i}.npz"
                        sim_data_path = os.path.join(
                            simulation_dir,
                            data_folder,
                            data_file
                        )
                        y_list.append(np.load(sim_data_path)['y'])
                else:
                    real_data_path = os.path.join(root_dir, "data/real_data/real_data.npz")
                    y = np.load(real_data_path)['y']
                    y_list = [y for _ in range(N_reps)]

                R = N_reps
                seed_list = [
                    generate_seed(mode="mcmc_sampling", link=est_link, r=sim_r, rep=i) for i in range(R)
                ]

                # Move folder creation here so we can save incrementally
                if data_type == "simulation":
                    folder_name = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}"
                    folder_path = Path(f"{root_dir}/outputs/{data_type}/samplings/{folder_name}")
                    folder_path.mkdir(parents=True, exist_ok=True)
                    filename_prefix = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}_rep_"
                else:
                    folder_name = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}"
                    folder_path = Path(f"{root_dir}/outputs/{data_type}/samplings/{folder_name}")
                    folder_path.mkdir(parents=True, exist_ok=True)
                    filename_prefix = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}_rep_"

                key = f"simlink_{sim_link}_r_{sim_r}_estlink_{est_link}" if data_type == "simulation" else f"estlink_{est_link}"
                all_idatas[key] = [None] * R
                
                # Save the configuration used for this run
                if data_type == "simulation":
                    config = {
                        "sim_link": sim_link,
                        "sim_r": sim_r,
                        "est_link": est_link,
                        "data_type": data_type,
                        "sampler": sampler,
                        "draws": draws,
                        "tune": tune,
                        "chains": chains,
                        "n_reps": R,
                        "mcmc_kwargs": mcmc_kwargs
                    }
                    config_filename = f"config_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}.json"
                else:
                    config = {
                        "est_link": est_link,
                        "data_type": data_type,
                        "sampler": sampler,
                        "draws": draws,
                        "tune": tune,
                        "chains": chains,
                        "n_reps": R,
                        "mcmc_kwargs": mcmc_kwargs
                    }
                    config_filename = f"config_{data_type}_estlink_{est_link}.json"
                    
                config_dir = Path(f"{root_dir}/outputs/{data_type}/config")
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(config_dir / config_filename, "w") as f:
                    json.dump(config, f, indent=4)
                    
                for i in range(R):
                    save_path = folder_path / f"{filename_prefix}{i}.npz"
                    all_tasks.append(
                        (i, key, y_list[i], seed_list[i], est_link, sampler, draws, tune, chains, mcmc_kwargs, save_path)
                    )

    actual_cores = 1 if n_jobs > 1 else cores

    if n_jobs == 1 or len(all_tasks) == 1:
        with tqdm(total=len(all_tasks), desc="Progress", unit="rep", dynamic_ncols=True) as pbar:
            for task in all_tasks:
                i, key, y_i, seed_i, est_link, sampler, draws, tune, chains, mcmc_kwargs, save_path = task
                
                if key.startswith("simlink_"):
                    parts = key.split('_')
                    sim_link = parts[1]
                    sim_r = parts[3]
                    est_link = parts[5]
                else:
                    est_link = key.split('_')[1]
                
                if save_path is not None and Path(save_path).exists():
                    pbar.set_postfix_str(msg)
                    pbar.update(1)
                    continue
                    
                idata = run_mcmc(
                    y=y_i, link=est_link, sampler=sampler, draws=draws, tune=tune,
                    chains=chains, cores=actual_cores, random_seed=seed_i,
                    progressbar=(len(all_tasks) == 1), silence_output=(len(all_tasks) > 1),
                    **mcmc_kwargs
                )
                if save_path is not None:
                    save_mcmc_posterior(idata, save_path, save_full=False)
                all_idatas[key][i] = idata
                pbar.set_postfix_str(msg)
                pbar.update(1)
    else:
        with Pool(processes=n_jobs) as pool:
            with tqdm(total=len(all_tasks), desc="Progress", dynamic_ncols=True) as pbar:
                for res in pool.imap_unordered(_run_single_mcmc_worker, all_tasks):
                    key = res['key']
                    i = res['i']
                    if res['success']:
                        if not res.get('skipped', False):
                            all_idatas[key][i] = res['idata']
                    else:
                        print(f"\nERROR in {key} replication {i}: {res['error']}")
                    
                    if key.startswith("simlink_"):
                        parts = key.split('_')
                        sim_link = parts[1]
                        sim_r = parts[3]
                        est_link = parts[5]
                        msg = f"{sim_link}, {sim_r}, {est_link}, {i}"
                    else:
                        est_link = key.split('_')[1]
                        msg = f"{est_link}, {i}"
                        
                    pbar.set_postfix_str(msg)
                    pbar.update(1)
                    
    return all_idatas

def main(
    N,
    sim_links,
    sim_rs,
    est_links,
    data_type,
    sampler,
    draws,
    tune,
    chains,
    cores,
    n_jobs
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    print("Starting batch MCMC sampling...\n")

    all_idatas = run_mcmc_pipeline(
        N_reps=N,
        sim_links=sim_links,
        sim_rs=sim_rs,
        est_links=est_links,
        data_type=data_type,
        sampler=sampler,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        n_jobs = n_jobs,
        root_dir = project_root,
    )

if __name__ == "__main__":
    start_time = time.time()
    data_type = "simulation"
    N = 100
    sim_links = ["splogit"]
    sim_rs = [0.25, 0.5, 1.0, 2.0, 4.0]
    est_links = ["splogit"]
    n_jobs = os.cpu_count()
    # n_jobs = 1
    
    # data_type = "real_data"
    # N = 1
    # sim_links = None
    # sim_rs = None
    # est_links = ["logit", "cloglog", "loglog", "plogit", "splogit", "lpe", "grg", "skewprobit", "rh", "glogit"]
    # n_jobs = 1
    
    
    sampler = "nuts"
    draws = 1000
    tune = 1000
    chains = 4
    cores = 4


    main(
        N = N,
        sim_links = sim_links,
        sim_rs = sim_rs,
        est_links = est_links,
        data_type = data_type,
        sampler = sampler,
        draws = draws,
        tune = tune,
        chains = chains,
        cores = cores,
        n_jobs = n_jobs
    )
    end_time = time.time() 
    total_seconds = end_time - start_time
    minutes, seconds = divmod(total_seconds, 60)