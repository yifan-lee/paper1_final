from typing import Optional, Literal
import numpy as np
import os
from pathlib import Path
from simulate_parameters import Params
from utils import generate_seed
LinkType = Literal["logit", "cloglog", "loglog", "plogit", "splogit",
                    "lpe", "grg", "skewprobit", "rh", "glogit"]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def inv_logit(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return _sigmoid(eta)

def inv_cloglog(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))

def inv_loglog(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return np.exp(-np.exp(np.clip(-eta, -20, 20)))

def inv_plogit(eta: np.ndarray, r: float | np.ndarray) -> np.ndarray:
    if np.any(r <= 0):
        raise ValueError("r must be positive for plogit")
    return np.where(
        r <= 1.0,
        _sigmoid(eta / r) ** r,
        1.0 - _sigmoid(-r * eta) ** (1.0 / r)
    )

def inv_splogit(eta: np.ndarray, r: float) -> np.ndarray:
    return inv_plogit(eta, r)

def inv_lpe(eta: np.ndarray, xi: float | np.ndarray) -> np.ndarray:
    if np.any(xi <= 0):
        raise ValueError("xi must be positive for lpe")
    logit_p = _sigmoid(eta)
    return np.power(logit_p, xi)

def inv_grg(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    gumbel_max = np.exp(-np.exp(np.clip(-eta, -20, 20)))
    reverse_gumbel = 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))
    return 0.5 * (gumbel_max + reverse_gumbel)

def _skew_normal_cdf(x: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    from scipy.stats import skewnorm
    return skewnorm.cdf(x, alpha)

def inv_skewprobit(eta: np.ndarray, lambda_skew: float) -> np.ndarray:
    return _skew_normal_cdf(eta, lambda_skew)

def inv_rh(eta: np.ndarray, gamma_rh: float | np.ndarray, mu: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    grh_arr = np.asarray(gamma_rh)
    if grh_arr.ndim > 0 and grh_arr.ndim < mu.ndim:
        target_shape = list(grh_arr.shape) + [1] * (mu.ndim - grh_arr.ndim)
        grh_arr = grh_arr.reshape(target_shape)
    
    if mu.ndim == eta.ndim:
        mu_exp = mu
    elif mu.ndim == eta.ndim - 1:
        mu_exp = mu[..., None]
    else:
        mu_exp = mu
        
    scaled_eta = eta / np.sqrt(np.exp(grh_arr * mu_exp))
    return norm.cdf(scaled_eta)

def _h_alpha_positive(eta: np.ndarray, alpha: float) -> np.ndarray:
    eps = 1e-10
    if abs(alpha) < eps:
        return eta
    elif alpha > 0:
        return (np.exp(alpha * eta) - 1.0) / alpha
    else:
        max_eta = (1.0 / abs(alpha)) - eps
        eta_clipped = np.clip(eta, None, max_eta)
        return -np.log(1.0 - alpha * eta_clipped) / alpha

def _h_alpha_negative(eta: np.ndarray, alpha: float) -> np.ndarray:
    eps = 1e-10
    eta_abs = np.abs(eta)
    if abs(alpha) < eps:
        return eta
    elif alpha > 0:
        return -(np.exp(alpha * eta_abs) - 1.0) / alpha
    else:
        max_eta_abs = (1.0 / abs(alpha)) - eps
        eta_abs_clipped = np.clip(eta_abs, None, max_eta_abs)
        return np.log(1.0 - alpha * eta_abs_clipped) / alpha

def inv_glogit(eta: np.ndarray, alpha1: float, alpha2: float) -> np.ndarray:
    h_eta = np.zeros_like(eta, dtype=float)
    positive_mask = eta > 0
    nonpositive_mask = ~positive_mask
    if np.any(positive_mask):
        h_eta[positive_mask] = _h_alpha_positive(eta[positive_mask], alpha1)
    if np.any(nonpositive_mask):
        h_eta[nonpositive_mask] = _h_alpha_negative(eta[nonpositive_mask], alpha2)
    return _sigmoid(h_eta)

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

def generate_responses(
    params: Params,
    link: LinkType,
    r_val: float = 0,
    return_prob: bool = False,
    rep: int = 0
) -> np.ndarray:
    if link not in LINK_FUNCTIONS:
        raise ValueError(f"Unknown link '{link}'.")
    
    N, I = len(params.mu), len(params.a)
    eta = (params.mu[:, None] - params.b[None, :]) * params.a[None, :]
    if link == "splogit":
        eta = eta + params.z * params.epsilon
        
    inv_link = LINK_FUNCTIONS[link]
    if link in {"plogit", "splogit"}:
        p = inv_link(eta, r_val)
    elif link == "lpe":
        p = inv_link(eta, params.xi)
    elif link == "skewprobit":
        p = inv_link(eta, params.lambda_skew)
    elif link == "rh":
        p = inv_link(eta, params.gamma_rh, params.mu)
    elif link == "glogit":
        p = inv_link(eta, params.alpha1, params.alpha2)
    else:
        p = inv_link(eta)
        
    p = np.clip(p, 1e-9, 1 - 1e-9).astype(np.float32)
    if return_prob:
        return p
        
    seed = generate_seed(mode="simulate_irtdata", link=link, r=r_val, rep=rep)
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=p, size=(N, I)).astype(np.int8)

def generate_all_simulation_data(
    params: Params,
    link: str,
    r_val: float,
    N_reps: int,
    base_dir: str = "data/simulation"
):
    """
    Generate and save N_reps simulation datasets, each with a unique seed based on rep index.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    for rep in range(N_reps):
        y = generate_responses(
            params=params,
            link=link,
            r_val=r_val,
            rep=rep
        )
        folder_path = Path(f"{base_dir}/simulation_data_{link}_r_{r_val}")
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"simulation_data_{link}_r_{r_val}_rep_{rep}.npz"
        file_path = folder_path / filename
        np.savez(file_path, y=y)
        print(f"✅ Simulation data rep {rep} saved to {file_path}")



if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    params_path = os.path.join(project_root, "data/parameters.npz")
    params = Params.load(params_path)

    link = "cloglog"
    r = 0.0
    N_reps = 100

    data_path = os.path.join(
        project_root,
        "data/simulation"
    )

    generate_all_simulation_data(
        params=params,
        link=link,
        r_val=r,
        N_reps=N_reps,
        base_dir=data_path
    )