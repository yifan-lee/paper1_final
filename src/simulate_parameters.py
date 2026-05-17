from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import os

from utils import generate_seed

@dataclass
class HyperParams:
    I: int = 500
    J: int = 40
    mu_a: float = np.log(0.5)
    tau_a: float = 25.0
    mu_b: float = 0.0
    tau_b: float = 1.0
    mu_theta: float = 0.0
    tau_theta: float = 1.0
    mu_log_r: float = 0.0
    tau_log_r: float = 1.0
    tau_epsilon: float = 25.0
    p_epsilon: float = 0.25
    alpha_xi: float = 2.0
    beta_xi: float = 1.0
    mu_lambda: float = 0.0
    tau_lambda: float = 1.0
    mu_alpha1: float = 0.0
    tau_alpha1: float = 1.0
    mu_alpha2: float = 0.0
    tau_alpha2: float = 1.0
    mu_gamma_rh: float = 0.0
    tau_gamma_rh: float = 1.0


@dataclass
class Params:
    hyper_params: HyperParams
    a: np.ndarray      
    b: np.ndarray      
    mu: np.ndarray    
    r: float 
    epsilon: np.ndarray
    z: np.ndarray       
    p_epsilon: float    
    xi: float            
    lambda_skew: float  
    gamma_rh: float 
    alpha1: float       
    alpha2: float    


    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(file_path, **asdict(self))
        print(f"✅ Parameters saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "Params":
        data = np.load(file_path, allow_pickle=True)
        kwargs = {}
        for k, v in data.items():
            if k == "hyper_params":
                kwargs[k] = HyperParams(**v.item())
            elif k in cls.__dataclass_fields__:
                kwargs[k] = v
        return cls(**kwargs)


def generate_all_parameters(
    hyper: Optional[HyperParams] = None,
) -> Params:
    """
    Generates all base and link-specific parameters across all models.
    """
    if hyper is None:
        hyper = HyperParams()

    seed = generate_seed(mode="simulate_parameter")
    rng = np.random.default_rng(seed)

    a = np.exp(
        rng.normal(
            loc=hyper.mu_a,
            scale=np.sqrt(1.0 / hyper.tau_a),
            size=hyper.J
        )
    ).astype(np.float32)

    b = rng.normal(
        loc=hyper.mu_b,
        scale=np.sqrt(1.0 / hyper.tau_b),
        size=hyper.J
    ).astype(np.float32)

    mu = rng.normal(
        loc=hyper.mu_theta,
        scale=np.sqrt(1.0 / hyper.tau_theta),
        size=hyper.I
    ).astype(np.float32)

    log_r = rng.normal(
        loc=hyper.mu_log_r,
        scale=np.sqrt(1.0 / hyper.tau_log_r)
    )
    r = float(np.exp(log_r))

    epsilon = rng.normal(
        loc=0.0,
        scale=np.sqrt(1.0 / hyper.tau_epsilon),
        size=(hyper.I, hyper.J)
    ).astype(np.float32)
    
    z = rng.binomial(
        n=1,
        p=hyper.p_epsilon,
        size=(hyper.I, hyper.J)
    ).astype(np.int8)

    xi = float(rng.gamma(shape=hyper.alpha_xi, scale=1.0/hyper.beta_xi))
    lambda_skew = float(rng.normal(loc=hyper.mu_lambda, scale=np.sqrt(1.0/hyper.tau_lambda)))
    gamma_rh = float(rng.normal(loc=hyper.mu_gamma_rh, scale=np.sqrt(1.0/hyper.tau_gamma_rh)))
    alpha1 = float(rng.normal(loc=hyper.mu_alpha1, scale=np.sqrt(1.0/hyper.tau_alpha1)))
    alpha2 = float(rng.normal(loc=hyper.mu_alpha2, scale=np.sqrt(1.0/hyper.tau_alpha2)))

    return Params(
        hyper_params=hyper,
        a=a, b=b, mu=mu, r=r, epsilon=epsilon, z=z, p_epsilon=hyper.p_epsilon,
        xi=xi, lambda_skew=lambda_skew, gamma_rh=gamma_rh,
        alpha1=alpha1, alpha2=alpha2
    )

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    params_path = os.path.join(project_root, "data/parameters.npz")
    hyper = HyperParams()
    params = generate_all_parameters(hyper=hyper)
    params.save(params_path)
