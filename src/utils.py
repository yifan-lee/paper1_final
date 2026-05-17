import hashlib
from typing import Literal, Union, Optional

ModeType = Literal["simulate_parameter", "simulate_irtdata", "mcmc_sampling"]

def generate_seed(
    root_seed: int = 718,
    mode: ModeType = "simulate_parameter",
    link: Optional[str] = None,
    r: Optional[Union[int, float]] = None,
    rep: int = 0
) -> int:
    """
    Generate a deterministic seed based on the experimental parameters.
    
    Parameters
    ----------
    root_seed : int
        The base root seed for the experiment.
    mode : Literal["simulate_parameter", "simulate_irtdata", "mcmc_sampling"]
        The current step/mode in the pipeline.
    link : str or None
        The link function being used (e.g., 'plogit', 'logit', 'lpe'). Can be None if mode is 'simulate_parameter'.
    r : int, float, or None
        An additional parameter required when mode is 'simulate_irtdata' 
        and link is 'plogit' or 'splogit'.
        
    Returns
    -------
    int
        A deterministic 32-bit unsigned integer seed suitable for numpy or random.
    """
    if mode != "simulate_parameter" and link is None:
        raise ValueError(f"Parameter 'link' must be provided when mode is '{mode}'.")
    
    if mode == "simulate_irtdata" and link in ["plogit", "splogit"]:
        if r is None:
            raise ValueError(f"Parameter 'r' must be provided when mode is 'simulate_irtdata' and link is '{link}'.")
        # Format the float to avoid precision differences changing the hash
        r_str = f"{float(r):.4f}"
    else:
        r_str = "None"
        
    # Construct a unique string identifier for this specific condition
    link_str = link if link is not None else "None"
    unique_string = f"root:{root_seed}|mode:{mode}|link:{link_str}|r:{r_str}|rep:{rep}"
    
    # Use SHA-256 to create a deterministic hash of the unique string
    hash_digest = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
    
    # Return a 32-bit unsigned integer (using first 8 hex characters)
    # This fits perfectly into numpy.random.default_rng(seed) which accepts uint32.
    return int(hash_digest[:8], 16)
