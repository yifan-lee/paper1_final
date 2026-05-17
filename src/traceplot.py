import os
import sys
import json
import numpy as np
import arviz as az
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib as mpl

from simulate_parameters import Params

# Set matplotlib fonts and sizes to be publication-ready
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42

def generate_all_traceplots(
    data_type: str,
    sim_link: str,
    sim_r: float,
    est_links: list[str],
    project_root: str,
    dpi: int = 150
):
    """
    Main orchestrator to load MCMC results and plot them.
    In simulation mode, loads true parameters and adds them to plots.
    """
    out_dir = os.path.join(project_root, "outputs", data_type)
    samplings_dir = os.path.join(out_dir, "samplings")
    true_params = None
    if data_type == "simulation":
        # Import Params locally to avoid circular imports
        params_path = os.path.join(project_root, "data", "parameters.npz")
        if os.path.exists(params_path):
            true_params = Params.load(params_path)
            print(f"Loaded true parameters from {params_path}")

    for est_link in est_links:
        print(f"\nProcessing estimation link: '{est_link}'...")
        
        if data_type == "simulation":
            config_filename = f"config_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}.json"
        else:
            config_filename = f"config_{data_type}_estlink_{est_link}.json"
        
        config_path = os.path.join(out_dir, "config", config_filename)
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        draws = config['draws']
        N_reps = config['n_reps']

        if data_type == "simulation":
            folder_name = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}"
        else:
            folder_name = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}"

        figures_dir = Path(f"{project_root}/figures/{data_type}/{folder_name}")
        figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading and plotting {N_reps} MCMC results...")
    
        for rep_idx in range(N_reps):
            if data_type == "simulation":
                filename = f"samplings_{data_type}_simlink_{sim_link}_{sim_r}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
            else:
                filename = f"samplings_{data_type}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.npz"
                
            file_path = os.path.join(samplings_dir, folder_name, filename)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            data = np.load(file_path)
            idata = az.from_dict(posterior=dict(data))
            
            print(f"Generating individual traceplots for {est_link} rep {rep_idx}...")
            _create_individual_traceplot(
                idata=idata, rep_idx=rep_idx, sim_link=sim_link, r_val=sim_r, 
                est_link=est_link, draws=draws, figures_dir=figures_dir, 
                true_params=true_params, dpi=dpi
            )
        
    print("\nDone! All individual traceplots generated.")


def _create_individual_traceplot(
    idata: az.InferenceData,
    rep_idx: int,
    sim_link: str,
    r_val: float,
    est_link: str,
    draws: int,
    figures_dir: str,
    true_params = None,
    dpi: int = 150,
) -> None:

    figure_name = f"traces_simlink_{sim_link}_r_{r_val}_estlink_{est_link}_draw_{draws}_rep_{rep_idx}.pdf"
    out_file = os.path.join(figures_dir, figure_name)
    
    with PdfPages(out_file) as pdf:
        all_vars = list(idata.posterior.data_vars)
        
        priority_vars = ["r", "tau_epsilon", "p_epsilon"]
        plot_vars = []
        for p_var in priority_vars:
            if p_var in all_vars:
                plot_vars.append(p_var)
                
        for var in all_vars:
            if var not in priority_vars and var not in ["log_r", "mu_raw", "epsilon", "z", "a"]:
                plot_vars.append(var)

        for var_name in plot_vars:
            coords = None
            if var_name in ["a", "b", "loga"]:
                coords = {f"{var_name}_dim_0": [0, 1, 2, 3, 4, 5]} # subset 6 items
            elif var_name == "mu":
                coords = {"mu_dim_0": [0, 1, 2, 3, 4, 5]} # subset 6 persons
                
            axes = az.plot_trace(idata, var_names=[var_name], coords=coords, compact=False)
            fig = axes[0, 0].figure
            fig.set_size_inches(10, 2.0 * axes.shape[0])
            
            # Extract true value mapping
            true_val = None
            if true_params is not None:
                if var_name == "r":
                    true_val = r_val
                elif var_name == "loga" and hasattr(true_params, "a"):
                    true_val = np.log(true_params.a)
                elif var_name == "a" and hasattr(true_params, "a"):
                    true_val = true_params.a
                elif var_name == "b" and hasattr(true_params, "b"):
                    true_val = true_params.b
                elif var_name == "mu" and hasattr(true_params, "mu"):
                    true_val = true_params.mu
                elif var_name == "tau_epsilon" and hasattr(true_params, "hyper_params"):
                    true_val = getattr(true_params.hyper_params, "tau_epsilon", None)
                elif var_name == "p_epsilon" and hasattr(true_params, "p_epsilon"):
                    true_val = true_params.p_epsilon
                elif var_name == "tau_b" and hasattr(true_params, "hyper_params"):
                    true_val = getattr(true_params.hyper_params, "tau_b", None)
                elif var_name == "mu_b" and hasattr(true_params, "hyper_params"):
                    true_val = getattr(true_params.hyper_params, "mu_b", None)
            
            # Add true values if available
            if true_val is not None:
                if np.isscalar(true_val) or np.asarray(true_val).ndim == 0:
                    axes[0][0].axvline(true_val, color='r', linestyle='--', alpha=0.8, label='True Value')
                    if len(axes[0]) > 1: axes[0][1].axhline(true_val, color='r', linestyle='--', alpha=0.8)
                elif coords and f"{var_name}_dim_0" in coords:
                    subset_indices = coords[f"{var_name}_dim_0"]
                    for j, idx in enumerate(subset_indices):
                        if j < axes.shape[0]:  
                            val = np.asarray(true_val)[idx]
                            axes[j][0].axvline(val, color='r', linestyle='--', alpha=0.8)
                            if len(axes[j]) > 1: axes[j][1].axhline(val, color='r', linestyle='--', alpha=0.8)

            fig.suptitle(f"Rep {rep_idx}: Traceplot for '{var_name}'", y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
            plt.close(fig)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_type = "simulation"
    sim_link = "plogit"
    sim_r = 4.0
    est_links = ["plogit"]

    generate_all_traceplots(
        data_type=data_type,
        sim_link=sim_link,
        sim_r=sim_r,
        est_links=est_links,
        project_root=project_root,
        dpi=150
    )
    