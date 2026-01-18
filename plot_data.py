import os
import numpy as np
import matplotlib.pyplot as plt
import csv, re
from pathlib import Path

INPUT_PARAMETERS = {
    'base_directory': '/Users/abhishek/work/free_surface_2025/wave_tank/15_degree_slope/PostProcessing/csv/epsilon_turb_avg_Z_new',
    'variable': 'epsilon_turb_avg_Z'
}

cfg = dict(INPUT_PARAMETERS)

dir = os.path.join(cfg.get("base_directory"),cfg.get("variable"))

def main():
    epsilon_plot(dir, cfg)

def load_cols(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = {h: [] for h in r.fieldnames}
        for row in r:
            for h in cols:
                cols[h].append(float(row[h]))
    return cols

def flux_plot():
    # Define the path and filename
    path = "/Users/abhishek/work/free_surface_2025/wang_kraus_scaled/elongated/data"
    filename = "eflux_x25.dat"

    # Join the directory and filename to make a full path
    filepath = os.path.join(path, filename)

    # Load data (skip the first row, assuming space-separated columns)
    data = np.loadtxt(filepath, skiprows=1)

    # Extract columns
    t = data[:, 0]  # 1st column
    eflux = data[:, 1]  # 4th column
    
    
    # Plot
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, eflux, marker='o', linestyle='-', label='Energy Flux at x=25m')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Total Energy Flux (${m}^3/{s}^3$)')
    plt.title('Plot of Total FLux with TIme')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def epsilon_plot(path, cfg):
    folder = Path(path)
    pat = re.compile(r"_t_([0-9.]+)\.csv$")
    
    pairs = []
    for p in folder.glob("*_t_*.csv"):
        m = pat.search(p.name)
        if m:
            pairs.append((float(m.group(1)), p))
    
    pairs.sort(key=lambda tp: tp[0])  # sort by time
    
    t = []
    eps = []
    for ti, p in pairs:
        t.append(ti)
        cols = load_cols(p)
        alpha = np.array(cols["alpha.water_avg_Y"])
        if 'turb' in cfg.get("variable"):
            eps_turb = np.array(cols["epsilon_turb_avg_Z"])
        elif 'eps' in cfg.get("variable"):
            eps_turb = np.array(cols["epsilon_avg_Z"])
        print("max eps_turb", max(eps_turb), "at t=", ti)
        eps_turb[np.where(alpha <=0.2)] = 'nan'
        #eps_turb[np.where(eps_turb >= 7)] = 'nan'
        x = np.array(cols["X"])
        eps.append(eps_turb)
    if not eps:
        raise FileNotFoundError("No matching x_t_*.csv files found.")
    
    
    n = len(eps[0])
    if any(len(x) != n for x in eps):
        raise ValueError("Not all X columns have the same length across files.")
    eps_avg = np.nanmean(np.array(eps, dtype=float), axis=0)
    eps_index = np.where(eps_avg >= 1e-6)
    x = x[eps_index]
    eps_avg = eps_avg[eps_index]
    xindex = np.where(x >= 7.2)
    x = x[xindex]
    eps_avg = eps_avg[xindex]
    
    # Plot
    if 'turb' in cfg.get("variable"):
        plot_label = r'Time Averaged and depth averaged $\epsilon$ ($\epsilon$)'
        plot_ylabel = r'$\epsilon$ (${m}^2/{s}^3$)'
        plot_title = 'Spatial distribution of time averaged and depth averaged turbulence dissipation rate'
    elif 'eps' in cfg.get("variable"):
        plot_label = r'Time Averaged and depth averaged $\epsilon$'
        plot_ylabel = r'$\epsilon$ (${m}^2/{s}^3$)'
        plot_title = 'Spatial distribution of time averaged and depth averaged dissipation rate'
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, eps_avg, linestyle='-')
    plt.xlabel('X (m)')
    plt.ylabel(plot_ylabel)
    plt.title(plot_title)
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
main()