import os
import numpy as np
import matplotlib.pyplot as plt
import csv, re
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

INPUT_PARAMETERS = {
    'base_directory': '/Users/abhishek/work/free_surface_2025/wave_tank/15_degree_slope/PostProcessing/logs',
    'variable': 'epsilon_turb'
}

cfg = dict(INPUT_PARAMETERS)

dir = os.path.join(cfg.get("base_directory"), cfg.get("variable"))

# Define Plots

plt.figure(figsize=(8, 5))

def main():
    epsilon_plot(dir, cfg)

def csv_load_cols(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = {h: [] for h in r.fieldnames}
        for row in r:
            for h in cols:
                cols[h].append(float(row[h]))
    return cols

def time_avg(pairs, var_name, window=None):
    
    if window is not None:
        tmin, tmax = window[0], window[1]
    else:
        tmin, tmax = pairs[0][0], pairs[-1][0]
    # Collect all data with X positions
    data_dict = {}  # dict of {x_position: [eps_values_at_different_times]}
    
    for ti, p in pairs:
        
        if ti >= tmin and ti <= tmax:
            cols = load_cols(p)
            
            eps_turb = np.array(cols[var_name])
            x = np.array(cols["X"])
            
            print(f"max {var_name}", max(eps_turb), "at t=", ti)
            
            # Store epsilon values by X position
            for x_pos, eps_val in zip(x, eps_turb):
                if x_pos not in data_dict:
                    data_dict[x_pos] = []
                data_dict[x_pos].append(eps_val)
    
    # Convert to sorted arrays
    x_positions = sorted(data_dict.keys())
    x_array = np.array(x_positions)
    
    # Calculate average for each X position
    eps_avg = []
    for x_pos in x_positions:
        values = np.array(data_dict[x_pos])
        # Use nanmean to handle any NaN values
        avg_val = np.nanmean(values)
        
        eps_avg.append(avg_val)
    
    eps_avg = np.array(eps_avg)
    
    return eps_avg, x_array
    
def cleanup(eps,x):
    eps_index = np.where(eps >= 10 ** -9)
    x_array = x[eps_index]
    eps_avg = eps[eps_index]
    mask = np.where((x_array < 4) & (eps_avg >= 10**-4))
    x_array = np.delete(x_array, mask)
    eps_avg = np.delete(eps_avg, mask)
    return eps_avg, x_array, mask
def load_cols(path):
    # Load data, skipping comment lines
    data = np.loadtxt(path, comments='#')
    
    # Read header from comment line
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                header_text = line[1:].strip()
                break
        else:
            raise ValueError(f"No header found in {path}")
    
    # Remove anything in parentheses (units)
    header_text = re.sub(r'\([^)]*\)', '', header_text)
    
    # Split by whitespace and remove empty strings
    headers = [h for h in header_text.split() if h]
    
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Verify column count matches
    if len(headers) != data.shape[1]:
        raise ValueError(f"Header has {len(headers)} columns {headers} but data has {data.shape[1]} columns in {path}")
    
    # Create dictionary
    cols = {}
    for i, h in enumerate(headers):
        cols[h] = data[:, i].tolist()
    
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
    pat = re.compile(r"_([0-9.]+)\.dat$")
    
    pairs = []
    for p in folder.glob("*.dat"):
        
        m = pat.search(p.name)
        if m:
            pairs.append((float(m.group(1)), p))
        else:
            print(f"Not matched: {p.name}")
    pairs.sort(key=lambda tp: tp[0])  # sort by time
    
    if not pairs:
        raise FileNotFoundError("No matching .dat files found.")
        
    # Determine which variable to extract
    if 'turb' in cfg.get("variable"):
        var_name = "epsilon_turb_avg"
    elif 'eps' in cfg.get("variable"):
        var_name = "epsilon_avg_Z"
    else:
        raise ValueError("Variable type not recognized in config")
    t_list = [6.8, 8, 10, 12, 14]
    for i in range(len(t_list)):
        if i > 0:
            eps_avg, x_array = time_avg(pairs, var_name, window=[t_list[0], t_list[i]])
            eps_avg, x_array, mask = cleanup(eps_avg, x_array)
            plt.plot(x_array, gaussian_filter1d(eps_avg, sigma=2), linestyle='-',
                     label=f'Time Average Window-t={t_list[0]}s-{t_list[i]}s')
    # Plot
    if 'turb' in cfg.get("variable"):
        plot_label = r'Time Averaged and depth averaged $\epsilon$ ($\epsilon$)'
        plot_ylabel = r'$\epsilon$ (${m}^2/{s}^3$)'
        plot_title = 'Spatial distribution of time averaged and depth averaged turbulence dissipation rate'
    elif 'eps' in cfg.get("variable"):
        plot_label = r'Time Averaged and depth averaged $\epsilon$'
        plot_ylabel = r'$\epsilon$ (${m}^2/{s}^3$)'
        plot_title = 'Spatial distribution of time averaged and depth averaged dissipation rate'
    
    plt.vlines(6.8, min(eps_avg), max(eps_avg), color='black')
    plt.text(5.5, 1e-4, r'$X_{c}=6.8$ m', color='black')
    plt.xlabel('X (m)')
    plt.ylabel(plot_ylabel)
    plt.title(plot_title)
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
main()