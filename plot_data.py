#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

INPUT_PARAMETERS = {
    'base_directory': [
        '/Users/abhishek/work/free_surface_2025/wave_tank/exp_beach_profile/eroded_profile/no_reflection/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/exp_beach_profile/initial_profile/no_reflection/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/exp_beach_profile/bar_nourishment/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/exp_beach_profile/berm_nourishment/no_reflection/logs',
        '/Users/abhishek/work/free_surface_2025/wave_tank/exp_beach_profile/profile_nourishment/no_reflection/logs',
    ],
    'variable': 'TKE_epsilon_turb_epsilon', # or 'TKE', or 'epsilon_turb'
    'output_parameters': ['TKE_avg', 'epsilon_turb_avg', 'epsilon_avg'], # columns (besides X) to read from .dat
    't_start': 18.5,
    't_end': 28,
    'window': False,
    'window_periods': [19, 20.8, 22.6, 24.4, 25, 26.4],
    'Xmask': None, #[23, 24], # range of X to mask out (e.g. near the wall)
    'streamwise_avg': True,
    'streamwise_avg_range': [19, 24], # range of X to average over for streamwise averaging
}

cfg = dict(INPUT_PARAMETERS)

def _plot_labels(var_name):
    if 'TKE' in var_name:
        return r'k (${m}^2/{s}^2$)', 'Spatial distribution of time averaged and depth averaged turbulent kinetic energy'
    if 'eps' in var_name:
        return r'$\epsilon$ (${m}^2/{s}^3$)', 'Spatial distribution of time averaged and depth averaged turbulent dissipation rate'
    return var_name, var_name

def main():
    dirs = cfg['base_directory']
    if isinstance(dirs, str):
        dirs = [dirs]
    output_params = cfg.get('output_parameters') or []
    if not output_params:
        raise ValueError("No 'output_parameters' set in INPUT_PARAMETERS")

    for var_name in output_params:
        plt.figure(figsize=(8, 5))
        sw_results = []
        for base_dir in dirs:
            data_dir = os.path.join(base_dir, cfg["variable"])
            case_name, sw_val = epsilon_plot(data_dir, cfg, base_dir, var_name)
            if sw_val is not None:
                sw_results.append((case_name, sw_val))

        plot_ylabel, plot_title = _plot_labels(var_name)
        plt.xlabel('X (m)')
        plt.ylabel(plot_ylabel)
        plt.title(plot_title)
        if 'eps' in var_name:
            plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Streamwise-averaged bar chart for this parameter
        if cfg.get('streamwise_avg') and sw_results:
            case_names = [r[0] for r in sw_results]
            sw_values = [r[1] for r in sw_results]
            xr = cfg['streamwise_avg_range']
            plt.figure(figsize=(8, 5))
            plt.plot(case_names, sw_values, marker='o', linestyle='-')
            plt.ylabel(plot_ylabel)
            plt.title(f'Streamwise averaged ({xr[0]}m - {xr[1]}m) {var_name}')
            if 'eps' in var_name:
                plt.yscale("log")
                plt.ylim(2e-4, 8e-3)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()

def time_avg(pairs, var_name, window=None, output_params=None):

    if window is not None:
        tmin, tmax = window[0], window[1]
    else:
        tmin, tmax = pairs[0][0], pairs[-1][0]
    # Collect all data with X positions
    data_dict = {}  # dict of {x_position: [eps_values_at_different_times]}

    for ti, p in pairs:

        if ti >= tmin and ti <= tmax:
            cols = load_cols(p, output_params=output_params)
            
            eps_turb = np.array(cols[var_name])
            x = np.array(cols["X"])
            
            #print(f"max {var_name}", max(eps_turb), "at t=", ti)
            
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
    
def streamwise_average(eps_avg, x_array, x_range):
    """Average parameter values over the given streamwise X range."""
    mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
    if np.any(mask):
        return np.mean(eps_avg[mask])
    return np.nan

def cleanup(eps,x):
    eps_index = np.where(eps >= 10 ** -9)
    x_array = x[eps_index]
    eps_avg = eps[eps_index]
    mask = np.where((x_array < 4) & (eps_avg >= 10**-4))
    x_array = np.delete(x_array, mask)
    eps_avg = np.delete(eps_avg, mask)
    return eps_avg, x_array, mask
def load_cols(path, output_params=None):
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

    # First column is always X coords; remaining columns optionally filtered by output_params
    if output_params is None:
        selected = headers
    else:
        missing = [p for p in output_params if p not in headers[1:]]
        if missing:
            raise ValueError(f"output_parameters {missing} not in header {headers[1:]} of {path}")
        selected = [headers[0]] + [h for h in headers[1:] if h in output_params]

    # Create dictionary using original column index from headers
    cols = {}
    for h in selected:
        i = headers.index(h)
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
    eflux = data[:, 1]  # 2nd column
    
    
    # Plot
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, eflux, marker='o', linestyle='-', label='Energy Flux at x=25m')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Total Energy Flux (${m}^3/{s}^3$)')
    plt.title('Plot of Total Flux with Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def epsilon_plot(path, cfg, base_dir, var_name):
    folder = Path(path)
    pat = re.compile(r"_([0-9.]+)\.dat$")
    print("folder:", folder)
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

    case_name = Path(base_dir).parent.name
    all_eps = []
    final_eps = np.array([])
    final_x = np.array([])
    if cfg.get('window'):
        t_list = cfg['window_periods']
        for i in range(1, len(t_list)):
            eps_avg, x_array = time_avg(pairs, var_name, window=[t_list[0], t_list[i]], output_params=cfg.get('output_parameters'))
            eps_avg, x_array, mask = cleanup(eps_avg, x_array)
            if cfg.get('Xmask') is not None:
                xm = cfg['Xmask']
                keep = ~((x_array >= xm[0]) & (x_array <= xm[1]))
                x_array = x_array[keep]
                eps_avg = eps_avg[keep]
            if len(eps_avg) > 0:
                plt.plot(x_array, gaussian_filter1d(eps_avg, sigma=2), linestyle='-',
                         label=f'Time Average Window-t={t_list[0]}s-{t_list[i]}s')
                all_eps.extend(eps_avg)
                final_eps = eps_avg
                final_x = x_array
    else:
        t_start = cfg['t_start']
        t_end = cfg['t_end']
        eps_avg, x_array = time_avg(pairs, var_name, window=[t_start, t_end], output_params=cfg.get('output_parameters'))
        eps_avg, x_array, mask = cleanup(eps_avg, x_array)
        
        if cfg.get('Xmask') is not None:
            xm = cfg['Xmask']
            keep = ~((x_array >= xm[0]) & (x_array <= xm[1]))
            x_array = x_array[keep]
            eps_avg = eps_avg[keep]
        if len(eps_avg) > 0:
            plt.plot(x_array, gaussian_filter1d(eps_avg, sigma=2), linestyle='-',
                     label=case_name)
            all_eps.extend(eps_avg)
            final_eps = eps_avg
            final_x = x_array

    sw_val = None
    if cfg.get('streamwise_avg') and cfg.get('streamwise_avg_range') and len(final_eps) > 0:
        sw_val = streamwise_average(final_eps, final_x, cfg['streamwise_avg_range'])

    return case_name, sw_val

if __name__ == "__main__":
    main()