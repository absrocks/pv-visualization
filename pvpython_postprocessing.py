#!/usr/bin/env python3
"""
driver.py — Select input files, pass config to a child pvpython script, and render a chosen array.
Saves one PNG per timestep as: {array}_t_{<time>}.png

Run:
  python3 driver.py
(Adjust INPUT_PARAMETERS and PVPYTHON_EXE as needed.)
"""
import subprocess
import sys
import os
import json
import logging
import tempfile
import shlex
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
INPUT_PARAMETERS = {
    'pattern_type': 'glob',
    'base_directory': './',
    'file_template': '*.foam',
    'output_directory': './out',
    'number_range': None,
    'start_time': 6,        # None --> to start from 0
    'end_time': 20,

    # ---- Averaging Options ----
    'averaging': {
        'axis': 'Y',        # 'X' | 'Y' | 'Z'
    },
    'clipping': {
        'enabled': True,      # set False to disable
        'axis': 'X',          # 'X' | 'Y' | 'Z'
        'Xmin': 21.0,
        'Xmax': 34.0,
    },
    'slice': {
        'enabled': True,      # set False to disable
    },
    # ---- OpenFOAM-specific options ----
    'openfoam': {
        'mode': 'decomposed',                                        # 'reconstructed' | 'decomposed' | 'auto'
        'mesh_regions': ['internalMesh'],                            # or [] / None
        'cell_arrays':  ['U', 'alpha.water', 'nut', 'UAvg'],         # or [] / None , 'UAvg', 'nut
        'point_arrays': ['U', 'alpha.water', 'nut', 'UAvg'],         # e.g., ['T']
    },

    # ---- Visualization options ----
    'visualization': {
        'image_size': [2400, 1800],          # [width, height]
        'color_map': 'Jet',                 # colormap preset name
        'array': 'UAvg',                    # REQUIRED: array to visualize
        'out_array': 'flux',
        'range': [1e-5, 1],                  # e.g., [0.0, 5.0]; None = auto
        'custom_label': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],               # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],  # e.g. None
        'label_format': '6.1e',             # '6.1e' | '6.2f'
        'show_scalar_bar': True,            # show scalar bar
        'background': [1, 1, 1],            # white background
        'camera_plane': 'XZ',               # NEW: 'XZ' | 'XY' | 'YZ'
        'show_axis': False,
    },
    
}

# If pvpython is not on PATH, set the absolute path here:
PROCESSING_OPTIONS = {
    'paraview_executable': 'pvbatch',                  # 'pvpython' | 'pvbatch'
    'paraview_args': ['--force-offscreen-rendering'],
}

MPI = {
    "enabled": True,                   # set False to run serial
    "launcher": "mpiexec",             # "mpiexec" | "srun" | etc.
    "n": 16,                            # number of ranks
    "extra_args": []                   # e.g. ["--bind-to","core"]
}
# -------------------------------
# Child pvpython script (string)
# -------------------------------
SCRIPT_CONTENT = r'''
import sys, os, json, argparse, re
from paraview.simple import *
from paraview import servermanager as sm
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkParallelCore import vtkMultiProcessController, vtkCommunicator
import numpy as np
import builtins as _bi
import vtk
from vtk import vtkMPI4PyCommunicator
from mpi4py import MPI

def main():
    args = parse_args()
    cfg = load_cfg(args.config_file)

    fname = args.files[0]
    if not os.path.exists(fname):
        print(f"ERROR: File not found: {fname}", file=sys.stderr)
        return 3
    else:
        print(f"[pvpython-child] Loaded: {fname}")

    # Load dataset
    src = pick_reader(fname, cfg)
    pnames, cnames = list_point_cell_arrays(src)
    info = src.GetDataInformation()
    npts, ncel = info.GetNumberOfPoints(), info.GetNumberOfCells()
    print(f"[pvpython-child] Points: {npts}  Cells: {ncel}")
    
    averaging = cfg.get("averaging")
    axis_letter = averaging.get("axis")
    vis_array = cfg.get("visualization")["array"]
    effective_vis_array = vis_array
    
    if npts == 0:
        print("ERROR: No data points found", file=sys.stderr)
        return 4
    
    if cfg.get("clipping")["enabled"] is True:
        src = apply_clipping(src, cfg.get("clipping")["axis"], xmin=cfg.get("clipping")["Xmin"], xmax=cfg.get("clipping")["Xmax"])
    
    
    if cfg.get("visualization")["show_axis"] is True:
        src1 = vis_slice_axis(src, axis_letter)
        print("Visualization axis is set")
        
    (xmin,xmax,ymin,ymax,zmin,zmax) =_domain_bounds(src)
    
    # Apply IsoVolume
    #src = apply_isovolume(src, cfg)
    
    # FLATTEN first, so everything downstream sees real vtkDataArrays:
    #src = flatten_dataset(src)
    
    try:
        base = vis_array
        src, avg_name = apply_spanwise_average(src, axis_letter=axis_letter, array_name=base)
        print(f"[pvpython-child] Calculated array: {avg_name}")
        # Compute average
        src, alpha_avg = apply_spanwise_average(src, axis_letter=axis_letter, array_name="alpha.water")
        
        if 'k' in cfg.get("visualization")["out_array"]:
            
            print(f"[pvpython-child] TKE output will be written")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, k_name = calculate_k(src, prime_vec_name=prime_name, axis_letter=axis_letter, result_name="TKE")
            effective_vis_array = k_name
            print(f"[pvpython-child] Added array: {effective_vis_array}")
        
        if 'eps' in cfg.get("visualization")["out_array"]:
            print(f"[pvpython-child] Epsilon output will be written")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, grad_name = apply_gradient(src, prime_name)
            src, s2_name = strain_rate(src, array_name=grad_name, out_name="S2")
            src, eps_name = calculate_epsilon(src, s2_name, axis_letter=axis_letter, result_name='epsilon')
            effective_vis_array = eps_name
            print(f"[pvpython-child] Added array: {effective_vis_array}")
            
        if 'energy' in cfg.get("visualization")["out_array"]:
            print(f"[pvpython-child] Energy output will be written")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, ke = calculate_ke(src, avg_name, result_name="KE")
            src, k_name = calculate_k(src, prime_vec_name=prime_name, axis_letter=axis_letter, result_name="TKE")
            src, grad_name = apply_gradient(src, prime_name)
            src, s2_name = strain_rate(src, array_name=grad_name, out_name="S2")
            src, eps_name = calculate_epsilon(src, s2_name, axis_letter=axis_letter, result_name='epsilon')
            
            effective_vis_array = [k_name, eps_name, ke, None]
            print(f"[pvpython-child] Added array: {effective_vis_array}")
        if 'flux' in cfg.get("visualization")["out_array"]:
            print(f"[pvpython-child] Flux output will be written")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, tke = calculate_k(src, prime_vec_name=prime_name, axis_letter=axis_letter, result_name="TKE")
        
            src, grad_name = apply_gradient(src, prime_name)
            src, s2_name = strain_rate(src, array_name=grad_name, out_name="S2")
            src, eps_name = calculate_epsilon(src, s2_name, axis_letter=axis_letter, result_name='epsilon')
        
        src = apply_isovolume(src, cfg, array_name=alpha_avg)
        
    except Exception as e:
        print(f"[pvpython-child][ERROR] Averaging/fluctuation step failed: {e}", file=sys.stderr)
        return 6
    
    # ---- Render & save ----
    try:
        if 'energy' in cfg.get("visualization")["out_array"]:
            src = apply_clipping(src, 'Y', ymin=0, ymax=0.1)
            src = Redistribute(src)
            src = energy(src, cfg, effective_vis_array)
        if 'flux' in cfg.get("visualization")["out_array"]:
            src = apply_slices(src, "Y")
            src = apply_slices(src, "X", loc=22)
            src, flux, flux_eps = calculate_fluxes(src, avg_name, tke, eps_name, out1='Flux', out2='Flux_eps')
            effective_vis_array = [flux, flux_eps]
            print(f"[pvpython-child] Added array: {effective_vis_array}")
            src = out_flux(src, cfg, effective_vis_array)
        else:
            src = apply_slices(src, "Y")
            color_by_array_and_save_pngs(src, cfg, zmin, zmax, desired_array=effective_vis_array)
        
    except Exception as e:
        print(f"[pvpython-child][ERROR] Visualization failed: {e}", file=sys.stderr)
        return 7

    print("[pvpython-child] Completed successfully.")
    return 0

def out_flux(src, cfg, effective_vis_array):
    flux_dat = cfg.get("output_directory") + "/" + "eflux.dat"
    print("The Flux data will be saved at: ",flux_dat)
    
    # write header once
    dat_init(flux_dat, effective_vis_array)
    
    tk = GetTimeKeeper()
    times = list(getattr(tk, "TimestepValues", []) or [])
    if not times:
        times = list(getattr(src, "TimestepValues", []) or [])
    
    # Optional window
    tmin = cfg.get("start_time", None)
    tmax = cfg.get("end_time", None)
    print("tmin",tmin, "tmax",tmax)
    
    for t in times:
        if (tmin is not None and t < tmin) or (tmax is not None and t > tmax):
            continue
        GetAnimationScene().AnimationTime = t
        try:
            src.UpdatePipeline(time=t)
        except Exception:
            src.UpdatePipeline()
        integ = integrate_variables(src)
        res, measure, missing = fetch_integrals(integ, effective_vis_array, components=None, return_average=True)
        if missing:
            print(f"[warn] t={t}: missing in integrator output: {missing}", flush=True)

        dat_append(flux_dat, t, effective_vis_array, res)
        print("res", res)
        print("Measure =", measure, " averages:",
          " ".join(f"{k}={res.get(k,{}).get('average', float('nan')):.6g}" for k in effective_vis_array),
          flush=True)
    return src

def calculate_magnitude(src, array_name):
    """
    If `array_name` has >1 components, create <name>_Magnitude via Calculator.
    Returns (proxy_with_mag, magnitude_name). If scalar, returns (src, None).
    """
    pnames, cnames = list_point_cell_arrays(src)
    if array_name in pnames:
        assoc = 'POINTS'
    elif array_name in cnames:
        assoc = 'CELLS'
    else:
        raise RuntimeError(
            f"calculate_magnitude: '{array_name}' not found on input. "
            f"POINTS={pnames}; CELLS={cnames}"
        )

    q = _quote_if_needed(array_name)
    calc = Calculator(Input=src)
    calc.AttributeType   = ('Point Data' if assoc == 'POINTS' else 'Cell Data')
    calc.ResultArrayName = f"{array_name}_Magnitude"
    calc.Function        = f"mag({q})"
    calc.UpdatePipeline()
    return calc, calc.ResultArrayName
    
def energy(src, cfg, effective_vis_array):
    """
    For each timestep in the source, compute spanwise-average of `base_array`,
    then print bounds at that time. Returns the averaging filter so caller can reuse.
    """
    out_dat = "logs/surface_averages_no_Z0_pot.dat"
    # write header once
    dat_init(out_dat, effective_vis_array)
    
    # 0) Gather times
    tk = GetTimeKeeper()
    times = list(getattr(tk, "TimestepValues", []) or [])
    if not times:
        times = list(getattr(src, "TimestepValues", []) or [])
    
    # Optional window
    tmin = cfg.get("start_time", None)
    tmax = cfg.get("end_time", None)
    print("tmin",tmin, "tmax",tmax)
    
    for t in times:
        
        if (tmin is not None and t < tmin) or (tmax is not None and t > tmax):
            continue
        GetAnimationScene().AnimationTime = t
        try:
            src.UpdatePipeline(time=t)
        except Exception:
            src.UpdatePipeline()
        
        # Query bounds on the averaged output (geometry is unchanged by averaging)
        (xmin,xmax,ymin,ymax,zmin,zmax) =_domain_bounds(src)
        print(f"src bounds : [{xmin},{xmax},{ymin},{ymax},{zmin},{zmax}]")
        
        pf, bfield = global_max_and_bounds_pf(src, cfg)
        gbounds = read_global_stats(pf, bfield, time=t)
        zz_max,xz_max = gbounds
        
        print(f"[pvpython-child] Maximum wave height at t={t}: "
              f" is {zz_max} at x={xz_max}",
              flush=True)
        src_y = apply_slices(src, "Y")
        #src_z = apply_clipping(src_y, 'Z', zmin=0, zmax=zmax+1)
        src_x = apply_clipping(src_y, 'X', xmin=xz_max-0.5, xmax=xz_max+0.5)
        src_pe, pe = calculate_pe(src_x, result_name="PE")
        effective_vis_array[-1] = pe
        integ = integrate_variables(src_pe)
        
        res, measure, missing = fetch_integrals(integ, effective_vis_array, return_average=True)
        if missing:
            print(f"[warn] t={t}: missing in integrator output: {missing}", flush=True)

        dat_append(out_dat, t, effective_vis_array, res)
        
        print("Measure =", measure, " averages:",
          " ".join(f"{k}={res.get(k,{}).get('average', float('nan')):.6g}" for k in effective_vis_array),
          flush=True)
        
        src = apply_clipping(src, cfg.get("clipping")["axis"], xmin=np.floor(xz_max) - 0.5, xmax=cfg.get("clipping")["Xmax"])
        src = Redistribute(src)
    return src

def Redistribute(src):
    d = RedistributeDataSet(Input=src)
    d.UpdatePipeline()
    src = d
    
    return src
    
def integrate_variables(src):
    integ = IntegrateVariables(Input=src)
    # NOTE: By default, IntegrateVariables returns pure integrals.
    integ.UpdatePipeline()
    
    return integ

def fetch_integrals(src, arrays, components=None, return_average=False):
    """
    Read integrated values for `arrays` from an IntegrateVariables *proxy* `src`.
    For multi-component arrays with no component specified, FIRST look for a
    magnitude column in the integrator output:
        '<name>_Magnitude', '<name>_input_1', '<name>_mag', '|<name>|', 'mag(<name>)'.
    If none exists, auto-create it via `calculate_magnitude` (post-integrate) and use that.
    """
    dobj = sm.Fetch(src)
    if dobj is None:
        raise RuntimeError("IntegrateVariables: Fetch returned None.")

    wrap = dsa.WrapDataObject(dobj)
    cd   = wrap.CellData
    pd   = wrap.PointData

    # Geometric measure (Volume/Area/Length) from CellData if present
    measure = float("nan")
    for key in ("Volume", "Area", "Length"):
        if key in cd.keys():
            m = np.asarray(cd[key], dtype=float).ravel()
            if m.size:
                measure = float(m[0]); break

    results    = {}
    missing    = []
    components = components or {}

    def _find_mag_column(base):
        for nm in (f"{base}_Magnitude", f"{base}_input_1", f"{base}_mag", f"|{base}|", f"mag({base})"):
            if nm in cd.keys():
                arr = np.asarray(cd[nm], dtype=float)
                return (arr[0] if arr.ndim == 2 and arr.shape[0] == 1 else arr), nm
            if nm in pd.keys():
                arr = np.asarray(pd[nm], dtype=float)
                return (arr[0] if arr.ndim == 2 and arr.shape[0] == 1 else arr), nm
        return None, None

    for name in arrays:
        # Pull from integrator output (prefer CellData)
        if name in cd.keys():
            raw = np.asarray(cd[name], dtype=float)
        elif name in pd.keys():
            raw = np.asarray(pd[name], dtype=float)
        else:
            missing.append(name)
            continue

        vals = raw[0] if (raw.ndim == 2 and raw.shape[0] == 1) else raw
        comp_sel = components.get(name, None)

        if np.isscalar(vals):
            val = float(vals)

        elif vals.ndim == 1:  # multi-component vector/tensor
            if comp_sel is None:
                alt, alt_nm = _find_mag_column(name)
                if alt is None:
                    # Create <name>_Magnitude on-the-fly (post-integrate), then re-fetch
                    src, mag_nm = calculate_magnitude(src, name)
                    dobj = sm.Fetch(src)
                    if dobj is None:
                        raise RuntimeError("IntegrateVariables (after mag creation): Fetch returned None.")
                    wrap = dsa.WrapDataObject(dobj)
                    cd, pd = wrap.CellData, wrap.PointData
                    # Try again
                    if mag_nm in cd.keys():
                        alt = np.asarray(cd[mag_nm], dtype=float)
                    elif mag_nm in pd.keys():
                        alt = np.asarray(pd[mag_nm], dtype=float)
                    else:
                        raise RuntimeError(f"IntegrateVariables: failed to create '{mag_nm}'.")
                val = float(np.asarray(alt).ravel()[0])
            elif isinstance(comp_sel, str) and comp_sel.lower() == "magnitude":
                alt, _ = _find_mag_column(name)
                if alt is None:
                    # Same as above: create then read
                    src, _ = calculate_magnitude(src, name)
                    dobj = sm.Fetch(src)
                    if dobj is None:
                        raise RuntimeError("IntegrateVariables (after mag creation): Fetch returned None.")
                    wrap = dsa.WrapDataObject(dobj)
                    cd, pd = wrap.CellData, wrap.PointData
                    alt, _ = _find_mag_column(name)
                    if alt is None:
                        raise RuntimeError(
                            f"IntegrateVariables: requested magnitude for '{name}', "
                            f"but could not create/find magnitude column."
                        )
                val = float(np.asarray(alt).ravel()[0])
            else:
                idx = int(comp_sel)
                if idx < 0 or idx >= vals.shape[0]:
                    raise RuntimeError(f"components['{name}'] index {idx} out of range 0..{vals.shape[0]-1}")
                val = float(vals[idx])

        else:
            raise RuntimeError(f"IntegrateVariables: '{name}' returned shape {vals.shape}, not supported.")

        entry = {'integral': val}
        if return_average and np.isfinite(measure) and measure != 0.0:
            entry['average'] = val / measure
        results[name] = entry

    return results, measure, missing


def global_max_and_bounds_pf(src, cfg):


    """
    Build a ProgrammableFilter that computes GLOBAL max of `array_name` and GLOBAL
    bounds, storing results in FieldData arrays:
      - "global_max__<array_name>"  (double, 1-tuple)
      - "global_bounds"             (double, 1-tuple, 6 components)
    Returns: (pf_proxy, max_field_name, bounds_field_name)
    """
    assoc = "POINTS"
    PF = r"""
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkParallelCore import vtkMultiProcessController, vtkCommunicator
from vtkmodules.vtkCommonCore import vtkDoubleArray
import numpy as np, math
from vtk import vtkMPI4PyCommunicator
from mpi4py import MPI

# --- MPI ---
ctrl = vtkMultiProcessController.GetGlobalController()
comm = vtkMPI4PyCommunicator.ConvertToPython(ctrl.GetCommunicator())
rank = comm.Get_rank()

inp = self.GetInputDataObject(0, 0)

out = self.GetOutputDataObject(0)
out.ShallowCopy(inp)

wrap = dsa.WrapDataObject(inp)
data = wrap.PointData if "__ASSOC__" == "POINTS" else wrap.CellData

pts = wrap.Points
xyz = np.asarray(pts, dtype=float)
z = xyz[:,2]
x = xyz[:,0]

zz_max, xz_max = 0, 0

i_local   = int(np.argmax(z))

z_local   = float(z[i_local])
x_at_local= float(x[i_local])

pair_local  = np.array([z_local, x_at_local])

pairs_global= comm.allgather(pair_local)

idx = int(np.argmax([p[0] for p in pairs_global]))
zz_max = float(pairs_global[idx][0])
xz_max = float(pairs_global[idx][1])

# --- write FieldData using explicit vtkDoubleArray ---
fd = out.GetFieldData()

# global max

# global bounds as 1-tuple, 6 components
abds = vtkDoubleArray()
abds.SetName("global_bounds")
abds.SetNumberOfComponents(2)
abds.SetNumberOfTuples(1)
abds.SetTuple(0, (zz_max, xz_max))
fd.RemoveArray(abds.GetName())
fd.AddArray(abds)
""".lstrip()

    code = (
        PF.replace("__ASSOC__", assoc)
    )

    pf = ProgrammableFilter(Input=src)
    pf.Script = code
    pf.RequestInformationScript = ''
    pf.RequestUpdateExtentScript = ''
    pf.PythonPath = ''
    pf.UpdatePipeline()

    return pf, "global_bounds"

def calculate_pe(src, result_name='PE', g=9.81):
    (xmin,xmax,ymin,ymax,zmin,zmax) =_domain_bounds(src)
    expr =  f"{float(g)} * (coordsZ - {float(zmin)})"
    calc_pe = Calculator(Input=src)
    calc_pe.ResultArrayName = result_name
    calc_pe.Function = expr
    calc_pe.UpdatePipeline()

    # Done
    return calc_pe, result_name
    
def _quote_if_needed(name: str) -> str:
    return name if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) else f'"{name}"'


def calculate_fluxes(src, vec_name, tke_name, eps, out1='flux', out2='flux_eps', g=9.81):
    # sanity: both arrays must live in same association
    pnames, cnames = list_point_cell_arrays(src)
    if (vec_name in pnames) and (tke_name in pnames):
        assoc = 'POINTS'
    elif (vec_name in cnames) and (tke_name in cnames):
        assoc = 'CELLS'
    else:
        raise RuntimeError(
            f"calculate_fluxes: '{vec_name}' and '{tke_name}' must both be in POINTS or both in CELLS. "
            f"POINTS={pnames}; CELLS={cnames}"
        )

    # stay on original assoc (no cell→point conversion)
    q = _quote_if_needed(vec_name)
    tkeq = _quote_if_needed(tke_name)
    epsq = _quote_if_needed(eps)

    xmin,xmax,ymin,ymax,zmin,zmax = _domain_bounds(src)  # your helper
    # vector flux pieces: scalar * vector
    kinetic   = f"0.5*dot({q},{q})*{q}"
    potential = f"{float(g)}*(coordsZ - {float(zmin)})*{q}"
    tke_flux  = f"{tkeq}*{q}"
    eps_flux = f"{epsq}*{q}"

    # Calculator 1: mechanical flux (kinetic + potential)
    calc1 = Calculator(Input=src)
    calc1.AttributeType   = ('Point Data' if assoc=='POINTS' else 'Cell Data')
    calc1.ResultArrayName = out1
    calc1.Function        = f"({kinetic}) + ({potential}) + ({tke_flux})"
    calc1.UpdatePipeline()

    # Calculator 2: turbulent kinetic energy flux
    calc2 = Calculator(Input=src)
    calc2.AttributeType   = ('Point Data' if assoc=='POINTS' else 'Cell Data')
    calc2.ResultArrayName = out2
    calc2.Function        = eps_flux
    calc2.UpdatePipeline()

    # Combine both outputs onto one proxy
    merged = AppendAttributes(Input=[calc1, calc2])
    merged.UpdatePipeline()
    return merged, out1, out2


def calculate_ke(src, vec_name, result_name='KE'):

    pnames, cnames = list_point_cell_arrays(src)
    if vec_name in pnames:
        assoc = 'POINTS'
    elif vec_name in cnames:
        assoc = 'CELLS'
    else:
        raise RuntimeError(
            f"calculate_energy: vector '{vec_name}' not found. "
            f"Point arrays: {pnames}; Cell arrays: {cnames}"
        )

    src_pts = ensure_points_for_array(src, vec_name)
    q = _quote_if_needed(vec_name)
    
    kinetic = f"0.5*dot({q},{q})"
    expr = f"{kinetic}"
    
    calc_ke = Calculator(Input=src_pts)
    calc_ke.ResultArrayName = result_name
    calc_ke.Function = expr
    calc_ke.UpdatePipeline()

    # Done
    return calc_ke, result_name
    
def _ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)

def dat_init(path: str, arrays: list, suffix="_avg"):
    """
    Create/overwrite a .dat file with a header:
    # Time <A1_suffix> <A2_suffix> ...
    """
    _ensure_parent(path)
    with open(path, "w") as f:
        cols = ["Time"] + [f"{name}{suffix}" for name in arrays]
        f.write("# " + " ".join(cols) + "\n")

def dat_append(path: str, t: float, arrays: list, results: dict, suffix="_avg"):
    """
    Append one row for time t.
    `results` is the dict returned by fetch_integrals(...)[0]
    (i.e., {name: {'integral': ..., 'average': ...}, ...})
    Missing or absent averages are written as NaN.
    """
    vals = []
    for name in arrays:
        avg = results.get(name, {}).get("average", float("nan"))
        # ensure float for formatting
        try:
            v = float(avg)
        except Exception:
            v = float("nan")
        vals.append(v)

    line = "{: .9g} ".format(t) + " ".join("{: .9g}".format(v) for v in vals) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())  # make it visible during long SLURM runs
        

def read_global_stats(pf_proxy, bounds_field_name, time=None):
    """
    Fetch `pf_proxy` (optionally at `time`) and read:
      - global max from FieldData[max_field_name]
      - global bounds from FieldData[bounds_field_name]
    Returns: (gmax: float|None, bounds: (6-tuple)|None)
    """
    from paraview.simple import MergeBlocks
    from paraview import servermanager as sm

    mb = MergeBlocks(Input=pf_proxy)
    try:
        mb.UpdatePipeline(time=time)
    except Exception:
        mb.UpdatePipeline()

    dobj = sm.Fetch(mb)
    if dobj is None:
        return None, None

    fd = dobj.GetFieldData()
    abds = fd.GetArray(bounds_field_name)
    bounds = None

    if abds is not None:
        ncomp = abds.GetNumberOfComponents()
        ntup  = abds.GetNumberOfTuples()

        # Expect at least one tuple
        if ntup >= 1 and ncomp > 0:
            bounds = tuple(float(abds.GetComponent(0, i)) for i in range(ncomp))
        else:
            bounds = None
    

    return bounds

def apply_slices(src, axis_letter, loc=None):

    (xmin,xmax,ymin,ymax,zmin,zmax) = _domain_bounds(src)
    pos   = [(xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2]
    
    # Apply Slice
    slice1 = Slice(registrationName='Slice1', Input=src)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    
    if axis_letter == 'Y':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[1] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [0.0, 1.0, 0.0]
     
    if axis_letter == 'X':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[0] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [1.0, 0.0, 0.0]
    
    if axis_letter == 'Z':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[2] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]
    
    src = slice1
    src.UpdatePipeline()
    
    return src
    
def resolve_derived_request(name, default_axis='Y'):
    """
    Accept 'U_avg', 'U_avg_Y', 'U_prime', 'U_prime_Z'.
    Returns (base, kind, axis) or None if not derived.
    """
    if not isinstance(name, str):
        return None
    m = re.match(r'^(?P<base>.+)_(?P<kind>avg|prime)(?:_(?P<axis>[XYZ]))?$', name, re.IGNORECASE)
    if not m:
        return None
    base = m.group('base')
    kind = m.group('kind').lower()
    axis = (m.group('axis') or default_axis).upper()
    return base, kind, axis
    
def parse_args():
    ap = argparse.ArgumentParser(description="Child pvpython runner")
    ap.add_argument("--config-file", required=True, help="Path to JSON config from driver")
    ap.add_argument("files", nargs="+", help="Input dataset files")
    return ap.parse_args()

def load_cfg(path):
    with open(path, "r") as f:
        return json.load(f)
def flatten_dataset(src):
    """
    Merge composite inputs (MultiBlock/Partitioned) into a single dataset
    so numpy_interface sees regular vtkDataArrays (with .shape).
    Safe to use even if input isn't composite.
    """
    mb = MergeBlocks(Input=src)
    # mb.MergePoints = 0  # optional: keep as-is; set to 1 to merge coincident points
    mb.UpdatePipeline()
    return mb

def list_point_cell_arrays_flat(src):
    """
    List arrays on a flattened dataset (post-MergeBlocks).
    """
    flat = flatten_dataset(src)
    info = flat.GetDataInformation()
    pdi = info.GetPointDataInformation()
    cdi = info.GetCellDataInformation()
    point_names, cell_names = [], []
    if pdi:
        for i in range(pdi.GetNumberOfArrays()):
            point_names.append(pdi.GetArrayInformation(i).GetName())
    if cdi:
        for i in range(cdi.GetNumberOfArrays()):
            cell_names.append(cdi.GetArrayInformation(i).GetName())
    return point_names, cell_names

def ensure_points_for_array(src, array_name):
    """
    Flatten first; if array is in CellData, convert to PointData for averaging.
    Returns a non-composite dataset ready for numpy ops.
    """
    flat = flatten_dataset(src)
    pnames, cnames = list_point_cell_arrays_flat(flat)
    if array_name in pnames:
        return flat
    if array_name in cnames:
        c2p = CellDatatoPointData(Input=flat)
        c2p.ProcessAllArrays = 1
        c2p.UpdatePipeline()
        return c2p
    # Not present; return flattened anyway (caller will error clearly)
    return flat

def list_point_cell_arrays(src):
    #info = src.GetDataInformation()
    #pdi = info.GetPointDataInformation()
    #cdi = info.GetCellDataInformation()
    #point_names, cell_names = [], []
    #if pdi:
    #    for i in range(pdi.GetNumberOfArrays()):
    #        point_names.append(pdi.GetArrayInformation(i).GetName())
    #if cdi:
    #    for i in range(cdi.GetNumberOfArrays()):
    #        cell_names.append(cdi.GetArrayInformation(i).GetName())
    return list_point_cell_arrays_flat(src)

def get_array_components(src, assoc, name):
    "Return number of components for (assoc, name). assoc in {'POINTS','CELLS'}"
    info = src.GetDataInformation()
    if assoc == "POINTS":
        pdi = info.GetPointDataInformation()
        for i in range(pdi.GetNumberOfArrays()):
            ai = pdi.GetArrayInformation(i)
            if ai.GetName() == name:
                return ai.GetNumberOfComponents()
    else:
        cdi = info.GetCellDataInformation()
        for i in range(cdi.GetNumberOfArrays()):
            ai = cdi.GetArrayInformation(i)
            if ai.GetName() == name:
                return ai.GetNumberOfComponents()
    return None

def openfoam_reader(fname, of_cfg):
    mode          = (of_cfg.get("mode") or "reconstructed").lower()
    mesh_regions  = of_cfg.get("mesh_regions")
    cell_arrays   = of_cfg.get("cell_arrays")
    point_arrays  = of_cfg.get("point_arrays")

    if mode == "auto":
        case_dir = os.path.dirname(os.path.abspath(fname)) or "."
        try:
            entries = os.listdir(case_dir)
        except Exception:
            entries = []
        has_proc = any(e.startswith("processor") and os.path.isdir(os.path.join(case_dir, e)) for e in entries)
        mode = "decomposed" if has_proc else "reconstructed"

    rdr = OpenFOAMReader(FileName=fname)
    rdr.CaseType = "Decomposed Case" if mode == "decomposed" else "Reconstructed Case"

    if mesh_regions:
        rdr.MeshRegions = mesh_regions
    if cell_arrays is not None:
        rdr.CellArrays = cell_arrays
    if point_arrays is not None:
        rdr.PointArrays = point_arrays

    rdr.UpdatePipeline()
    return rdr

def pick_reader(fname, cfg):
    low = fname.lower()
    if low.endswith(".foam"):
        return openfoam_reader(fname, cfg.get("openfoam", {}))
    if low.endswith(".vtm"):
        return XMLMultiBlockDataReader(FileName=[fname])
        
def _axis_index(axis_letter):
    return {'X': 0, 'Y': 1, 'Z': 2}[axis_letter.upper()]

def _domain_bounds(src):
    """Return (xmin,xmax,ymin,ymax,zmin,zmax) from the current source."""
    info = src.GetDataInformation()
    b = info.GetBounds()
    if b is None:
        raise RuntimeError("Cannot get dataset bounds for clipping.")
    return b  # (xmin,xmax, ymin,ymax, zmin,zmax)

def vis_slice_axis(src, axis_letter, loc=None):

    # create a new 'Extract Surface'
    cur = MergeBlocks(Input=src)
    #src.UpdatePipeline()
    (xmin,xmax,ymin,ymax,zmin,zmax) = _domain_bounds(src)
    pos   = [(xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2]
    
    # Apply Slice
    slice1 = Slice(registrationName='Slice1', Input=cur)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    
    if axis_letter == 'Y':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[1] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [0.0, 1.0, 0.0]
     
    if axis_letter == 'X':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[0] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [1.0, 0.0, 0.0]
    
    if axis_letter == 'Z':
        # init the 'Plane' selected for 'SliceType'
        if loc is not None:
            pos[2] = loc
        slice1.SliceType.Origin = pos
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]
    
    slice1.UpdatePipeline()
    sliceShow = Show(slice1)
    sliceShow.Representation = 'Outline'
    
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=slice1)

    # create a new 'Redistribute DataSet'
    redistributeDataSet1 = RedistributeDataSet(registrationName='RedistributeDataSet1', Input=extractSurface1)
    redistributeDataSet1.NumberOfPartitions = 0
    redistributeDataSet1.GenerateGlobalCellIds = 1
    
    redistributeDataSet1.UpdatePipeline()
    DataShow = Show(redistributeDataSet1)
    DataShow.Representation = 'Feature Edges'
    
    # create a new 'Annotate Time Filter'
    annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=redistributeDataSet1)
    annotateTimeFilter1.Format = 'Time: {time:.2f}s'
    
    annotateTimeFilter1Display = Show(annotateTimeFilter1)

    # trace defaults for the display properties.
    try:
        annotateTimeFilter1Display.Set(
            WindowLocation='Upper Center',
            FontSize=24,
        )
    except Exception:
        try:
            annotateTimeFilter1Display.WindowLocation = 'Upper Center'
            annotateTimeFilter1Display.FontSize = 24
        except Exception:
            raise RuntimeError("Couldn't set the Time Filter Option")
            
    
    return src
    
def apply_clipping(src, axis, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
    """
    If cfg['clipping'] is enabled, apply a Box clip:
      - along the requested axis, use [min, max] from config
      - along the other axes, span the whole domain
    Returns the clipped source (Clip filter output).
    """
    
    if axis not in ('X','Y','Z'):
        raise RuntimeError(f"Invalid clipping axis: {axis}")
    

    # Use domain bounds to span the other two axes
    (dxmin,dxmax,dymin,dymax,dzmin,dzmax) = _domain_bounds(src)
    
    if axis == "X":
        if xmin is None or xmax is None:
            raise RuntimeError("xmin and xmax are not defined")
        elif not (xmax > xmin):
            raise RuntimeError(f"Clipping {axis} range must have max > min (got {xmin}, {xmax}).")
        else:
            pos   = [xmin, dymin-1, dzmin-1]
            leng  = [xmax - xmin, dymax - dymin + 5, dzmax - dzmin + 5]
            amin,amax = xmin, xmax
            
    elif axis == "Y":
        if ymin is None or ymax is None:
            raise RuntimeError("ymin and ymax are not defined")
        elif not (ymax > ymin):
            raise RuntimeError(f"Clipping {axis} range must have max > min (got {ymin}, {ymax}).")
        else:
            pos   = [dxmin-1, ymin, dzmin-1]
            leng  = [dxmax - dxmin + 5, ymax - ymin, dzmax - dzmin + 5]
            amin,amax = ymin, ymax
            
    elif axis == "Z":
        if zmin is None or zmax is None:
            raise RuntimeError("zmin and zmax are not defined")
        elif not (zmax > zmin):
            raise RuntimeError(f"Clipping {axis} range must have max > min (got {zmin}, {zmax}).")
        else:
            pos   = [dxmin-1, dymin-1, zmin]
            leng  = [dxmax - dxmin + 5, dymax - dymin + 5, zmax - zmin]
            amin, amax = zmin, zmax
        

    # Build a Box clip
    clip1 = Clip(Input=src)
    clip1.ClipType = 'Box'
    # If your ParaView build exposes HyperTreeGridClipper/Scalars/Value, leave them untouched;
    # we just use the Box to cut a spatial slab.
    clip1.ClipType.Position = pos
    clip1.ClipType.Length   = leng
    clip1.UpdatePipeline()

    print(f"[pvpython-child] Applied Box clip on {axis} from {amin} to {amax}")
    return clip1

def set_camera_plane(view, src, cfg, zmin, zmax, plane="XZ", dist_factor=1.5):
    """
    Orient camera to show a principal plane.
    'XZ' -> look along +Y, Z is up (XZ plane visible)
    'XY' -> look along +Z, Y is up
    'YZ' -> look along +X, Z is up
    """
    info = src.GetDataInformation()
    b = info.GetBounds()  # (xmin,xmax, ymin,ymax, zmin,zmax)
    if not b:
        return
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    cz = 0.5 * (b[4] + b[5])
    rx = (b[1] - b[0])
    ry = (b[3] - b[2])
    rz = (b[5] - b[4])
    R = dist_factor * _bi.max(rx, ry, rz, 1e-6)
    
    xx0 = cfg.get("clipping")["Xmin"]
    xx1 = cfg.get("clipping")["Xmax"]
    xlim = np.arange(xx0, xx1+1)
    zlim = np.linspace(zmin, zmax, 3)
    
    # For view axes:
    view.AxesGrid.XTitle = 'X (m)'
    view.AxesGrid.YTitle = 'Y (m)'
    view.AxesGrid.ZTitle = 'Z (m)  '
    
    plane = (plane or "XZ").upper()
    if plane == "XY":
        # look along +Z
        view.CameraPosition = [cx, cy, cz + R]
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraViewUp = [0, 1, 0]
    elif plane == "YZ":
        # look along +X
        view.CameraPosition = [cx + R, cy, cz]
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraViewUp = [0, 0, 1]
    elif plane == "XZ":
        # XZ (default) -> look along +Y
        view.CameraPosition = [cx, -cy - R, cz]
        view.CameraFocalPoint = [cx, cy, cz]
        view.CenterOfRotation = [cx, cy, cz]
        view.CameraViewUp = [0, 0, 1]
        view.CameraFocalDisk = 1.0
        view.CameraParallelProjection = 1
        if cfg.get("visualization")["show_axis"] is True:
            # Set Axis
            view.AxesGrid.Visibility = 1
            view.AxesGrid.AxesToLabel = 5
            
            # For data axes:
            view.AxesGrid.XAxisUseCustomLabels = 1
            view.AxesGrid.XAxisLabels = xlim.tolist()
            
            view.AxesGrid.ZAxisUseCustomLabels = 1
            view.AxesGrid.ZAxisLabels = [np.round(zmin,2), 0 , zmax]

    try:
        view.ResetCamera(False)  # keep our orientation, just fit
    except Exception:
        pass
    

def _apply_preset_safe(lut, preset, view, vis):
    #print("vis",vis)
    tried = [preset, preset.title(), preset.upper(), preset.capitalize()]
    for name in tried:
        try:
            lut.ApplyPreset(name, True)
            break
        except Exception:
            pass
    try:
        ApplyPreset(lut, preset, True)
    except Exception:
        pass
    
    if view is not None:
        try:
            sb = GetScalarBar(lut, view)
            if sb is not None:
                sb.AutomaticLabelFormat = 0
                if vis.get("custom_label") is not None:
                    sb.UseCustomLabels = 1
                    sb.CustomLabels=vis.get("custom_label")
                    pass
                sb.LabelFormat = '%-#'+vis.get("label_format") #'%-#6.1e'
                sb.RangeLabelFormat = '%-#'+vis.get("label_format") #'%-#6.1e'
                if "eps" in vis.get("out_array"):
                    sb.Title='$\\epsilon$'
                    
        except Exception:
            # Ignore if scalar bar isn't available/visible yet
            pass
    return True

def find_array_assoc(src, name):
    """Return ('POINTS'|'CELLS', ncomp) for the first match of array `name`."""
    info = src.GetDataInformation()

    pdi = info.GetPointDataInformation()
    for i in range(pdi.GetNumberOfArrays()):
        ai = pdi.GetArrayInformation(i)
        if ai.GetName() == name:
            return 'POINTS', ai.GetNumberOfComponents()

    cdi = info.GetCellDataInformation()
    for i in range(cdi.GetNumberOfArrays()):
        ai = cdi.GetArrayInformation(i)
        if ai.GetName() == name:
            return 'CELLS', ai.GetNumberOfComponents()

    return None, None  # not found


def print_array_components(src, name, label=None):
    """Print association and #components of array `name` on `src`."""
    assoc, ncomp = find_array_assoc(src, name)
    if assoc is None:
        print(f"[debug] Array '{name}' not found on source.")
    else:
        tag = f" ({label})" if label else ""
        print(f"[debug] Array '{name}'{tag}: assoc={assoc}, components={ncomp}")

def _safe_time_str(t):
    s = str(t)
    return s.replace(" ", "_").replace(":", "_").replace("/", "_").replace("\\", "_")

def apply_isovolume(src, cfg, array_name=None, threshold_range=None):
    """
    Always apply IsoVolume using an available scalar (default: 'alpha.water').
    - Validates presence in PointData or CellData (as loaded by the reader).
    - Sets InputScalars association accordingly.
    - Uses a default ThresholdRange if none provided.
    Returns: IsoVolume output.
    """
    # Defaults (edit here if you want different behavior)
    field = array_name if isinstance(array_name, str) and array_name else 'alpha.water'
    rng = threshold_range if (isinstance(threshold_range, (list, tuple)) and len(threshold_range) == 2) else [0.5, 2.0]

    # Check availability on current source (no need to flatten just to list names)
    pnames, cnames = list_point_cell_arrays(src)
    if field in pnames:
        assoc = 'POINTS'
    elif field in cnames:
        assoc = 'CELLS'
    else:
        raise RuntimeError(
            f"IsoVolume field '{field}' not found on input. "
            f"Make sure it is included in 'openfoam.point_arrays' or 'openfoam.cell_arrays'. "
            f"Point arrays: {pnames}; Cell arrays: {cnames}"
        )

    r0, r1 = float(rng[0]), float(rng[1])
    if not (r1 >= r0):
        raise RuntimeError(f"IsoVolume range must have max >= min (got {rng}).")

    iso = IsoVolume(Input=src)
    iso.InputScalars = [assoc, field]
    iso.ThresholdRange = [r0, r1]
    iso.UpdatePipeline()

    print(f"[pvpython-child] Applied IsoVolume on {assoc}:{field} in [{r0}, {r1}]")
    return iso


def apply_spanwise_average(src, axis_letter='Y', array_name='U'):
    """
    Generic spanwise averaging for scalar or vector arrays.
    Adds ONE array to PointData: {array_name}_avg_<AXIS>
    Returns: (new_source, avg_name)
    """
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    A = axis_map.get(axis_letter.upper(), 1)

    # Work on a *flattened* dataset; convert to points if needed
    src_pts = ensure_points_for_array(src, array_name)

    # Use a template with custom tokens; then do .replace() so we don't collide with
    # either % or {} formatters inside the inner script.
    PF_TEMPLATE = """
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support as ns
import numpy as np
import builtins as _bi  # <-- add this

def _robust_tolerance(arr):
    
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size < 3:
        return 1e-9
    v = np.unique(np.sort(arr))
    if v.size < 3:
        return 1e-9
    d = np.diff(v)
    h = np.percentile(d, 10)
    if not np.isfinite(h) or h <= 0:
        med = np.median(d)
        h = med if (np.isfinite(med) and med > 0) else 1e-9
    # use Python's built-in max, NOT vtk algos.max
    return _bi.max(h * 0.5, 1e-9)

inp = self.GetInputDataObject(0, 0)
if inp is None:
    raise RuntimeError("No input dataset.")

wrap = dsa.WrapDataObject(inp)
pts = wrap.Points
if pts is None or pts.shape[0] == 0:
    raise RuntimeError("No points on input.")

pd = wrap.PointData
name = "__ARRAY__"
if name not in pd.keys():
    raise RuntimeError("Array '%s' not found in PointData." % name)

data = np.asarray(pd[name])

if data.ndim == 1:
    data = data.reshape(-1, 1)
elif data.ndim == 2:
    pass
else:
    raise RuntimeError("Unsupported array shape for '%s': %s" % (name, data.shape))

A = __AXIS_INDEX__
others = [i for i in (0,1,2) if i != A]
x0 = np.asarray(pts[:, others[0]])
x1 = np.asarray(pts[:, others[1]])

t0 = _robust_tolerance(x0)
t1 = _robust_tolerance(x1)
k0 = np.round(x0 / t0).astype(np.int64)
k1 = np.round(x1 / t1).astype(np.int64)

keys = (k0.astype(np.int64) << 21) ^ (k1.astype(np.int64) & ((1<<21)-1))
uniq_keys, inv = np.unique(keys, return_inverse=True)

# Vectorized group-wise mean using bincount
G = uniq_keys.size
if data.ndim == 1:
    data = data.reshape(-1, 1)
C = data.shape[1]

counts = np.bincount(inv, minlength=G).astype(float)
avg = np.empty_like(data, dtype=float)
for c in range(C):
    sums_c = np.bincount(inv, weights=data[:, c], minlength=G)
    avg_by_group = sums_c / counts
    avg[:, c] = avg_by_group[inv]

out = self.GetOutputDataObject(0)
out.ShallowCopy(inp)
avg_vtk = ns.numpy_to_vtk(avg.copy(), deep=1)
avg_vtk.SetName("__ARRAY___avg___AXIS__")
out.GetPointData().AddArray(avg_vtk)
""".lstrip()

    pf_code = (
        PF_TEMPLATE
        .replace("__ARRAY__", array_name)
        .replace("__AXIS_INDEX__", str(A))
        .replace("__AXIS__", axis_letter.upper())
    )

    pf = ProgrammableFilter(Input=src_pts)
    pf.Script = pf_code
    pf.RequestInformationScript = ''
    pf.RequestUpdateExtentScript = ''
    pf.PythonPath = ''
    pf.UpdatePipeline()

    avg_name = f"{array_name}_avg_{axis_letter.upper()}"
    return pf, avg_name

def strain_rate(src, array_name, out_name=None):
    """
    From a vector gradient array (9 comps; 4 for 2D) named `array_name`,
    compute S2 = sum_ij S_ij^2 where S = 0.5 * (G + G^T).
    Returns: (source_with_S2, out_array_name)
    """
    out_name = out_name or f"S2_{array_name}"

    PF = """
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support as ns
import numpy as np

inp = self.GetInputDataObject(0, 0)
wrap = dsa.WrapDataObject(inp)
pd = wrap.PointData
cd = wrap.CellData

name = "__GRAD__"
if name in pd.keys():
    data = pd
    out_to_points = True
elif name in cd.keys():
    data = cd
    out_to_points = False
else:
    raise RuntimeError("strain_rate PF: array '%s' not found in PointData or CellData." % name, nut)

arr = data[name]              # dsa array view; may be (N,9) or (N,3,3)

shape = arr.shape
# Normalize to a NumPy array and an (N,3,3) tensor stack
if len(shape) == 2 and shape[1] == 9:
    G = np.asarray(arr).reshape(-1, 3, 3)
elif len(shape) == 3 and shape[1] == 3 and shape[2] == 3:
    G = np.asarray(arr)  # already (N,3,3)
elif len(shape) == 2 and shape[1] == 4:
    # 2D vector gradient → pad to 3×3
    G = np.zeros((shape[0], 3, 3), dtype=float)
    flat = np.asarray(arr)
    # [dUx/dx, dUx/dy, dUy/dx, dUy/dy]
    G[:,0,0] = flat[:,0]; G[:,0,1] = flat[:,1]
    G[:,1,0] = flat[:,2]; G[:,1,1] = flat[:,3]
else:
    raise RuntimeError(
        f"strain_rate PF: unsupported gradient array shape {shape}; "
        f"expected (N,9) or (N,3,3) [3D], or (N,4) [2D]"
    )

# Debug: show what we actually saw
#print("[pf] reading", name, " assoc=", "POINTS" if out_to_points else "CELLS", " shape=", G.shape)

S  = 0.5 * (G + np.swapaxes(G, 1, 2))
S2 = np.sum(S * S, axis=(1, 2))

out = self.GetOutputDataObject(0)
out.ShallowCopy(inp)
arr = ns.numpy_to_vtk(S2.copy(), deep=1)
arr.SetName("__OUT__")
if out_to_points:
    out.GetPointData().AddArray(arr)
else:
    out.GetCellData().AddArray(arr)
""".lstrip()

    code = (PF
            .replace("__GRAD__", array_name)
            .replace("__OUT__", out_name))

    pf = ProgrammableFilter(Input=src)
    pf.Script = code
    pf.RequestInformationScript = ''
    pf.RequestUpdateExtentScript = ''
    pf.PythonPath = ''
    pf.UpdatePipeline()
    return pf, out_name

def add_fluctuation(src, base_array, avg_array, out_name):
    """
    Create fluctuation array: out_name = base_array - avg_array (component-wise).
    Works on PointData; assumes both arrays already exist there (your pipeline flattens first).
    """
    PF = """
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support as ns
import numpy as np

inp = self.GetInputDataObject(0, 0)
wrap = dsa.WrapDataObject(inp)
pd = wrap.PointData

a = "__A__"
b = "__B__"
if a not in pd.keys():
    raise RuntimeError("Base array '%s' not found in PointData." % a)
if b not in pd.keys():
    raise RuntimeError("Avg array '%s' not found in PointData." % b)

A = np.asarray(pd[a])
B = np.asarray(pd[b])
if A.ndim == 1: A = A.reshape(-1,1)
if B.ndim == 1: B = B.reshape(-1,1)
if A.shape != B.shape:
    raise RuntimeError("Shape mismatch: %s vs %s" % (A.shape, B.shape))

P = A - B

out = self.GetOutputDataObject(0)
out.ShallowCopy(inp)
Pv = ns.numpy_to_vtk(P.copy(), deep=1)
Pv.SetName("__OUT__")
out.GetPointData().AddArray(Pv)
""".lstrip()

    code = (
        PF.replace("__A__", base_array)
          .replace("__B__", avg_array)
          .replace("__OUT__", out_name)
    )

    pf = ProgrammableFilter(Input=src)
    pf.Script = code
    pf.RequestInformationScript = ''
    pf.RequestUpdateExtentScript = ''
    pf.PythonPath = ''
    pf.UpdatePipeline()
    return pf

def apply_gradient(src, array_name, assoc=None, opts=None):
    """
    Compute gradients of a scalar or vector array.
    - array_name: name of the array (string)
    - assoc: 'POINTS' or 'CELLS' (auto-detected if None)
    - opts: dict of extra flags, e.g. {
        'result_name': 'grad_field',
        'compute_vorticity': True,
        'vorticity_name': 'vort_field',
        'compute_divergence': False,
        'divergence_name': 'div_field',
        'compute_qcriterion': False,
        'qcriterion_name': 'Q_field',
      }
    Returns: (grad_filter_output, result_array_name)
    """
    if not isinstance(array_name, str) or not array_name:
        raise RuntimeError("apply_gradient: 'array_name' must be a non-empty string.")

    opts = opts or {}
    result_name = opts.get('result_name', f"grad_{array_name}")

    # Auto-detect association if not provided
    if assoc is None:
        pnames, cnames = list_point_cell_arrays(src)
        if array_name in pnames:
            assoc = 'POINTS'
        elif array_name in cnames:
            assoc = 'CELLS'
        else:
            raise RuntimeError(
                f"apply_gradient: array '{array_name}' not found. "
                f"Point arrays: {pnames}; Cell arrays: {cnames}"
            )
    else:
        assoc = assoc.upper()
        if assoc not in ('POINTS', 'CELLS'):
            raise RuntimeError("apply_gradient: 'assoc' must be 'POINTS' or 'CELLS'.")

    grad = Gradient(Input=src)
    grad.ScalarArray = [assoc, array_name]
    grad.ResultArrayName = result_name

    # Optional derived quantities
    if opts.get('compute_vorticity'):
        grad.ComputeVorticity = 1
        grad.VorticityArrayName = opts.get('vorticity_name', f"vort_{array_name}")
    if opts.get('compute_divergence'):
        grad.ComputeDivergence = 1
        grad.DivergenceArrayName = opts.get('divergence_name', f"div_{array_name}")
    if opts.get('compute_qcriterion'):
        grad.ComputeQCriterion = 1
        grad.QCriterionArrayName = opts.get('qcriterion_name', f"Q_{array_name}")

    grad.UpdatePipeline()
    return grad, result_name

def calculate_epsilon(src, s2_array, axis_letter='Y', result_name='eps', nut_name='nut', nu=1e-6):
    """
    Compute epsilon = <2*nut*S2>_axis + <2*nu*S2>_axis, where S2 is a scalar array (e.g., from strain-rate).
    Returns: (src_with_eps, result_name)
    """
    # --- check inputs exist somewhere ---
    pnames, cnames = list_point_cell_arrays(src)
    if (s2_array not in pnames) and (s2_array not in cnames):
        raise RuntimeError(f"calculate_epsilon: '{s2_array}' not found. Point arrays: {pnames}; Cell arrays: {cnames}")
    if (nut_name not in pnames) and (nut_name not in cnames):
        raise RuntimeError(f"calculate_epsilon: '{nut_name}' not found. Point arrays: {pnames}; Cell arrays: {cnames}")

    # --- ensure both live on points (averaging pipeline works on points) ---
    src_pts = ensure_points_for_array(src, s2_array)
    src_pts = ensure_points_for_array(src_pts, nut_name)

    # --- calculators for eps_t and eps_m (keep both by chaining) ---
    eps_t_name = f"eps_t_{s2_array}"
    calc_t = Calculator(Input=src_pts)
    calc_t.ResultArrayName = eps_t_name
    calc_t.Function = f"2*{nut_name}*{s2_array}"
    calc_t.UpdatePipeline()

    eps_m_name = f"eps_m_{s2_array}"
    calc_m = Calculator(Input=calc_t)
    calc_m.ResultArrayName = eps_m_name
    calc_m.Function = f"{2.0*float(nu)}*{s2_array}"
    calc_m.UpdatePipeline()
    # --- spanwise average each scalar we just created ---
    calc_m, eps_t_avg = apply_spanwise_average(calc_m, axis_letter=axis_letter, array_name=eps_t_name)
    calc_m, eps_m_avg = apply_spanwise_average(calc_m, axis_letter=axis_letter, array_name=eps_m_name)

    # --- sum the averaged parts into final epsilon ---
    calc_sum = Calculator(Input=calc_m)
    calc_sum.ResultArrayName = result_name
    calc_sum.Function = f"{eps_t_avg}+{eps_m_avg}"
    calc_sum.UpdatePipeline()

    return calc_sum, result_name

    
    
def calculate_k(src, prime_vec_name, axis_letter='Y', result_name='k'):
    """
    Compute turbulent kinetic energy-like quantity:
        k = 0.5 * ( <u'_x^2> + <u'_y^2> + <u'_z^2> )
    where <...> is spanwise average along axis_letter.

    Assumes `prime_vec_name` is a 3-component vector in PointData or CellData.
    Returns: (src_with_k, result_name)
    """

    # 0) Ensure the vector exists & find association
    pnames, cnames = list_point_cell_arrays(src)
    if prime_vec_name in pnames:
        assoc = 'POINTS'
    elif prime_vec_name in cnames:
        assoc = 'CELLS'
    else:
        raise RuntimeError(
            f"calculate_k: vector '{prime_vec_name}' not found. "
            f"Point arrays: {pnames}; Cell arrays: {cnames}"
        )

    # 1) If needed, convert to points so averaging works on points
    #    (apply_spanwise_average handles scalars in PointData best)
    src_pts = ensure_points_for_array(src, prime_vec_name)

    # 2) Make squared-component scalars via Calculator
    #    ParaView Calculator uses component names like <V>_X, <V>_Y, <V>_Z
    comps = ['X', 'Y', 'Z']
    comp_sq_names = []
    for c in comps:
        calc = Calculator(Input=src_pts)
        calc.ResultArrayName = f"{prime_vec_name.lower()}_{c.lower()}2"  # e.g., U_prime_Y_x2
        calc.Function = f"{prime_vec_name}_{c}*{prime_vec_name}_{c}"
        calc.UpdatePipeline()
        src_pts = calc  # chain filters
        comp_sq_names.append(calc.ResultArrayName)

    # 3) Spanwise-average each squared scalar
    avg_names = []
    for name in comp_sq_names:
        src_pts, avg_name = apply_spanwise_average(
            src_pts, axis_letter=axis_letter, array_name=name
        )
        avg_names.append(avg_name)

    # 4) Sum the averaged squares and multiply by 0.5 to get k
    expr = "0.5*(" + "+".join(avg_names) + ")"
    calc_k = Calculator(Input=src_pts)
    calc_k.ResultArrayName = result_name
    calc_k.Function = expr
    calc_k.UpdatePipeline()

    # Done
    return calc_k, result_name

def color_by_array_and_save_pngs(src, cfg, zmin=None, zmax=None, desired_array=None, *more_arrays):
    """
    Render 1 or many arrays.
    - Single array: behaves like before, saves into output_directory.
    - Multiple arrays: creates subfolders per array and saves there.
    zmin/zmax are accepted for future use (e.g., camera/clipping); ignored if None.
    """
    vis = cfg.get("visualization", {}) or {}
    img_size = vis.get("image_size") or [1200, 800]
    # ensure ints
    try:
        w = int(round(float(img_size[0]))); h = int(round(float(img_size[1])))
    except Exception:
        raise RuntimeError(f"Invalid visualization.image_size: {img_size}. Expected [width, height].")
    img_res = (w, h)

    cmap     = vis.get("color_map")
    rng      = vis.get("range")         # None or [min, max]
    show_bar = bool(vis.get("show_scalar_bar", False))
    bg       = vis.get("background", None)
    cam_plane = vis.get("camera_plane")
    out_array = vis.get("out_array")
    
    outdir_root = cfg.get("output_directory") or "."
    os.makedirs(outdir_root, exist_ok=True)
    if os.path.exists(outdir_root) and not os.path.isdir(outdir_root):
        raise RuntimeError(f"Path exists but is not a directory: {outdir_root}")

    # collect arrays to render
    arrays = []
    if desired_array is not None:
        if isinstance(desired_array, (list, tuple, set)):
            arrays.extend(list(desired_array))
        else:
            arrays.append(desired_array)
    if more_arrays:
        arrays.extend(list(more_arrays))
    if not arrays:
        # fallback to config if nothing explicitly passed
        a = vis.get("array")
        if not a:
            raise RuntimeError("No array(s) provided for visualization.")
        arrays = [a]

    # One render-view reused across arrays for speed
    view = GetActiveViewOrCreate('RenderView')
    if bg and isinstance(bg, (list, tuple)) and len(bg) == 3:
        view.Background = bg
    view.ViewSize = [w, h]

    # common helper: resolve & render a single array to a specific folder
    def _render_one(target_array, folder):
        # resolve suffixes like 'U_avg'/'U_prime' → add axis if needed
        averaging = (cfg.get("averaging") or {})
        axis_letter = (averaging.get("axis") or "Y").upper()

        arr = str(target_array)
        # If user asked 'X_avg' without axis, append default axis if not found
        if arr.endswith("_avg") and f"{arr}_{axis_letter}" not in list_point_cell_arrays(src)[0]:
            arr = f"{arr}_{axis_letter}"
        if arr.endswith("_prime") and f"{arr}_{axis_letter}" not in list_point_cell_arrays(src)[0]:
            arr = f"{arr}_{axis_letter}"

        pnames, cnames = list_point_cell_arrays(src)
        if arr in pnames:
            assoc = "POINTS"
        elif arr in cnames:
            assoc = "CELLS"
        else:
            raise RuntimeError(f"Requested array '{arr}' not found. "
                               f"POINT arrays: {pnames}; CELL arrays: {cnames}")

        # show & color
        disp = Show(src, view)
        view.Update()

        # optional: orient camera (XZ by default)
        try:
            set_camera_plane(view, src, cfg, zmin, zmax, plane=cam_plane)
        except Exception:
            RuntimeError(f"Cannot set the camera")

        ncomp = get_array_components(src, assoc, arr)
        if ncomp and ncomp > 1:
            ColorBy(disp, (assoc, arr, "Magnitude"))
        else:
            ColorBy(disp, (assoc, arr))
        disp.SetScalarBarVisibility(view, show_bar)
        view.Update()

        # colormap + range
        lut = GetColorTransferFunction(arr)
        if "eps" in out_array:
            lut.UseLogScale=1
        _apply_preset_safe(lut, str(cmap), view, vis)
        if rng and isinstance(rng, (list, tuple)) and len(rng) == 2:
            r0, r1 = float(rng[0]), float(rng[1])
            if not (r1 > r0):
                raise RuntimeError("Invalid 'range'; expected [min, max] with max > min.")
            lut.RescaleTransferFunction(r0, r1)
            pwf = GetOpacityTransferFunction(arr)
            pwf.RescaleTransferFunction(r0, r1)

        # time handling
        tk = GetTimeKeeper()
        times = list(getattr(tk, "TimestepValues", []) or [])
        if not times:
            times = list(getattr(src, "TimestepValues", []) or [])
        os.makedirs(folder, exist_ok=True)
        start_time = cfg.get("start_time")
        end_time = cfg.get("end_time")
        if times:
            for t in times:
                if (start_time is not None and end_time is not None):
                    if (start_time <= t <= end_time):
                        GetAnimationScene().AnimationTime = t
                        view.Update()
                        fname = f"{arr}_t_{_safe_time_str(t)}.png"
                        SaveScreenshot(os.path.join(folder, fname), view, ImageResolution=img_res)
                        print(f"[pvpython-child] Saved {os.path.join(folder, fname)}")
                else:
                    GetAnimationScene().AnimationTime = t
                    view.Update()
                    fname = f"{arr}_t_{_safe_time_str(t)}.png"
                    SaveScreenshot(os.path.join(folder, fname), view, ImageResolution=img_res)
                    print(f"[pvpython-child] Saved {os.path.join(folder, fname)}")
        else:
            view.Update()
            fname = f"{arr}_t_static.png"
            SaveScreenshot(os.path.join(folder, fname), view, ImageResolution=img_res)
            print(f"[pvpython-child] Saved {os.path.join(folder, fname)}")

        # hide display before next array
        Hide(src, view)
        view.Update()

    # single vs multiple
    if len(arrays) == 1:
        subdir = os.path.join(outdir_root, str(arrays[0]))
        _render_one(arrays[0], subdir)
    else:
        for arr in arrays:
            subdir = os.path.join(outdir_root, str(arr))
            _render_one(arr, subdir)


if __name__ == "__main__":
    raise SystemExit(main())
'''

# -------------------------------
# Driver utilities
# -------------------------------
def find_input_files(cfg: dict) -> list:
    base = Path(cfg['base_directory']).expanduser().resolve()
    if cfg['pattern_type'] == 'glob':
        return [str(p) for p in sorted(base.glob(cfg['file_template']))]
    return []

def run_pvpython_child(script_text: str, files: list, cfg_obj: dict) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as sfile, \
         tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as cfile:
        sfile.write(script_text)
        script_path = sfile.name
        json.dump(cfg_obj, cfile)
        cfg_path = cfile.name
    
    # Build base command from config
    exe = PROCESSING_OPTIONS.get('paraview_executable', 'pvpython')
    extra = PROCESSING_OPTIONS.get('paraview_args', []) or []
    if not isinstance(extra, (list, tuple)):
        raise RuntimeError("PROCESSING_OPTIONS['paraview_args'] must be a list")
    
    base = [str(exe)] + [str(a) for a in extra] + [script_path, "--config-file", cfg_path] + [str(f) for f in files]
    
    # Prepend MPI launcher if enabled
    if MPI.get("enabled"):
        launch = [str(MPI.get('launcher', 'mpiexec')), "-n", str(MPI.get('n', 2))]
        launch += [str(a) for a in (MPI.get('extra_args', []) or [])]
        cmd = launch + base
    else:
        cmd = base
    
    print("[driver] Running:", " ".join(shlex.quote(c) for c in cmd))
    
    # Ensure unbuffered Python in the child
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
        except FileNotFoundError:
            # Fallback if 'stdbuf' is not installed
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
        
        # Stream output as it arrives
        for line in proc.stdout:
            print(line, end="")  # already includes newline
            sys.stdout.flush()
        
        proc.wait()
        return proc.returncode
    finally:
        for p in (script_path, cfg_path):
            try:
                os.remove(p)
            except OSError:
                pass

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve output directory to absolute path before sending to child
    cfg = dict(INPUT_PARAMETERS)
    outdir = Path(INPUT_PARAMETERS.get('output_directory', './')).expanduser().resolve()
    cfg['output_directory'] = str(outdir)

    files = find_input_files(cfg)
    if not files:
        logging.error("No files matched pattern %r in %s",
                      cfg['file_template'],
                      Path(cfg['base_directory']).resolve())
        return 1

    os.makedirs(outdir, exist_ok=True)

    rc = run_pvpython_child(
        SCRIPT_CONTENT,
        files=files,
        cfg_obj=cfg
    )
    if rc != 0:
        logging.error("Child pvpython exited with code %d", rc)
    else:
        logging.info("Child pvpython completed successfully. Images saved to %s", outdir)
    return rc

if __name__ == "__main__":
    sys.exit(main())
