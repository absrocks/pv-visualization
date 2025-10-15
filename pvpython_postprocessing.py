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
    'start_time': 0,        # None --> to start from 0
    'end_time': 6,

    # ---- Averaging Options ----
    'averaging': {
        'axis': 'Y',        # 'X' | 'Y' | 'Z'
    },
    'clipping': {
        'enabled': False,      # set False to disable
        'axis': 'X',          # 'X' | 'Y' | 'Z'
        'Xmin': 21.0,
        'Xmax': 34.0,
    },
    # ---- OpenFOAM-specific options ----
    'openfoam': {
        'mode': 'decomposed',            # 'reconstructed' | 'decomposed' | 'auto'
        'mesh_regions': ['internalMesh'],   # or [] / None
        'cell_arrays':  ['U', 'alpha.water', 'nut', 'UAvg'],         # or [] / None , 'UAvg', 'nut
        'point_arrays': ['U', 'alpha.water', 'nut', 'UAvg'],         # e.g., ['T']
    },

    # ---- Visualization options ----
    'visualization': {
        'image_size': [1200, 800],          # [width, height]
        'color_map': 'Jet',                 # colormap preset name
        'array': 'U',                       # REQUIRED: array to visualize
        'out_array': 'U',
        'range': [0, 2],                      # e.g., [0.0, 5.0]; None = auto
        'custom_label': None, # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],  # e.g. None
        'label_format': '6.2f',  # '6.1e' | '6.2f'
        'show_scalar_bar': True,            # show scalar bar
        'background': [1, 1, 1],            # white background
        'camera_plane': 'XZ',    # NEW: 'XZ' | 'XY' | 'YZ'
        'axis': True,
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
    "n": 64,                            # number of ranks
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
import numpy as np
import builtins as _bi

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

    if npts == 0:
        print("ERROR: No data points found", file=sys.stderr)
        return 4

    if cfg.get("clipping")["enabled"] is True:
        src = apply_clipping(src, cfg)
        
    if cfg.get("visualization")["axis"] is True:
        src= apply_slices(src)
    
    
    # Apply IsoVolume
    src = apply_isovolume(src, cfg)
    
    
    # FLATTEN first, so everything downstream sees real vtkDataArrays:
    src = flatten_dataset(src)

    # ---- Decide/compute derived arrays if requested ----
    vis_array = cfg.get("visualization")["array"]
    averaging = cfg.get("averaging")
    default_axis = averaging.get("axis")
    effective_vis_array = vis_array
    (xmin,xmax,ymin,ymax,zmin,zmax) =_domain_bounds(src)
    
    
    try:
        base = vis_array
        axis_letter = default_axis
        # Compute average
        
        if 'k' in cfg.get("visualization")["out_array"]:
            print(f"[pvpython-child] TKE output will be written")
            src, avg_name = apply_spanwise_average(src, axis_letter=axis_letter, array_name=base)
            print(f"[pvpython-child] Calculated array: {avg_name}")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, k_name = calculate_k(src, prime_vec_name=prime_name, axis_letter=axis_letter, result_name="TKE")
            effective_vis_array = k_name
            print(f"[pvpython-child] Added array: {k_name}")
        
        if 'eps' in cfg.get("visualization")["out_array"]:
            print(f"[pvpython-child] Epsilon output will be written")
            src, avg_name = apply_spanwise_average(src, axis_letter=axis_letter, array_name=base)
            print(f"[pvpython-child] Calculated array: {avg_name}")
            prime_name = f"{base}_prime_{axis_letter}"
            src = add_fluctuation(src, base_array="U", avg_array=avg_name, out_name=prime_name)
            src, grad_name = apply_gradient(src, prime_name)
            src, s2_name = strain_rate(src, array_name=grad_name, out_name="S2")
            src, eps_name = calculate_epsilon(src, s2_name, axis_letter=axis_letter, result_name='epsilon')
            effective_vis_array = eps_name
            print(f"[pvpython-child] Added array: {eps_name}")
        
    except Exception as e:
        print(f"[pvpython-child][ERROR] Averaging/fluctuation step failed: {e}", file=sys.stderr)
        return 6
    
    # ---- Render & save ----
    try:
        src, avg_name = apply_spanwise_average(src, axis_letter=axis_letter, array_name=base)
        print(f"[pvpython-child] Calculated array: {avg_name}")
        #src, zmax, array_max = print_array_stats(src, avg_name)
        src = get_coords(src, cfg, base)
        #effective_vis_array = avg_name
        #color_by_array_and_save_pngs(src, cfg, zmin, zmax, desired_array=effective_vis_array)
    except Exception as e:
        print(f"[pvpython-child][ERROR] Visualization failed: {e}", file=sys.stderr)
        return 7

    print("[pvpython-child] Completed successfully.")
    return 0

def get_coords(src, cfg, array, axis_letter='Y'):
    """
    For each timestep in the source, compute spanwise-average of `base_array`,
    then print bounds at that time. Returns the averaging filter so caller can reuse.
    """
    # 0) Gather times
    tk = GetTimeKeeper()
    times = list(getattr(tk, "TimestepValues", []) or [])
    if not times:
        times = list(getattr(src_base, "TimestepValues", []) or [])
    if not times:
        print("[pvpython-child] get_coords: no timesteps found", flush=True)
        return src

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
            
        xmin, xmax, zmin, zmax, amax = print_array_stats(src, array)
        # Query bounds on the averaged output (geometry is unchanged by averaging)
        (xxmin,xxmax,yymin,yymax,zzmin,zzmax) =_domain_bounds(src)
        print(f"[pvpython-child] bounds at t={t}: "
              f"{array} maximum is {array_max:6g}, x[{xmin:.6g} {xmax:.6g}] z[{zmin:.6g} {zmax:.6g}], with "
              f"bounds {xxmin,xxmax,yymin,yymax,zzmin,zzmax}",
              flush=True)

    return src


def print_array_stats(src, name, sample=5, label=None):
    """
    Print where the array lives (POINTS/CELLS), its shape, min/max/mean,
    and the first few tuples.
    """
    
        # Presence / association on proxy
    pnames, cnames = list_point_cell_arrays(src)
    if name in pnames:
        assoc = 'POINTS'
    elif name in cnames:
        assoc = 'CELLS'
    else:
        print(f"[stats] '{name}' not found (POINTS={pnames}, CELLS={cnames})")
        return
        
    # Current scene time
    tk = GetTimeKeeper()
    cur_t = getattr(tk, "Time", None)

    # Flatten and force execution specifically *at* cur_t
    flat = MergeBlocks(Input=src)
    try:
        flat.UpdatePipeline(time=cur_t)
    except Exception:
        flat.UpdatePipeline()

    # Fetch concrete VTK dataset
    vtkobj = sm.Fetch(flat)
    if vtkobj is None:
        print("[stats] Fetch returned None (nothing to inspect).")
        return

    # Wrap and get array
    wrap = dsa.WrapDataObject(vtkobj)
    data = wrap.PointData if assoc == 'POINTS' else wrap.CellData
    if name not in data.keys():
        print(f"[stats] '{name}' not present on fetched {assoc} at current time.")
        return
    wrap = dsa.WrapDataObject(vtkobj)
    pts  = wrap.Points
    if pts is None:
        raise RuntimeError("get_xyz_coords: dataset has no points.")
    xyz = np.asarray(pts, dtype=float)
    
    # Force a *pure NumPy* view to avoid vtk numpy_interface reducers
    arr = data[name]
    np_arr = np.array(arr, dtype=float, copy=False)
    zmax = np.max(xyz[:,2])
    zmin = min(xyz[:,2])
    xmax = max(xyz[:,0])
    xmin = min(xyz[:,0])
    amax = np_arr.max()
    
    src.UpdatePipeline()
    
    return xmin, xmax, zmin, zmax, amax

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

def apply_slices(src):

    # create a new 'Extract Surface'
    cur = MergeBlocks(Input=src)
    #src.UpdatePipeline()
    (xmin,xmax,ymin,ymax,zmin,zmax) = _domain_bounds(src)
    pos   = [(xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2]
    
    # Apply Slice
    slice1 = Slice(registrationName='Slice1', Input=cur)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    
    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = pos
    slice1.SliceType.Normal = [0.0, 1.0, 0.0]
    #slice1.UpdatePipeline()
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
    
def apply_clipping(src, cfg):
    """
    If cfg['clipping'] is enabled, apply a Box clip:
      - along the requested axis, use [min, max] from config
      - along the other axes, span the whole domain
    Returns the clipped source (Clip filter output).
    """
    clip_cfg = cfg.get('clipping')
    axis = clip_cfg.get('axis')
    if axis not in ('X','Y','Z'):
        raise RuntimeError(f"Invalid clipping axis: {axis}")

    # Figure out which keys to read
    min_key = f"{axis}min"
    max_key = f"{axis}max"
    if (min_key not in clip_cfg) or (max_key not in clip_cfg):
        raise RuntimeError(f"Clipping requires '{min_key}' and '{max_key}' in config.")

    amin = float(clip_cfg[min_key])
    amax = float(clip_cfg[max_key])
    if not (amax > amin):
        raise RuntimeError(f"Clipping {axis} range must have max > min (got {amin}, {amax}).")

    # Use domain bounds to span the other two axes
    (xmin,xmax,ymin,ymax,zmin,zmax) = _domain_bounds(src)
    
    pos   = [xmin, ymin, zmin]
    leng  = [xmax - xmin, ymax - ymin + 1, zmax - zmin + 1]

    i = _axis_index(axis)
    pos[i]  = amin
    leng[i] = amax - amin

    # Build a Box clip
    clip1 = Clip(Input=src)
    clip1.ClipType = 'Box'
    # If your ParaView build exposes HyperTreeGridClipper/Scalars/Value, leave them untouched;
    # we just use the Box to cut a spatial slab.
    clip1.ClipType.Position = pos
    clip1.ClipType.Length   = leng
    clip1.UpdatePipeline()

    print(f"[pvpython-child] Applied Box clip on {axis} in [{amin}, {amax}]")
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
    else:
        # XZ (default) -> look along +Y
        view.CameraPosition = [cx, -cy - R, cz]
        view.CameraFocalPoint = [cx, cy, cz]
        view.CenterOfRotation = [cx, cy, cz]
        
        view.CameraViewUp = [0, 0, 1]
        view.CameraFocalDisk = 1.0
        view.CameraParallelProjection = 1
        if cfg.get("visualization")["axis"] is True:
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
            pass

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
