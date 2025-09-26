import subprocess
import os
import sys
import glob
import re
import tempfile
import logging
from pathlib import Path
from datetime import datetime

# --- File Pattern Configuration ---
FILE_PATTERNS = {
    'pattern_type': 'iteration',
    'base_directory': './',
    'file_template': '*.foam',
    'number_range': None,  # None for auto-detection, or [start, end, step]
}

# --- Analysis Configuration ---
BATCH_CONFIG = {

    # ==============================================================================
    # SELECT VARIABLE
    # ==============================================================================
    # 'data_array': 'w',
    'array_U': 'U',
    'array_alpha1': 'alpha.water',
    'array_nuSgs':  'nut',
    'merge_blocks': 'True',  # True if case is decomposed/multiblock (recommended!)
    'nu': 1e-6,  # molecular viscosity
    'bin_scale': '1.0',  # >1 coarsens bins (fewer bins -> faster Delaunay)
    'force_global': 'True',

    # ==============================================================================
    # SLICING OR AVERAGING?
    # ==============================================================================
    # 'analysis_mode': 'averaging',  # 'averaging' or 'slicing'
    'analysis_mode': 'clipping',  # 'averaging' or 'slicing'

    # ==============================================================================
    # AVERAGING
    # ==============================================================================
    'averaging': {
        'axis': 'Y',
    },

    # ==============================================================================
    # SLICING
    # ==============================================================================
    'clipping': {
        'axis': 'X',
        'Xmin': 20,  # Use None for auto-center
        'Xmax': 28,
    },

    'visualization': {
        'image_size': [1200, 800],
        'color_map': 'jet',
    }
}

# --- Processing Options ---
PROCESSING_OPTIONS = {
    'output_directory': './batch_output/',
    'continue_on_error': True,
    'paraview_executable': 'pvpython',
    'paraview_args': ['--force-offscreen-rendering'],
    'timeout_seconds': 300,  # 5 minutes per file
    'log_file': 'batch_processing.log'
}


# ==============================================================================
# IMPLEMENTATION
# ==============================================================================

def setup_logging():
    """Setup logging for the batch processor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROCESSING_OPTIONS['log_file']),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)



def find_files():
    """Find and sort files matching the pattern."""
    base_dir = Path(FILE_PATTERNS['base_directory'])
    template = FILE_PATTERNS['file_template']
    number_range = FILE_PATTERNS.get('number_range', None)

    if number_range is not None:
        # Use specified range
        files = []
        start, end, step = number_range
        current = start
        while current <= end:
            filename = base_dir / template.format(current)
            if filename.exists():
                files.append(str(filename.absolute()))
            current += step
        return sorted(files)

    else:
        # Auto-detect files
        glob_pattern = template.replace('{}', '*')
        search_path = base_dir / glob_pattern
        matching_files = glob.glob(str(search_path))

        if not matching_files:
            return []

        # Sort by extracted number
        files_with_numbers = []
        regex_template = re.escape(template).replace(r'\{\}', r'([+-]?(?:\d+\.?\d*|\.\d+))')

        for file_path in matching_files:
            abs_path = os.path.abspath(file_path)
            filename = Path(file_path).name
            match = re.match(regex_template, filename)

            if match:
                try:
                    number_str = match.group(1)
                    try:
                        number = int(number_str)
                    except ValueError:
                        number = float(number_str)
                    files_with_numbers.append((number, abs_path))
                except (ValueError, IndexError):
                    continue

        files_with_numbers.sort(key=lambda x: x[0])
        return [file_path for _, file_path in files_with_numbers]


def generate_output_filename(input_file):
    """Generate output filename for the processed image."""

    # Format number for consistent sorting

    mode = 'avg' if BATCH_CONFIG['analysis_mode'] == 'averaging' else 'slice'
    axis = (BATCH_CONFIG['averaging']['axis'] if BATCH_CONFIG['analysis_mode'] == 'averaging'
            else BATCH_CONFIG['slicing']['axis'])

    filename = f"{BATCH_CONFIG['data_array']}_{mode}_{axis}_{number_str}.png"
    output_dir = Path(PROCESSING_OPTIONS['output_directory'])
    return str(output_dir / filename)

files = sorted(glob.glob("/project/smarras/am2455/wave_beach_profile/wang_kraus_scaled/pv_clip/slope_26.vtm"))

if not files:
    raise RuntimeError("No slope_*.vtm files found!")


def generate_processing_script(input_file, output_file):
    """Generate a standalone script to process a single file."""

    # Convert paths to absolute to avoid issues
    input_abs = os.path.abspath(input_file)
    output_abs = os.path.abspath(output_file)

    script_content = f"""#!/usr/bin/env python3
import os
import sys
from paraview.simple import *
src = XMLMultiBlockDataReader(FileName=files)
src.UpdatePipeline()
cur = src
if merge_blocks:
    try:
        cur = MergeBlocks(Input=cur); cur.UpdatePipeline()
    except Exception:
        pass

if force_global:
    d3 = RedistributeDataSet(Input=cur)
    # try all known property names across ParaView versions
    set_ok = False
    for prop in ("TargetPartitions", "NumberOfPartitions", "TargetPartitionCount"):
        try:
            setattr(d3, prop, 1)
            set_ok = True
            break
        except AttributeError:
            pass
    # best-effort: disable preserving original partitions
    for prop in ("PreservePartitions", "PreservePartitioning"):
        try:
            setattr(d3, prop, 0)
            break
        except AttributeError:
            pass
    if not set_ok:
        print("[warn] Could not set target partition count; using filter defaults.")
    d3.UpdatePipeline()
    cur = d3

if use_cell_centers:
    cc = CellCenters(Input=cur); cc.VertexCells = 0; cc.UpdatePipeline()
    cur = cc

# ---------- PF #1: compute U' (keep topology), pass through required arrays ----------
pf_prepare = ProgrammableFilter(Input=cur)


SCRIPT1 = (
    'import builtins, numpy as np\n'
    'from vtkmodules.util import numpy_support as ns\n'
    'import vtk\n'
    f'array_U      = "{BATCH_CONFIG["array_U"]}"\n'
    f'array_alpha1 = "{BATCH_CONFIG["array_alpha1"]}"\n'
    f'array_nuSgs  = "{BATCH_CONFIG["array_nuSgs"]}"\n'
    f'span_dir     = "{BATCH_CONFIG["averaging"]["axis"]}".lower()\n'
    f'bin_scale    = {BATCH_CONFIG["bin_scale"]}\n'
    f'nu           = {BATCH_CONFIG["nu"]}\n'
)

SCRIPT1 += '''
def robust_tolerance(vals):
    arr = np.asarray(vals, dtype=float).ravel()
    if arr.size < 3: return 1e-9
    v = np.unique(np.sort(arr))
    if v.size < 3: return 1e-9
    d = np.diff(v)
    h = np.percentile(d, 10)
    if not np.isfinite(h) or h <= 0:
        med = np.median(d); h = med if (np.isfinite(med) and med > 0) else 1e-9
    return builtins.max(h*0.5, 1e-9)

def ensure_NxC(A, N_expected):
    A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape((-1,1))
    elif A.ndim == 2 and A.shape[0] != N_expected and A.shape[1] == N_expected:
        A = A.T
    if A.shape[0] != N_expected:
        raise RuntimeError("Array length mismatch: expected N=%d, got %s" % (N_expected, A.shape))
    return A

def bin_keys(pts, span_dir, bin_scale=1.0):
    ax = {"x":0,"y":1,"z":2}[span_dir]
    keep = [i for i in (0,1,2) if i != ax]
    k1, k2 = keep
    tol1 = robust_tolerance(pts[:,k1]) * float(bin_scale)
    tol2 = robust_tolerance(pts[:,k2]) * float(bin_scale)
    o1 = float(np.min(pts[:,k1])); o2 = float(np.min(pts[:,k2]))
    i1 = np.rint((pts[:,k1]-o1)/tol1).astype(np.int64)
    i2 = np.rint((pts[:,k2]-o2)/tol2).astype(np.int64)
    return (ax, k1, k2, i1, i2)

inp = self.GetInput()
pts_vtk = inp.GetPoints()
if pts_vtk is None: raise RuntimeError("No points on input.")
pts = ns.vtk_to_numpy(pts_vtk.GetData()); N = pts.shape[0]

pd = inp.GetPointData()
U     = pd.GetArray(array_U)
alpha = pd.GetArray(array_alpha1)
nusgs = pd.GetArray(array_nuSgs)
if U is None:     raise RuntimeError("Point-data array '%s' not found." % array_U)
if alpha is None: raise RuntimeError("Point-data array '%s' not found." % array_alpha1)
if nusgs is None: raise RuntimeError("Point-data array '%s' not found." % array_nuSgs)

U     = ensure_NxC(ns.vtk_to_numpy(U),     N)  # (N,3)
alpha = ensure_NxC(ns.vtk_to_numpy(alpha), N)  # (N,1)
nusgs = ensure_NxC(ns.vtk_to_numpy(nusgs), N)  # (N,1)
if U.shape[1] != 3: raise RuntimeError("'%s' must be 3 components." % array_U)

# Build U_bar per bin, then U' per point
ax,k1,k2,i1,i2 = bin_keys(pts, span_dir, bin_scale=bin_scale)
sums={}; cnts={}
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    if key not in sums: sums[key]=U[n].astype(float).copy(); cnts[key]=1
    else: sums[key]+=U[n]; cnts[key]+=1
Ubar_bin = {k: (sums[k]/cnts[k]) for k in sums}

Uprime = np.zeros_like(U)
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    Uprime[n,:] = U[n,:] - Ubar_bin[key]

# Output: keep topology, add arrays U_prime (for Gradient), keep alpha and nuSgs present
out = self.GetOutput()
out.ShallowCopy(inp)
Up_vtk   = ns.numpy_to_vtk(Uprime, deep=1); Up_vtk.SetName(array_U + "_prime")
out.GetPointData().AddArray(Up_vtk)
# alpha and nuSgs already on the data; no need to duplicate
'''
#ghostCells1 = GhostCells(registrationName='GhostCells1', Input=slope_0vtm)
#ghostCells1.BuildIfRequired = 1
if force_global:
    try:
        ghost = GhostCells(Input=pf_prepare)
        ghost.MinimumNumberOfGhostLevels = 1  # try 1 first; 2 if gradients still noisy at partition seams
        ghost.UpdatePipeline()
        grad_input = ghost
        RenameSource("Ghosted_Uprime", ghost)
        Show(ghost);
        Render()
    except Exception:
        # If the filter isn't available for the dataset type, just fall back
        grad_input = pf_prepare
        print("[warn] GhostCellsGenerator not available; using pf_prepare directly for Gradient.")
else:
    grad_input = pf_prepare

pf_prepare.Script = SCRIPT1
RenameSource("Uprime_prepare", pf_prepare)
Show(pf_prepare); Render()

# ---------- Gradient(U') using the 'Gradient' filter ----------
gradient = Gradient(registrationName='GradUprime', Input=grad_input)
gradient.ResultArrayName = "GradUprime"
gradient.FasterApproximation = 0
# Primary property on this proxy is 'ScalarArray' (works for vectors too):
set_ok = True
try:
    gradient.ScalarArray = ['POINTS', array_U + "_prime"]
except AttributeError:
    set_ok = False
if not set_ok:
    # Fallback for older wrappers: SetInputArrayToProcess
    from paraview import vtk
    gradient.SetInputArrayToProcess(
        0,  # idx
        0,  # port
        0,  # connection
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        array_U + "_prime"
    )
gradient.UpdatePipeline()
RenameSource("GradUprime", gradient)
Show(gradient); Render()

# ---------- PF #2: per-bin reduced output (U_bar, k_avg, eps, alpha1_bar) ----------
pf_bins = ProgrammableFilter(Input=gradient)
pf_bins.OutputDataSetType = 'vtkPolyData'

SCRIPT2 = (
    'import builtins, numpy as np\n'
    'from vtkmodules.util import numpy_support as ns\n'
    'import vtk\n'
    f'array_U      = "{array_U}"\n'
    f'array_alpha1 = "{array_alpha1}"\n'
    f'array_nuSgs  = "{array_nuSgs}"\n'
    'grad_name    = "GradUprime"\n'
    f'span_dir     = "{span_dir}".lower()\n'
    f'bin_scale    = {bin_scale}\n'
    f'nu           = {nu}\n'
)

SCRIPT2 += '''
def robust_tolerance(vals):
    arr = np.asarray(vals, dtype=float).ravel()
    if arr.size < 3: return 1e-9
    v = np.unique(np.sort(arr))
    if v.size < 3: return 1e-9
    d = np.diff(v)
    h = np.percentile(d, 10)
    if not np.isfinite(h) or h <= 0:
        med = np.median(d); h = med if (np.isfinite(med) and med > 0) else 1e-9
    return builtins.max(h*0.5, 1e-9)

def ensure_NxC(A, N_expected):
    A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape((-1,1))
    elif A.ndim == 2 and A.shape[0] != N_expected and A.shape[1] == N_expected:
        A = A.T
    if A.shape[0] != N_expected:
        raise RuntimeError("Array length mismatch: expected N=%d, got %s" % (N_expected, A.shape))
    return A

def bin_keys(pts, span_dir, bin_scale=1.0):
    ax = {"x":0,"y":1,"z":2}[span_dir]
    keep = [i for i in (0,1,2) if i != ax]
    k1, k2 = keep
    tol1 = robust_tolerance(pts[:,k1]) * float(bin_scale)
    tol2 = robust_tolerance(pts[:,k2]) * float(bin_scale)
    o1 = float(np.min(pts[:,k1])); o2 = float(np.min(pts[:,k2]))
    i1 = np.rint((pts[:,k1]-o1)/tol1).astype(np.int64)
    i2 = np.rint((pts[:,k2]-o2)/tol2).astype(np.int64)
    return (ax, k1, k2, i1, i2)

def tensor_from_vtk9(T):
    # [Gxx,Gxy,Gxz, Gyx,Gyy,Gyz, Gzx,Gzy,Gzz]
    return np.array([[T[0], T[1], T[2]],
                     [T[3], T[4], T[5]],
                     [T[6], T[7], T[8]]], dtype=float)

inp = self.GetInput()
pts_vtk = inp.GetPoints()
if pts_vtk is None: raise RuntimeError("No points on pf_bins input.")
pts = ns.vtk_to_numpy(pts_vtk.GetData()); N = pts.shape[0]

pd = inp.GetPointData()
U      = ensure_NxC(ns.vtk_to_numpy(pd.GetArray(array_U)),      N)  # (N,3)
alpha1 = ensure_NxC(ns.vtk_to_numpy(pd.GetArray(array_alpha1)), N)  # (N,1)
nusgs  = ensure_NxC(ns.vtk_to_numpy(pd.GetArray(array_nuSgs)),  N)  # (N,1)
Grad   = ns.vtk_to_numpy(pd.GetArray(grad_name))                        # (N,9)

if U.shape[1] != 3 or Grad.shape[1] != 9:
    raise RuntimeError("Unexpected component counts: U:%s Grad:%s" % (U.shape, Grad.shape))

# Per-bin indices
ax,k1,k2,i1,i2 = bin_keys(pts, span_dir, bin_scale=bin_scale)

# U_bar and centroids
sums={}; cnts={}; pos={}
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    if key not in sums:
        sums[key]=U[n].astype(float).copy(); cnts[key]=1; pos[key]=pts[n].astype(float).copy()
    else:
        sums[key]+=U[n]; cnts[key]+=1; pos[key]+=pts[n]
Ubar_bin = {k: (sums[k]/cnts[k]) for k in sums}
cent_bin = {k: (pos[k]/cnts[k])  for k in pos}

# k_avg from U' squares
sqsum={}
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    up = U[n,:] - Ubar_bin[key]
    if key not in sqsum: sqsum[key]=(up*up).copy()
    else: sqsum[key]+= (up*up)
msq = {k: (sqsum[k]/cnts[k]) for k in sqsum}
kbin = {k: 0.5*float(msq[k][0]+msq[k][1]+msq[k][2]) for k in msq}

# eps via S' from Grad(U')
ss_sums = {}; ss_nusgs_sums = {}
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    G = tensor_from_vtk9(Grad[n])
    S = 0.5*(G + G.T)
    ss = float(np.sum(S*S))
    if key not in ss_sums:
        ss_sums[key] = ss
        ss_nusgs_sums[key] = ss * float(nusgs[n,0])
    else:
        ss_sums[key]      += ss
        ss_nusgs_sums[key]+= ss * float(nusgs[n,0])
avg_ss       = {k: (ss_sums[k]      / cnts[k]) for k in ss_sums}
avg_nusgs_ss = {k: (ss_nusgs_sums[k]/ cnts[k]) for k in ss_nusgs_sums}
eps_bin      = {k: 2.0*(nu*avg_ss[k] + avg_nusgs_ss[k]) for k in avg_ss}

# alpha1_bar per bin
sumsA={}
for n in range(N):
    key=(int(i1[n]),int(i2[n]))
    if key not in sumsA: sumsA[key] = float(alpha1[n,0])
    else: sumsA[key] += float(alpha1[n,0])
alpha1_bin = {k: (sumsA[k]/cnts[k]) for k in sumsA}

# Assemble per-bin outputs (M points), flattened onto kept plane
keys = list(Ubar_bin.keys()); M = len(keys)
XYZ = np.zeros((M,3)); UBAR = np.zeros((M,3)); K = np.zeros((M,1)); EPS = np.zeros((M,1)); A1 = np.zeros((M,1))
for j,key in enumerate(keys):
    mpos = cent_bin[key]
    XYZ[j,:] = mpos
    XYZ[j, ax] = 0.0
    UBAR[j,:]  = Ubar_bin[key]
    K[j,0]     = kbin[key]
    EPS[j,0]   = eps_bin[key]
    A1[j,0]    = alpha1_bin[key]

outPD = vtk.vtkPolyData()
pts_out = vtk.vtkPoints(); pts_out.SetData(ns.numpy_to_vtk(XYZ, deep=1))
outPD.SetPoints(pts_out)
Ubar_vtk = ns.numpy_to_vtk(UBAR, deep=1); Ubar_vtk.SetName(array_U + "_bar")
k_vtk    = ns.numpy_to_vtk(K,    deep=1); k_vtk.SetName("k_avg")
eps_vtk  = ns.numpy_to_vtk(EPS,  deep=1); eps_vtk.SetName("eps")
a1_vtk   = ns.numpy_to_vtk(A1,   deep=1); a1_vtk.SetName(array_alpha1 + "_bar")
outPD.GetPointData().AddArray(Ubar_vtk)
outPD.GetPointData().AddArray(k_vtk)
outPD.GetPointData().AddArray(eps_vtk)
outPD.GetPointData().AddArray(a1_vtk)
self.GetOutput().ShallowCopy(outPD)
print("[bins] M =", M, "-> arrays: U_bar, k_avg, eps, alpha1_bar")
'''

pf_bins.Script = SCRIPT2
RenameSource("Ubar_k_eps_alpha_bins_PF", pf_bins)
Show(pf_bins); Render()
print("[OK] Created: Uprime_prepare (internal), GradUprime (internal), Ubar_k_eps_alpha_bins_PF (final per-bin).")

# create a new 'Delaunay 2D'
delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=pf_bins)
delaunay2D1.ProjectionPlaneMode = 'Best-Fitting Plane'

# create a new 'Iso Volume'
isoVolume1 = IsoVolume(registrationName='IsoVolume1', Input=delaunay2D1)
isoVolume1.InputScalars = ['POINTS', array_alpha1 + '_bar']
isoVolume1.ThresholdRange = [0.5, 1.2]

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=isoVolume1)
annotateTimeFilter1.Format = 'Time: {time:.2f}s'

# ----------------------------------------------------------------
# restore active source
#SetActiveSource(annotateTimeFilter1)

annotateTimeFilter1.UpdatePipeline()
Show(annotateTimeFilter1); Render()

# ---------------- Setup render view ----------------
view = CreateRenderView()
view.ViewSize = [1920, 1080]
view.Background = [1, 1, 1]   # white background

# ---------------- Show Uprime_prepare as reference (geometry only) ----------------
rep_geom = Show(pf_prepare, view)
#rep_geom.Representation = "Surface"
rep_geom.ColorArrayName = ['POINTS', '']   # no coloring (solid)
rep_geom.DiffuseColor = [0.8, 0.8, 0.8]  # light gray
rep_geom.EdgeColor = [0, 0, 0]
rep_geom.Representation = 'Feature Edges'   # enable surface edges
rep_geom.Opacity = 1.0
RenameSource("Uprime_prepare_geom", pf_prepare)

# trace defaults for the display properties.
annotateTimeFilter1Display = Show(annotateTimeFilter1, view, 'TextSourceRepresentation')
annotateTimeFilter1Display.WindowLocation = 'Any Location'
annotateTimeFilter1Display.Position = [0.43894507410636446, 0.6799171270718232]
annotateTimeFilter1Display.FontSize = 22

# ---------------- Show final field (annotateTimeFilter1) ----------------
rep = Show(isoVolume1, view)
rep.Representation = "Surface"

# --- Color by 'eps' ---
ColorBy(rep, ('POINTS', 'eps'))
lut = GetColorTransferFunction('eps')
# log scale (since eps spans decades)
lut.MapControlPointsToLogSpace()
lut.UseLogScale = 1

lut.RescaleTransferFunction(1e-5, 1)   # set min,max

# apply "jet" color scheme
ApplyPreset(lut, "jet", True)
# scalar bar formatting
bar = GetScalarBar(lut, view)
bar.Title = "$\\epsilon$"
bar.ComponentTitle = ""
bar.Visibility = 1
bar.LabelFormat = '%6.1e'   # 6.2f formatting
bar.TitleFontSize = 18
bar.LabelFontSize = 16
bar.TitleBold = 1
bar.LabelBold = 1

# custom tick labels
bar.UseCustomLabels = 1
bar.CustomLabels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# --- Axes grid (with custom titles) ---
view.AxesGrid = 'GridAxes3DActor'
view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle = "X (m)"
view.AxesGrid.YTitle = "Y"
view.AxesGrid.ZTitle = "Z (m)"
view.AxesGrid.XTitleFontSize = 18
view.AxesGrid.ZTitleFontSize = 18
view.AxesGrid.XTitleBold = 1
view.AxesGrid.ZTitleBold = 1

# Optional: reset camera
view.ResetCamera()
view.Update()

# ---------------- Save images per timestep ----------------
tk = GetTimeKeeper()
times = list(tk.TimestepValues)

outdir = "/project/smarras/am2455/wave_beach_profile/wang_kraus_scaled/results/eps_animation"
os.makedirs(outdir, exist_ok=True)

for t in times:
    UpdatePipeline(time=t, proxy=annotateTimeFilter1)
    fname = os.path.join(outdir, f"eps_{t:.2f}.png")
    SaveScreenshot(fname, view, ImageResolution=[1920,1080])
    print("saved", fname)

