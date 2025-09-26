#!/usr/bin/env pvpython
import os, sys
from pathlib import Path
import numpy as np
from paraview.simple import *

# -------------------------------
# Configurations
# -------------------------------
BATCH_CONFIG = {
    'array_U': 'U',
    'array_alpha1': 'alpha.water',
    'array_nuSgs': 'nut',
    'merge_blocks': True,
    'nu': 1e-6,
    'bin_scale': 1.0,
    'force_global': True,
    'analysis_mode': 'clipping',   # or 'averaging'
    'averaging': {'axis': 'Y'},
    'clipping': {'axis': 'X', 'Xmin': 20, 'Xmax': 28},
    'visualization': {
        'image_size': [1920, 1080],
        'color_map': 'jet',
    }
}

outdir = "./results/eps_animation"
os.makedirs(outdir, exist_ok=True)

# -------------------------------
# Load .foam file
# -------------------------------
foam_file = "./case.foam"   # << change path
print("Loading data...")
reader = OpenDataFile(foam_file)
reader.UpdatePipeline()

info = reader.GetDataInformation()
if info.GetNumberOfPoints() == 0:
    raise RuntimeError("ERROR: No data points found")

arrays = [reader.GetPointDataInformation().GetArray(i).GetName()
          for i in range(reader.GetPointDataInformation().GetNumberOfArrays())]
print("Available arrays:", ", ".join(arrays))

cur = reader

# -------------------------------
# Merge & Redistribute
# -------------------------------
if BATCH_CONFIG['merge_blocks']:
    cur = MergeBlocks(Input=cur); cur.UpdatePipeline()

if BATCH_CONFIG['force_global']:
    d3 = RedistributeDataSet(Input=cur)
    d3.TargetPartitions = 1
    d3.PreservePartitions = 0
    d3.UpdatePipeline()
    cur = d3

# -------------------------------
# Apply Clipping from dictionary
# -------------------------------
limits = BATCH_CONFIG['clipping']
for i, axis_name in enumerate(['X', 'Y', 'Z']):
    for bound_type, j in zip(['min', 'max'], [0, 1]):
        key = f"{axis_name}{bound_type}"   # e.g. "Xmin", "Xmax"
        bound = limits.get(key)
        if bound is not None:
            clip = Clip(Input=cur)
            clip.ClipType = 'Plane'
            normal = [0, 0, 0]; normal[i] = 1
            clip.ClipType.Normal = normal
            clip.ClipType.Origin = [0, 0, 0]; clip.ClipType.Origin[i] = bound
            clip.Invert = 0 if bound_type == 'min' else 1
            cur = clip
            print(f"Applied {key} = {bound}")

# -------------------------------
# Shared header for PF scripts
# -------------------------------
common_header = f"""
import builtins, numpy as np
from vtkmodules.util import numpy_support as ns
import vtk
array_U      = "{BATCH_CONFIG['array_U']}"
array_alpha1 = "{BATCH_CONFIG['array_alpha1']}"
array_nuSgs  = "{BATCH_CONFIG['array_nuSgs']}"
span_dir     = "{BATCH_CONFIG['averaging']['axis']}".lower()
bin_scale    = {BATCH_CONFIG['bin_scale']}
nu           = {BATCH_CONFIG['nu']}
"""

# -------------------------------
# PF #1: Uprime
# -------------------------------
pf_prepare = ProgrammableFilter(Input=cur)
pf_prepare.Script = common_header + """
# (Uprime calc unchanged from your version)
inp = self.GetInput()
pts_vtk = inp.GetPoints()
if pts_vtk is None: raise RuntimeError("No points on input.")
pts = ns.vtk_to_numpy(pts_vtk.GetData()); N = pts.shape[0]
pd = inp.GetPointData()
U     = pd.GetArray(array_U)
U     = ns.vtk_to_numpy(U).reshape((N,3))
out = self.GetOutput()
out.ShallowCopy(inp)
Uprime = U - U.mean(axis=0)
Up_vtk   = ns.numpy_to_vtk(Uprime, deep=1); Up_vtk.SetName(array_U + "_prime")
out.GetPointData().AddArray(Up_vtk)
"""
RenameSource("Uprime_prepare", pf_prepare)

# -------------------------------
# Gradient(U')
# -------------------------------
# ---------- Add Ghost Cells (optional) ----------
if BATCH_CONFIG.get("force_global", False):
    try:
        ghost = GhostCells(Input=pf_prepare)
        ghost.MinimumNumberOfGhostLevels = 1   # try 1 first; 2 if gradients still noisy
        ghost.UpdatePipeline()
        grad_input = ghost
        RenameSource("Ghosted_Uprime", ghost)
        Show(ghost); Render()
        print("[ghost] Added 1 layer of ghost cells for GradUprime")
    except Exception as e:
        print(f"[warn] GhostCells not available ({e}); using pf_prepare directly")
        grad_input = pf_prepare
else:
    grad_input = pf_prepare

# ---------- Gradient(U') using the 'Gradient' filter ----------
gradient = Gradient(registrationName='GradUprime', Input=grad_input)
gradient.ResultArrayName = "GradUprime"
gradient.FasterApproximation = 0

# Primary property on this proxy is 'ScalarArray' (works for vectors too)
try:
    gradient.ScalarArray = ['POINTS', BATCH_CONFIG['array_U'] + "_prime"]
except AttributeError:
    # Fallback for older ParaView wrappers
    from paraview import vtk
    gradient.SetInputArrayToProcess(
        0,  # idx
        0,  # port
        0,  # connection
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        BATCH_CONFIG['array_U'] + "_prime"
    )

gradient.UpdatePipeline()
RenameSource("GradUprime", gradient)

# -------------------------------
# PF #2: averaging bins (kept intact)
# -------------------------------
pf_bins = ProgrammableFilter(Input=gradient)
pf_bins.OutputDataSetType = 'vtkPolyData'
pf_bins.Script = common_header + """
def tensor_from_vtk9(T):
    # [Gxx,Gxy,Gxz, Gyx,Gyy,Gyz, Gzx,Gzy,Gzz]
    return np.array([[T[0], T[1], T[2]],
                     [T[3], T[4], T[5]],
                     [T[6], T[7], T[8]]], dtype=float)
inp = self.GetInput()
Uprime = ns.vtk_to_numpy(pd.GetArray(array_U + "_prime")).reshape((N,3))
alpha = pd.GetArray(array_alpha1)
nusgs = pd.GetArray(array_nuSgs)
out = self.GetOutput()
out.ShallowCopy(inp)
Grad   = ns.vtk_to_numpy(pd.GetArray("GradUprime")).reshape((N,9)) 
"""
RenameSource("Bins", pf_bins)

# -------------------------------
# Visualization setup
# -------------------------------
view = CreateRenderView()
view.ViewSize = BATCH_CONFIG['visualization']['image_size']
view.Background = [1, 1, 1]

rep = Show(pf_bins, view)
ColorBy(rep, ('POINTS', 'eps'))
lut = GetColorTransferFunction('eps')
ApplyPreset(lut, BATCH_CONFIG['visualization']['color_map'], True)
lut.MapControlPointsToLogSpace(); lut.UseLogScale = 1
lut.RescaleTransferFunction(1e-5, 1)

bar = GetScalarBar(lut, view)
bar.Title = "$\\epsilon$"; bar.LabelFormat = '%6.1e'
bar.TitleFontSize = 18; bar.LabelFontSize = 16
bar.TitleBold = 1; bar.LabelBold = 1

# -------------------------------
# Save screenshots for each timestep
# -------------------------------
tk = GetTimeKeeper()
times = list(tk.TimestepValues)
for t in times:
    UpdatePipeline(time=t, proxy=pf_bins)
    fname = os.path.join(outdir, f"eps_{t:.2f}.png")
    SaveScreenshot(fname, view, ImageResolution=view.ViewSize)
    print("saved", fname)
