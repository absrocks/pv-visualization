# ==== ParaView Macro: per-bin U_bar, k_avg, eps, alpha1_bar (reduced points) ====
from paraview.simple import *

# ---------------- user settings ----------------
array_U          = "U"        # 3-comp vector (required)
array_alpha1     = "alpha.water"   # scalar (required)  -> will output alpha1_bar
array_nuSgs      = "nut"    # scalar (required)  -> used only inside eps, not output
span_dir         = "y"        # 'x'|'y'|'z'  (average ALONG this axis; keep the other two)
use_cell_centers = False      # True if arrays are on Cell Data
merge_blocks     = True       # True if case is decomposed/multiblock (recommended!)
nu               = 1e-6       # molecular viscosity
bin_scale        = 1.0        # >1 coarsens bins (fewer bins -> faster Delaunay)
force_global     = True  # set True to gather to 1 rank before averaging
# ------------------------------------------------

src = GetActiveSource()
if src is None:
    raise RuntimeError("Select a source in the Pipeline first.")

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
    f'array_U      = "{array_U}"\n'
    f'array_alpha1 = "{array_alpha1}"\n'
    f'array_nuSgs  = "{array_nuSgs}"\n'
    f'span_dir     = "{span_dir}".lower()\n'
    f'bin_scale    = {bin_scale}\n'
)

SCRIPT1 += r"""
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
"""
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

SCRIPT2 += r"""
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
"""

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
