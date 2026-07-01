"""
metasurface_mask.py
-------------------------------------------------------------------------------
Generate a metasurface pattern mask (.npy) for ARTEMIS macroscopic properties,
drawing the geometry directly from physical dimensions (in nm) -- no GDS needed.

The mask is consumed by
    MacroscopicProperties::InitializeMacroMultiFabFromNumpy
Hard requirements (ARTEMIS aborts / mis-places the pattern otherwise):
  * shape  = (nx, ny, 1)                     nx,ny MUST equal amr.n_cell x/y
  * dtype  = float64                         ARTEMIS only reads float64
  * layout = mask[i, j, 0], i=x cell, j=y cell, C-order (matches idx2d = i*ny+j)

Two modes (selected in the input deck, not here):
  * binary  (macroscopic.npy_variable_value = 0): put 1.0 where material is,
            0 elsewhere; the solver assigns the constant macroscopic.*_npy_value.
  * variable(macroscopic.npy_variable_value = 1): put the ACTUAL physical value
            the solver stores, and 0 where you want the background untouched:
                sigma    -> conductivity [S/m]
                epsilon  -> absolute permittivity eps_r * ep0   [F/m]
                mu       -> absolute permeability mu_r * mu0     [H/m]

Placement/thickness in z is done in the input deck:
    macroscopic.npy_k_index       = <first z cell>
    macroscopic.npy_k_thickness   = <number of z cells>   (thickness in z)
    macroscopic.npy_variable_value = 0 or 1
-------------------------------------------------------------------------------
"""

import numpy as np

NM = 1e-9  # nm -> m

# =============================================================================
# 1) DOMAIN  --  must match your ARTEMIS input deck EXACTLY
#    (geometry.prob_lo / prob_hi are in meters; amr.n_cell are the x/y cells)
#    2x2 ARRAY (fine mesh): 160 x 160 cells, dx = dy = 5 nm.
#    Domain 800 nm = 2 x 400 nm periods -> 2x2 pillars; periodic BC = infinite array.
#    5 nm cells -> 50 nm hole = 10 cells (smooth).
# =============================================================================
prob_lo = np.array([0.0, 0.0])                 # (x, y) lower corner [m]
prob_hi = np.array([800 * NM, 800 * NM])       # (x, y) upper corner [m]   <-- EDIT
n_cell  = np.array([160, 160])                 # (nx, ny)                  <-- EDIT (== amr.n_cell)

nx, ny = int(n_cell[0]), int(n_cell[1])
dx = (prob_hi - prob_lo) / n_cell              # cell size [m]
print(f"grid: {nx} x {ny} cells,  dx = {dx[0]/NM:.2f} nm, dy = {dx[1]/NM:.2f} nm")

# cell-CENTER coordinates [m]; indexing='ij' so X[i,j]=xc[i], Y[i,j]=yc[j]
xc = prob_lo[0] + (np.arange(nx) + 0.5) * dx[0]
yc = prob_lo[1] + (np.arange(ny) + 0.5) * dx[1]
X, Y = np.meshgrid(xc, yc, indexing="ij")

mask = np.zeros((nx, ny), dtype=np.float64)    # background = 0

# =============================================================================
# 2) PAINT HELPERS  --  all coordinates/sizes in METERS (use the NM factor)
# =============================================================================
def paint_rect(cx, cy, w, h, value):
    """Axis-aligned rectangle centered at (cx,cy), full width w, full height h."""
    sel = (np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= h / 2)
    mask[sel] = value

def paint_circle(cx, cy, r, value):
    sel = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    mask[sel] = value

def paint_ellipse(cx, cy, rx, ry, value):
    sel = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    mask[sel] = value

# =============================================================================
# 3) BUILD THE METASURFACE  --  periodic array of meta-atoms (EDIT this block)
#    Example: square lattice of circular pillars. All dims in nm.
# =============================================================================
material_value = 1.0          # 1.0 for binary mask (all pillars are SiN)
period    = 400 * NM          # lattice period P
p_diam    = 120 * NM          # pillar diameter
p_rad     = p_diam / 2
hole_diam = 50 * NM           # off-center air hole diameter
hole_rad  = hole_diam / 2
hole_off  = 20 * NM           # hole center shifted LEFT (-x) from pillar center (non-concentric)

# region to tile (default: whole domain). Restrict if the metasurface is a patch.
tile_lo = prob_lo.copy()
tile_hi = prob_hi.copy()

na = int((tile_hi[0] - tile_lo[0]) // period)
nb = int((tile_hi[1] - tile_lo[1]) // period)
for a in range(na):
    for b in range(nb):
        cx = tile_lo[0] + (a + 0.5) * period
        cy = tile_lo[1] + (b + 0.5) * period
        paint_circle(cx, cy, p_rad, material_value)          # SiN pillar body
        paint_circle(cx - hole_off, cy, hole_rad, 0.0)       # carve off-center air hole (left)

# --- resolution sanity check: the hole is the smallest feature -----------------
cells_per_hole = hole_diam / dx[0]
if cells_per_hole < 5:
    print(f"WARNING: the {hole_diam/NM:.0f} nm hole spans only {cells_per_hole:.1f} cells "
          f"-> coarse. Refine dx (shrink domain / more cells) for a smoother hole.")

# =============================================================================
# 4) SANITY PLOT + SAVE as (nx, ny, 1) float64
# =============================================================================
print(f"fill fraction = {np.count_nonzero(mask)/mask.size:.3f}, "
      f"min/max = {mask.min()}/{mask.max()}")

# optional preview -- skipped silently if matplotlib is unavailable / headless
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(mask.T, origin="lower",
               extent=[prob_lo[0]/NM, prob_hi[0]/NM, prob_lo[1]/NM, prob_hi[1]/NM])
    plt.xlabel("x [nm]"); plt.ylabel("y [nm]")
    plt.colorbar(label="mask value"); plt.title("metasurface mask")
    plt.tight_layout(); plt.savefig("metasurface_mask_preview.png", dpi=150)
    print("preview -> metasurface_mask_preview.png")
except Exception as e:
    print(f"(preview skipped: {e})")

mask_3d = mask[:, :, np.newaxis].astype(np.float64)   # (nx, ny, 1)
np.save("metasurface_mask.npy", mask_3d)
print(f"saved metasurface_mask.npy  shape={mask_3d.shape}  dtype={mask_3d.dtype}")
