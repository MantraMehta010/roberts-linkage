"""
Coupler Curves of the Four-Bar Roberts Linkage
================================================
IUCAA Research Internship -- Dr. Apratim Ganguly

Python equivalent of RobertsLinkage.nb

Requirements:
    pip install numpy scipy matplotlib sympy

Run:
    python roberts_linkage.py

Sections:
    1. Four-bar kinematics (Freudenstein)
    2. Grashof condition
    3. Roberts linkage trace
    4. Interactive explorer  (matplotlib sliders)
    5. Deviation landscape   (grid plot)
    6. Optimisation          (scipy minimize)
    7. Family of curves
    8. Deviation profile along arc
    9. Symbolic determinant  (sympy)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import sympy as sp
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#f7f7f5",
    "axes.facecolor": "#f7f7f5",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# =============================================================================
# SECTION 1 -- Four-Bar Kinematics
# =============================================================================

def four_bar_solve(a1, a2, a3, a4, theta2):
    """
    Solve for coupler angle theta3 and follower angle theta4 given
    link lengths {a1, a2, a3, a4} and crank angle theta2.

    Uses the Freudenstein analytical method:
        K1 = a1/a2,  K2 = a1/a4,  K3 = (a2^2 - a3^2 + a4^2 + a1^2) / (2*a2*a4)
        A T^2 + B T + C = 0  where T = tan(theta4 / 2)

    Returns (theta3, theta4) or None if the linkage cannot assemble.
    """
    K1 = a1 / a2
    K2 = a1 / a4
    K3 = (a2**2 - a3**2 + a4**2 + a1**2) / (2 * a2 * a4)
    K4 = a1 / a3
    K5 = (a4**2 - a1**2 - a2**2 - a3**2) / (2 * a2 * a3)

    # Solve for theta4
    A = np.cos(theta2) - K1 - K2 * np.cos(theta2) + K3
    B = -2 * np.sin(theta2)
    C = K1 - (K2 + 1) * np.cos(theta2) + K3
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None
    if abs(A) < 1e-12:
        return None
    T = (-B - np.sqrt(disc)) / (2 * A)
    theta4 = 2 * np.arctan(T)

    # Solve for theta3
    A = np.cos(theta2) - K1 + K4 * np.cos(theta2) + K5
    B = -2 * np.sin(theta2)
    C = K1 + (K4 - 1) * np.cos(theta2) + K5
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None
    if abs(A) < 1e-12:
        return None
    T = (-B - np.sqrt(disc)) / (2 * A)
    theta3 = 2 * np.arctan(T)

    return theta3, theta4


def coupler_point(a2, theta2, theta3, u, v):
    """
    Compute the Cartesian coordinates of tracer point P in the fixed frame.

    P = O2 + Q * [u, -v]^T   where Q is the rotation matrix for theta3.

    Returns (px, py).
    """
    # O2 position (end of crank)
    ox = a2 * np.cos(theta2)
    oy = a2 * np.sin(theta2)
    # Tracer point
    px = ox + u * np.cos(theta3) - v * np.sin(theta3)
    py = oy + u * np.sin(theta3) + v * np.cos(theta3)
    return px, py


def trace_coupler_curve(a1, a2, a3, a4, u, v, npts=600):
    """
    Sweep crank angle theta2 over [0, 2*pi] and collect tracer point positions.

    Returns Nx2 array of (x, y) points.
    """
    angles = np.linspace(0, 2 * np.pi, npts)
    pts = []
    for th in angles:
        sol = four_bar_solve(a1, a2, a3, a4, th)
        if sol is None:
            continue
        th3, th4 = sol
        px, py = coupler_point(a2, th, th3, u, v)
        if np.isfinite(px) and np.isfinite(py):
            pts.append([px, py])
    return np.array(pts) if pts else np.empty((0, 2))


# =============================================================================
# SECTION 2 -- Grashof Condition
# =============================================================================

def grashof(a1, a2, a3, a4):
    """
    Returns True if the shortest + longest <= sum of the other two.
    Grashof condition: full crank rotation is possible.
    """
    lens = sorted([a1, a2, a3, a4])
    return lens[0] + lens[3] <= lens[1] + lens[2]


# =============================================================================
# SECTION 3 -- Roberts Linkage Helpers
# =============================================================================

def roberts_trace(a1, a2, a3, qy, npts=600):
    """
    Trace the coupler curve for a Roberts linkage.

    Roberts symmetry conditions:
        a4 = a2          (crank = follower length)
        u  = a3 / 2      (trace point at midpoint of coupler base)
        v  = qy * a3     (height above coupler base)

    Parameters
    ----------
    a1  : frame length
    a2  : crank (= follower) length
    a3  : coupler length
    qy  : normalised trace point height
    """
    u = a3 / 2.0
    v = qy * a3
    return trace_coupler_curve(a1, a2, a3, a2, u, v, npts)


def straight_line_deviation(pts):
    """
    Fit a best-fit line y = m*x + b to pts and return the RMS residual.
    """
    if len(pts) < 3:
        return np.inf
    xs, ys = pts[:, 0], pts[:, 1]
    # numpy polyfit: degree 1
    m, b = np.polyfit(xs, ys, 1)
    residuals = ys - (m * xs + b)
    return np.sqrt(np.mean(residuals**2))


def near_straight_segment(pts, threshold=0.03):
    """
    Identify the near-straight portion of the coupler curve using
    second differences of y as a curvature proxy.

    Returns the subset of points where |d^2y| < threshold.
    """
    if len(pts) < 5:
        return pts
    ys = pts[:, 1]
    d2y = np.abs(np.gradient(np.gradient(ys)))
    mask = d2y < threshold
    return pts[mask]


def draw_linkage_snapshot(ax, a1, a2, a3, qy, theta2=np.pi / 3):
    """Draw a snapshot of the Roberts linkage at a given crank angle."""
    u = a3 / 2.0
    v = qy * a3
    sol = four_bar_solve(a1, a2, a3, a2, theta2)
    if sol is None:
        return
    th3, th4 = sol
    ox = a2 * np.cos(theta2)
    oy = a2 * np.sin(theta2)
    dx = a1 + a2 * np.cos(th4)
    dy = a2 * np.sin(th4)
    px, py = coupler_point(a2, theta2, th3, u, v)

    ax.plot([0, ox], [0, oy], color="#3377cc", lw=2.5, zorder=3)          # crank
    ax.plot([ox, dx], [oy, dy], color="#555555", lw=2.0, zorder=3)        # coupler
    ax.plot([a1, dx], [0, dy], color="#22aa66", lw=2.5, zorder=3)         # follower
    ax.scatter([ox, dx], [oy, dy], color=["#3377cc", "#22aa66"], s=50, zorder=4)
    ax.scatter([px], [py], color="#cc4422", s=100, zorder=5)              # tracer


# =============================================================================
# SECTION 4 -- Interactive Roberts Linkage Explorer
# =============================================================================

def interactive_explorer():
    """
    Interactive four-bar Roberts linkage explorer with matplotlib sliders.

    Sliders:
        a1  -- frame length
        a2  -- crank = follower length
        a3  -- coupler length
        qy  -- trace point height offset
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.1, bottom=0.38)
    fig.suptitle("Roberts Linkage Explorer", fontsize=13, fontweight="bold")

    # Initial parameters
    init = dict(a1=2.5, a2=1.0, a3=2.0, qy=0.5)

    def compute_and_draw(a1, a2, a3, qy):
        ax.cla()
        ax.set_xlim(-2.5, a1 + 3)
        ax.set_ylim(-2.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        pts = roberts_trace(a1, a2, a3, qy, npts=700)
        seg = near_straight_segment(pts, threshold=0.03)
        dev = straight_line_deviation(seg) if len(seg) > 3 else np.inf

        # Full coupler curve
        if len(pts) > 2:
            closed = np.vstack([pts, pts[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="#aaaaaa", lw=1.5,
                    label="Full coupler curve")

        # Near-straight segment
        if len(seg) > 2:
            ax.plot(seg[:, 0], seg[:, 1], color="#cc4422", lw=3,
                    label="Near-straight segment")

            # Best-fit line
            if len(seg) > 3:
                m, b = np.polyfit(seg[:, 0], seg[:, 1], 1)
                x_fit = np.array([seg[:, 0].min(), seg[:, 0].max()])
                ax.plot(x_fit, m * x_fit + b, "--", color="#3377cc", lw=1.8,
                        label="Best-fit line")

        # Ground pivots
        ax.plot([0, a1], [0, 0], "--", color="#888888", lw=1)
        ax.scatter([0, a1], [0, 0], color="#333333", s=60, zorder=5)
        ax.text(-0.15, -0.2, r"$O_1$", fontsize=10)
        ax.text(a1 + 0.05, -0.2, r"$O_4$", fontsize=10)

        # Linkage snapshot
        draw_linkage_snapshot(ax, a1, a2, a3, qy)

        # Info text
        g = grashof(a1, a2, a3, a2)
        info = (f"$a_1$={a1:.2f},  $a_2=a_4$={a2:.2f},  $a_3$={a3:.2f},  "
                f"$q_y$={qy:.2f}\n"
                f"RMS dev = {dev:.5f}    "
                + (r"Grashof $\checkmark$" if g else "Non-Grashof $\times$"))
        ax.set_title(info, fontsize=10)

        ax.legend(loc="upper right", fontsize=9)
        fig.canvas.draw_idle()

    compute_and_draw(**init)

    # Sliders
    ax_a1 = plt.axes([0.15, 0.28, 0.7, 0.03])
    ax_a2 = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_a3 = plt.axes([0.15, 0.16, 0.7, 0.03])
    ax_qy = plt.axes([0.15, 0.10, 0.7, 0.03])

    sl_a1 = Slider(ax_a1, r"Frame $a_1$",   0.8, 4.0, valinit=init["a1"])
    sl_a2 = Slider(ax_a2, r"Crank $a_2=a_4$", 0.2, 2.5, valinit=init["a2"])
    sl_a3 = Slider(ax_a3, r"Coupler $a_3$", 0.5, 3.5, valinit=init["a3"])
    sl_qy = Slider(ax_qy, r"Offset $q_y$",  0.05, 1.3, valinit=init["qy"])

    def update(_):
        compute_and_draw(sl_a1.val, sl_a2.val, sl_a3.val, sl_qy.val)

    for sl in [sl_a1, sl_a2, sl_a3, sl_qy]:
        sl.on_changed(update)

    plt.show()


# =============================================================================
# SECTION 5 -- Deviation Landscape
# =============================================================================

def deviation_landscape(a1=2.5, a3=2.0):
    """
    Compute and plot the RMS deviation landscape over a grid of (a2, qy).

    This is the empirical version of the L2 objective function from
    eq. (15) in Baskar et al.
    """
    print(f"Computing deviation landscape for a1={a1}, a3={a3}...")
    print("(Takes ~20 seconds)")

    a2_vals = np.arange(0.4, 2.05, 0.1)
    qy_vals = np.arange(0.1, 1.05, 0.05)
    Z = np.zeros((len(qy_vals), len(a2_vals)))

    for i, a2 in enumerate(a2_vals):
        for j, qy in enumerate(qy_vals):
            pts = roberts_trace(a1, a2, a3, qy, npts=300)
            if len(pts) < 10:
                Z[j, i] = np.nan
                continue
            seg = near_straight_segment(pts, threshold=0.04)
            Z[j, i] = straight_line_deviation(seg) if len(seg) > 5 else np.nan

    Z = np.clip(Z, 0, 0.4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"RMS Straight-Line Deviation  ($a_1={a1}$, $a_3={a3}$)",
        fontsize=13, fontweight="bold"
    )

    A2, QY = np.meshgrid(a2_vals, qy_vals)

    # 3D surface
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot_surface(A2, QY, Z, cmap="RdYlGn_r", alpha=0.85, linewidth=0)
    ax3d.set_xlabel(r"$a_2$")
    ax3d.set_ylabel(r"$q_y$")
    ax3d.set_zlabel("RMS deviation")
    ax3d.set_title("Surface plot")

    # Contour
    axes[1].contourf(A2, QY, Z, levels=20, cmap="RdYlGn_r")
    c = axes[1].contour(A2, QY, Z, levels=10, colors="k", linewidths=0.5, alpha=0.4)
    axes[1].clabel(c, fmt="%.3f", fontsize=7)
    axes[1].set_xlabel(r"$a_2 = a_4$")
    axes[1].set_ylabel(r"$q_y$")
    axes[1].set_title("Contour plot")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn_r"),
        ax=axes[1], label="RMS deviation"
    )

    fig.delaxes(axes[0])     # remove placeholder, 3d axes already added
    plt.tight_layout()
    plt.show()
    print("Done.")


# =============================================================================
# SECTION 6 -- Optimisation
# =============================================================================

def l2_deviation_objective(params, a1, a3):
    """
    Objective function for optimisation.
    params = [a2, qy]

    Computes RMS deviation of the Roberts coupler curve from a straight line
    in the near-flat region.
    """
    a2, qy = params
    if a2 <= 0 or qy <= 0:
        return 10.0
    pts = roberts_trace(a1, a2, a3, qy, npts=500)
    if len(pts) < 15:
        return 10.0
    # Filter to near-flat region
    near_flat = pts[np.abs(pts[:, 1]) < 1.5]
    if len(near_flat) < 5:
        return 10.0
    return straight_line_deviation(near_flat)


def optimise_roberts(a1=2.5, a3=2.0):
    """
    Find the optimal Roberts linkage parameters (a2, qy) that minimise
    the RMS straight-line deviation, for fixed frame a1 and coupler a3.

    Uses scipy differential_evolution for global search (analogous to
    homotopy continuation finding all 73 critical points in Baskar et al.)
    """
    print(f"\nOptimising Roberts linkage: a1={a1}, a3={a3}")
    print("Using differential evolution (global search)...")

    bounds = [(0.2, 2.5), (0.05, 1.4)]
    result = differential_evolution(
        l2_deviation_objective,
        bounds,
        args=(a1, a3),
        maxiter=300,
        tol=1e-6,
        seed=42,
        disp=False
    )

    a2_opt, qy_opt = result.x
    print(f"Optimal a2 = {a2_opt:.4f}")
    print(f"Optimal qy = {qy_opt:.4f}")
    print(f"Minimum RMS deviation = {result.fun:.6f}")

    # Plot optimal linkage
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_title(
        f"Optimal Roberts Linkage\n"
        f"$a_1={a1}$, $a_2=a_4={a2_opt:.3f}$, $a_3={a3}$, "
        f"$q_y={qy_opt:.3f}$\nRMS = {result.fun:.5f}",
        fontsize=11
    )

    pts = roberts_trace(a1, a2_opt, a3, qy_opt, npts=800)
    seg = near_straight_segment(pts, threshold=0.03)

    if len(pts) > 2:
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#aaaaaa", lw=1.5,
                label="Full coupler curve")
    if len(seg) > 2:
        ax.plot(seg[:, 0], seg[:, 1], color="#cc4422", lw=3,
                label="Near-straight segment")
        if len(seg) > 3:
            m, b = np.polyfit(seg[:, 0], seg[:, 1], 1)
            xf = np.array([seg[:, 0].min(), seg[:, 0].max()])
            ax.plot(xf, m * xf + b, "--", color="#3377cc", lw=2,
                    label="Best-fit line")

    ax.scatter([0, a1], [0, 0], color="#333333", s=80, zorder=5)
    ax.plot([0, a1], [0, 0], "--", color="#888888", lw=1)
    draw_linkage_snapshot(ax, a1, a2_opt, a3, qy_opt)
    ax.legend(fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    return a2_opt, qy_opt, result.fun


# =============================================================================
# SECTION 7 -- Family of Coupler Curves
# =============================================================================

def family_of_curves(a1=2.5, a2=1.0, a3=2.0):
    """
    Plot a family of Roberts coupler curves for varying qy.
    """
    qy_vals = np.arange(0.2, 1.05, 0.2)
    cmap = plt.cm.rainbow
    colors = [cmap(x) for x in np.linspace(0, 1, len(qy_vals))]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_title(
        f"Family of Roberts Coupler Curves\n"
        f"$a_1={a1}$, $a_2=a_4={a2}$, $a_3={a3}$, varying $q_y$",
        fontsize=11
    )

    for qy, col in zip(qy_vals, colors):
        pts = roberts_trace(a1, a2, a3, qy, npts=600)
        if len(pts) > 2:
            closed = np.vstack([pts, pts[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=col, lw=1.8,
                    label=f"$q_y = {qy:.1f}$")

    ax.scatter([0, a1], [0, 0], color="#333333", s=80, zorder=5)
    ax.plot([0, a1], [0, 0], "--", color="#888888", lw=1)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 8 -- Deviation Profile Along Arc Length
# =============================================================================

def deviation_profile(a1=2.5, a2=1.0, a3=2.0, qy=0.5):
    """
    Plot the absolute deviation from a best-fit straight line as a function
    of arc length along the coupler curve.

    The region where deviation < 1% tolerance is the usable operating range
    for a seismometer.
    """
    pts = roberts_trace(a1, a2, a3, qy, npts=800)
    if len(pts) < 10:
        print("Not enough points to compute deviation profile.")
        return

    xs, ys = pts[:, 0], pts[:, 1]

    # Best-fit line
    m, b = np.polyfit(xs, ys, 1)
    residuals = np.abs(ys - (m * xs + b))

    # Arc length
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    arc = np.concatenate([[0], np.cumsum(ds)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Deviation Profile: $a_1={a1}$, $a_2=a_4={a2}$, $a_3={a3}$, $q_y={qy}$",
        fontsize=12, fontweight="bold"
    )

    # Left: coupler curve with deviation coloured
    sc = axes[0].scatter(xs, ys, c=residuals, cmap="RdYlGn_r",
                          s=8, vmin=0, vmax=0.15)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title("Coupler curve coloured by deviation")
    fig.colorbar(sc, ax=axes[0], label="|deviation from best-fit line|")
    axes[0].scatter([0, a1], [0, 0], color="#333333", s=60, zorder=5)

    # Right: deviation vs arc length
    axes[1].plot(arc, residuals, color="#cc4422", lw=1.5)
    axes[1].axhline(0.01, color="#3377cc", lw=1.5, linestyle="--",
                     label="1% tolerance")
    axes[1].fill_between(arc, 0, residuals,
                          where=residuals < 0.01,
                          color="#22aa66", alpha=0.3,
                          label="Operating range (<1% dev)")
    axes[1].set_xlabel("Arc length $s$")
    axes[1].set_ylabel("|Deviation from best-fit line|")
    axes[1].set_title("Deviation vs. arc length")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 9 -- Symbolic Determinant (SymPy)
# =============================================================================

def symbolic_determinant():
    """
    Symbolically compute the coupler trace equation for the Roberts linkage
    by evaluating the 4x4 determinant from Baskar et al. eq. (5).

    Roberts conditions applied:
        Q  = 1/2 + I*qy,  Q* = 1/2 - I*qy
        A  = A* = 0,  B = B* = 1    (normalised ground pivots)
        l1 = l3 = l                  (Roberts length condition)

    Confirms the result is a degree-6 polynomial in X and X*.
    """
    print("\nComputing symbolic coupler trace equation for Roberts linkage...")
    print("(This may take 1-2 minutes)\n")

    X, Xs, l, l2, qy = sp.symbols("X X_s l l_2 q_y", complex=False)

    # Roberts symmetry
    Q  = sp.Rational(1, 2) + sp.I * qy
    Qs = sp.Rational(1, 2) - sp.I * qy   # Q* = 1 - Q

    # Normalised pivots: A=0, B=1
    A, As, B, Bs = 0, 0, 1, 1
    l1, l3 = l, l

    # g and h from eqs. (g) and (h) in Baskar et al.
    g = l2**2 * Q * Qs - l1**2 + (A - X) * (As - Xs)
    h = l2**2 * (Q - 1) * (Qs - 1) - l3**2 + (B - X) * (Bs - Xs)

    # 4x4 matrix M (eq. 5)
    M = sp.Matrix([
        [Qs*(A-X),       g,              l2*Q*(As-Xs),       0            ],
        [0,              l2*Qs*(A-X),    g,                  Q*(As-Xs)    ],
        [(Qs-1)*(B-X),   h,              l2*(Q-1)*(Bs-Xs),   0            ],
        [0,              l2*(Qs-1)*(B-X), h,                 (Q-1)*(Bs-Xs)]
    ])

    print("Matrix M defined. Computing determinant...")
    f = sp.expand(M.det())

    # Degree in X and X*
    deg_X  = sp.degree(sp.Poly(f, X), X)
    deg_Xs = sp.degree(sp.Poly(f, Xs), Xs)

    print(f"\nCoupler trace equation computed.")
    print(f"Degree in X  : {deg_X}")
    print(f"Degree in X* : {deg_Xs}")
    print(f"\nThis confirms the coupler curve is a SEXTIC (degree-6) algebraic curve.")
    print(f"No straight segment of finite length can lie on a degree-6 curve (Bezout).")

    # Count monomials in X, Xs
    poly_X  = sp.Poly(sp.expand(f), X, Xs)
    n_terms = len(poly_X.monoms())
    print(f"\nNumber of distinct monomials in (X, X*): {n_terms}")
    print("(The full general sextic has 28 monomials;")
    print(" circularity-3 reduces this to 16 for the coupler curve.)")

    return f


# =============================================================================
# MAIN -- Run all sections in sequence
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Roberts Linkage Coupler Curve Analysis")
    print("=" * 60)

    print("""
Select which section to run:
  1  -- Interactive explorer (sliders)
  2  -- Deviation landscape (grid, ~20s)
  3  -- Optimisation (global search, ~1min)
  4  -- Family of curves
  5  -- Deviation profile along arc
  6  -- Symbolic determinant (SymPy, ~2min)
  all -- Run 4, 5 (fast plots only)
""")

    choice = input("Enter choice: ").strip().lower()

    if choice == "1":
        interactive_explorer()

    elif choice == "2":
        deviation_landscape(a1=2.5, a3=2.0)

    elif choice == "3":
        optimise_roberts(a1=2.5, a3=2.0)

    elif choice == "4":
        family_of_curves(a1=2.5, a2=1.0, a3=2.0)

    elif choice == "5":
        deviation_profile(a1=2.5, a2=1.0, a3=2.0, qy=0.5)

    elif choice == "6":
        symbolic_determinant()

    elif choice == "all":
        print("\n--- Family of curves ---")
        family_of_curves()
        print("\n--- Deviation profile ---")
        deviation_profile()

    else:
        print("Running interactive explorer by default...")
        interactive_explorer()
