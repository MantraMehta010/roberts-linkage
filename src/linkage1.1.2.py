import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#f7f7f5",
    "axes.facecolor": "#f7f7f5",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# SECTION 1 -- Geometry (rest-position coordinates)

def geometry_defaults():
    """
    Rest-position coordinates matching the actual Linkage build:
        A = (-265, 0), B = (265, 0)       -- fixed pivots
        C = (-132.5, -254.96)             -- base point, left
        D = ( 132.5, -254.96)             -- base point, right
        P = (0, 0)                        -- apex of triangle C-P-D
        L = (0, -29.2)                    -- apex of triangle C-L-D
                                              (a SEPARATE triangle, same base)
    """
    A = np.array([-265.0, 0.0])
    B = np.array([265.0, 0.0])
    C = np.array([-132.5, -254.96])
    D = np.array([132.5, -254.96])
    P = np.array([0.0, 0.0])
    L = np.array([0.0, -29.2])
    return A, B, C, D, P, L


def geometry_at_base_depth(W, q, L0, isosceles="apex_at_C"):
    """
    Build the rest geometry with A, B, P held at the SAME construction as
    geometry_defaults() (P pinned to the A-B line, apex_at_C congruence),
    but with the base depth q (of C, D) and the load point L0 = (0, depth)
    as free inputs.

    Under the "apex_at_C" congruence condition (AC = CP; CD = AP), with P
    pinned at the origin, the x-position of C, D is c = W/2 IDENTICALLY,
    for any q -- this was verified symbolically in an earlier version of
    this module. That closed form is used directly here (no per-call
    symbolic solve needed), which makes this function fast enough to call
    in a search loop.

    L0 must be a length-2 array/tuple; only its y-component (depth) is
    used here -- L is always placed on the centerline (x=0), matching the
    symmetric construction throughout this module.
    """
    if isosceles != "apex_at_C":
        raise ValueError("Only 'apex_at_C' is implemented in this closed form.")
    c_val = W / 2.0
    A = np.array([-W, 0.0])
    B = np.array([W, 0.0])
    C = np.array([-c_val, -q])
    D = np.array([c_val, -q])
    P = np.array([0.0, 0.0])
    L = np.array([0.0, float(np.asarray(L0)[1])])
    return A, B, C, D, P, L


# =============================================================================
# SECTION 2 -- Forward Kinematics (two independent base cranks,
#              P and L each an independent apex on base C-D)
# =============================================================================

class RobertsLinkageCLD:
    """
    Encapsulates one Roberts Linkage instance and provides forward
    kinematics for the corrected topology: A and B are each independently
    driven by the SAME drive angle theta (matching the file's rpm=+15 /
    rpm=-15 setup -- see module docstring), and P, L are each found as an
    independent circle-circle intersection on the resulting base C, D.
    """

    def __init__(self, A, B, C0, D0, P0, L0):
        self.A, self.B = A, B
        self.C0, self.D0, self.P0, self.L0 = C0, D0, P0, L0

        self.AC = np.linalg.norm(C0 - A)
        self.BD = np.linalg.norm(D0 - B)
        self.CP = np.linalg.norm(P0 - C0)
        self.DP = np.linalg.norm(P0 - D0)
        self.CL = np.linalg.norm(L0 - C0)
        self.DL = np.linalg.norm(L0 - D0)

        self.phi0_A = self._angle_from_vertical(C0 - A)
        self.phi0_B = self._angle_from_vertical(D0 - B)

    @staticmethod
    def _angle_from_vertical(vec):
        down = np.array([0.0, -1.0])
        cosang = np.dot(vec, down) / np.linalg.norm(vec)
        ang = np.arccos(np.clip(cosang, -1, 1))
        sign = np.sign(vec[0]) if vec[0] != 0 else 1
        return sign * ang

    def _get_C(self, theta):
        ang = self.phi0_A + theta
        return self.A + self.AC * np.array([np.sin(ang), -np.cos(ang)])

    def _get_D(self, theta):
        # SAME theta added to phi0_B (which is itself the mirror-image
        # angle of phi0_A) -- reproduces the file's rpm=+15/-15 pair.
        ang = self.phi0_B + theta
        return self.B + self.BD * np.array([np.sin(ang), -np.cos(ang)])

    @staticmethod
    def _circle_intersect(C, D, rC, rD, ref):
        """Intersection of circle(C, rC) and circle(D, rD), returning
        whichever of the two solutions is nearer to `ref` (used to pick
        the branch continuous with the rest configuration)."""
        d = np.linalg.norm(D - C)
        if d > rC + rD or d < abs(rC - rD) or d == 0:
            return None
        a = (rC**2 - rD**2 + d**2) / (2 * d)
        h2 = rC**2 - a**2
        if h2 < 0:
            return None
        h = np.sqrt(h2)
        mid = C + a * (D - C) / d
        perp = np.array([-(D - C)[1], (D - C)[0]]) / d
        X1, X2 = mid + h * perp, mid - h * perp
        return X1 if np.linalg.norm(X1 - ref) < np.linalg.norm(X2 - ref) else X2

    def configuration(self, theta):
        """
        Full forward kinematics at drive angle theta (radians).
        Returns (C, D, P, L). P and L are each computed independently
        from C, D -- NOT from each other.
        """
        C = self._get_C(theta)
        D = self._get_D(theta)
        P = self._circle_intersect(C, D, self.CP, self.DP, self.P0)
        L = self._circle_intersect(C, D, self.CL, self.DL, self.L0)
        return C, D, P, L


# =============================================================================
# SECTION 3 -- Trace Generation
# =============================================================================

def trace_point(linkage, which, theta_deg_range=5.0, npts=201):
    """
    Sweep theta over [-theta_deg_range, +theta_deg_range] and collect the
    traced (x, y) positions of 'P' or 'L'.
    """
    idx = {"C": 0, "D": 1, "P": 2, "L": 3}[which]
    thetas = np.radians(np.linspace(-theta_deg_range, theta_deg_range, npts))
    pts = []
    for t in thetas:
        cfg = linkage.configuration(t)
        p = cfg[idx]
        if p is not None:
            pts.append(p)
    return np.array(pts)


# =============================================================================
# SECTION 4 -- Flatness / Stability Diagnostics
# =============================================================================

def flatness_rms(pts):
    if len(pts) < 3:
        return np.inf
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = pts[:, 1] - (m * pts[:, 0] + c)
    return np.sqrt(np.mean(resid**2))


def curvature_sign(pts):
    if len(pts) < 3:
        return np.nan
    x, y = pts[:, 0], pts[:, 1]
    xc = x - x.mean()
    a, b, c = np.polyfit(xc, y, 2)
    return a


def stability_label(a):
    if np.isnan(a):
        return "undetermined"
    return "STABLE (convex)" if a > 0 else "UNSTABLE (concave)"


def curvature_sign_local(linkage, which="L", theta_deg_range=0.3, npts=201):
    """
    Local (small-angle) curvature estimate, intended to approximate the
    TRUE derivative-based curvature at theta=0 rather than a curvature
    averaged over some arbitrary window.

    IMPORTANT, found by direct testing: curvature_sign's value is NOT
    constant as theta_deg_range grows -- for these traces, larger windows
    pick up real higher-order (quartic etc.) shape, not just noise, and
    the fitted quadratic coefficient drifts smoothly and substantially as
    a result (e.g. at one geometry tested here, going from
    theta_deg_range=0.3 to 3.0 changed the sign of the result entirely).
    That is a genuine curve-shape effect, not a float64 precision-floor
    artifact -- so the fix is to use a small, fixed window (default 0.3
    deg here) and confirm convergence by shrinking it further, NOT to
    average across a range of window sizes (which was the wrong
    diagnostic used in an earlier version of this module for a different,
    genuinely noise-dominated problem).

    Returns (a, converged) where `converged` is True if halving
    theta_deg_range changes the result by less than 5% (a basic check
    that the window is already small enough to be probing the local
    derivative rather than the trace's larger-scale shape).
    """
    pts1 = trace_point(linkage, which, theta_deg_range=theta_deg_range, npts=npts)
    a1 = curvature_sign(pts1)
    pts2 = trace_point(linkage, which, theta_deg_range=theta_deg_range / 2.0, npts=npts)
    a2 = curvature_sign(pts2)
    if np.isnan(a1) or np.isnan(a2):
        return np.nan, False
    denom = max(abs(a1), abs(a2), 1e-300)
    converged = abs(a1 - a2) / denom < 0.05
    return a1, converged


# =============================================================================
# SECTION 7 -- Stability-Crossing Search (over base depth q)
# =============================================================================

def find_q_crossing(W, L0, q_range=(50.0, 3000.0), n=60, theta_deg_range=0.3,
                     isosceles="apex_at_C"):
    """
    Scan the base depth q (holding L's rest position fixed) and locate
    where L's local curvature changes sign.

    NOTE on what varies smoothly and what doesn't, found by direct
    testing: varying L's OWN rest depth (with q fixed) does NOT produce a
    smooth sign change anywhere in the physically sensible range -- L
    stays convex from near P all the way down to just above the C-D
    baseline, and the only sign flip found that way happens discontinuously
    exactly where L's rest point crosses the C-D line itself (a genuine
    topological flip of the C-L-D triangle, not a smooth tuning effect).
    Varying the BASE depth q instead (this function), with L's rest
    position held fixed, DOES give a smooth, well-behaved sign change --
    that is the meaningful "tuning knob" in this corrected model.

    Returns the q (mm) closest to the middle of q_range at which a sign
    change occurs, or None if no sign change is found.
    """
    qs = np.linspace(q_range[0], q_range[1], n)
    avals = np.empty_like(qs)
    for i, q in enumerate(qs):
        A, B, C, D, P, L = geometry_at_base_depth(W, q, L0, isosceles)
        linkage = RobertsLinkageCLD(A, B, C, D, P, L)
        a, converged = curvature_sign_local(linkage, "L", theta_deg_range)
        avals[i] = a if converged else np.nan

    valid = ~np.isnan(avals)
    qs_v, av_v = qs[valid], avals[valid]
    if len(qs_v) < 2:
        return None
    sign_changes = np.where(np.diff(np.sign(av_v)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    # bracket around the first sign change found
    i = sign_changes[0]
    return qs_v[i], qs_v[i + 1]


# =============================================================================
# SECTION 8 -- Solve for the Exact Crossing Depth
# =============================================================================

def solve_q_crossing(W, L0, q_bracket, theta_deg_range=0.3,
                      isosceles="apex_at_C", xtol=1e-3):
    """
    Root-find the exact base depth q at which L's local curvature is zero,
    given a bracket [q_lo, q_hi] known (e.g. from find_q_crossing) to
    contain a sign change.

    Verified for the default geometry (W=265, L0=(0,-29.2)): q* ~= 1229.33
    mm (aspect ratio q*/W ~= 4.64), confirmed by checking that the result
    is stable under halving theta_deg_range further (see
    curvature_sign_local) -- this is a genuine, converged root, not a
    precision-floor artifact like the retracted q~=2693 finding in an
    earlier (topologically incorrect) version of this module.

    Implementation note: the convergence GATE in curvature_sign_local
    (relative agreement between theta_deg_range and theta_deg_range/2) is
    deliberately NOT applied inside this root-finder's objective
    function. Near the root itself a is, by definition, close to zero,
    which makes any RELATIVE convergence criterion fail spuriously (a
    small absolute difference between two already-tiny numbers can still
    look like a large relative disagreement) -- this is the same
    pathology documented in an earlier version of this module for a
    different reliability check. The fix here is simpler: just use a
    small, fixed theta_deg_range directly (default 0.3 deg, already
    confirmed by external testing -- see this module's development notes
    -- to be well converged away from the root), without re-checking
    convergence on every brentq evaluation.
    """
    from scipy.optimize import brentq

    def f(q):
        A, B, C, D, P, L = geometry_at_base_depth(W, q, L0, isosceles)
        linkage = RobertsLinkageCLD(A, B, C, D, P, L)
        pts = trace_point(linkage, "L", theta_deg_range=theta_deg_range, npts=201)
        return curvature_sign(pts)

    q_lo, q_hi = q_bracket
    f_lo, f_hi = f(q_lo), f(q_hi)
    if (f_lo > 0) == (f_hi > 0):
        raise ValueError(
            f"q_bracket={q_bracket} does not bracket a sign change "
            f"(a(q_lo)={f_lo:.3e}, a(q_hi)={f_hi:.3e})."
        )
    return brentq(f, q_lo, q_hi, xtol=xtol)


# =============================================================================
# SECTION 5 -- Interactive Explorer
# =============================================================================

def interactive_explorer(theta_range_deg=10.0):
    """
    Interactive Roberts Linkage explorer (corrected C-D base, independent
    P/L apexes) with sliders for:
        theta   -- drive angle (deg)
        L depth -- L's rest y-coordinate (mm); L is always kept on the
                   centerline (x=0), matching the symmetric construction
                   used throughout this module
        q       -- base depth of C, D (mm)
        W       -- half-width of A, B (mm)

    Changing q, W, or L depth rebuilds the rest geometry (via
    geometry_at_base_depth) and recomputes both traces; changing theta
    just re-evaluates the existing linkage at a new drive angle.
    """
    W0, q0 = 265.0, 254.96
    L_depth0 = -29.2

    def build(W, q, L_depth):
        A, B, C0, D0, P0, L0 = geometry_at_base_depth(W, q, np.array([0.0, L_depth]))
        linkage = RobertsLinkageCLD(A, B, C0, D0, P0, L0)
        P_trace = trace_point(linkage, "P", theta_deg_range=theta_range_deg, npts=121)
        L_trace = trace_point(linkage, "L", theta_deg_range=theta_range_deg, npts=121)
        return A, B, C0, D0, P0, L0, linkage, P_trace, L_trace

    state = {}
    (state["A"], state["B"], state["C0"], state["D0"], state["P0"], state["L0"],
     state["linkage"], state["P_trace"], state["L_trace"]) = build(W0, q0, L_depth0)

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.1, bottom=0.32)
    fig.suptitle("Roberts Linkage Explorer (corrected: independent C-P-D / C-L-D)",
                 fontsize=11, fontweight="bold")

    ax_theta = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_L = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_q = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_W = plt.axes([0.15, 0.05, 0.7, 0.03])

    s_theta = Slider(ax_theta, "theta (deg)", -theta_range_deg, theta_range_deg, valinit=0.0)
    s_L = Slider(ax_L, "L depth (mm)", -300.0, 50.0, valinit=L_depth0)
    s_q = Slider(ax_q, "q (mm)", 50.0, 3000.0, valinit=q0)
    s_W = Slider(ax_W, "W (mm)", 100.0, 500.0, valinit=W0)

    def rebuild(_=None):
        (state["A"], state["B"], state["C0"], state["D0"], state["P0"], state["L0"],
         state["linkage"], state["P_trace"], state["L_trace"]) = build(
            s_W.val, s_q.val, s_L.val)
        redraw()

    def redraw(_=None):
        A, B, C0, D0 = state["A"], state["B"], state["C0"], state["D0"]
        linkage = state["linkage"]
        P_trace, L_trace = state["P_trace"], state["L_trace"]

        ax.cla()
        theta = np.radians(s_theta.val)
        C, D, P, L = linkage.configuration(theta)

        ax.plot(P_trace[:, 0], P_trace[:, 1], color="#aaaaaa", lw=1.2, label="P trace")
        ax.plot(L_trace[:, 0], L_trace[:, 1], color="#cc4422", lw=1.8, label="L trace")

        ax.plot([A[0], C[0]], [A[1], C[1]], color="#3377cc", lw=2.5)
        ax.plot([B[0], D[0]], [B[1], D[1]], color="#22aa66", lw=2.5)
        if P is not None:
            ax.plot([C[0], P[0]], [C[1], P[1]], color="#aa55cc", lw=2.0)
            ax.plot([D[0], P[0]], [D[1], P[1]], color="#3377aa", lw=2.0)
            ax.scatter([P[0]], [P[1]], color="#cc2222", s=80, zorder=5, label="P")
        if L is not None:
            ax.plot([C[0], L[0]], [C[1], L[1]], color="#cc8822", lw=2.0)
            ax.plot([D[0], L[0]], [D[1], L[1]], color="#886622", lw=2.0)
            ax.scatter([L[0]], [L[1]], color="#2222cc", s=80, zorder=5, label="L")
        ax.scatter([A[0], B[0]], [A[1], B[1]], color="#000000", s=60, marker="^")
        ax.scatter([C[0], D[0]], [C[1], D[1]], color="#333333", s=40)

        a_P = curvature_sign(P_trace)
        a_L = curvature_sign(L_trace)
        ax.set_title(f"theta={s_theta.val:.2f} deg, L depth={s_L.val:.1f} mm, "
                     f"q={s_q.val:.1f} mm, W={s_W.val:.1f} mm\n"
                     f"P: a={a_P:.2e} ({stability_label(a_P)})  |  "
                     f"L: a={a_L:.2e} ({stability_label(a_L)})", fontsize=9)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")
        fig.canvas.draw_idle()

    # theta only needs a redraw (cheap); L/q/W changes need a full rebuild
    # (new link lengths, new traces) since they change the rest geometry.
    s_theta.on_changed(redraw)
    s_L.on_changed(rebuild)
    s_q.on_changed(rebuild)
    s_W.on_changed(rebuild)
    redraw()
    plt.show()


# =============================================================================
# SECTION 6 -- Static Figure Output
# =============================================================================

def plot_traces(theta_deg_range=5.0, save_path=None):
    """
    Static figure: linkage snapshot at rest, plus the traces of P and L
    over a small-angle sweep, using the corrected independent-apex model.
    """
    A, B, C0, D0, P0, L0 = geometry_defaults()
    linkage = RobertsLinkageCLD(A, B, C0, D0, P0, L0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.plot([A[0], C0[0]], [A[1], C0[1]], color="#3377cc", lw=2.5, label="AC")
    ax.plot([B[0], D0[0]], [B[1], D0[1]], color="#22aa66", lw=2.5, label="BD")
    ax.plot([C0[0], P0[0]], [C0[1], P0[1]], color="#aa55cc", lw=2.0, label="CP")
    ax.plot([D0[0], P0[0]], [D0[1], P0[1]], color="#3377aa", lw=2.0, label="DP")
    ax.plot([C0[0], L0[0]], [C0[1], L0[1]], color="#cc8822", lw=2.0, label="CL")
    ax.plot([D0[0], L0[0]], [D0[1], L0[1]], color="#886622", lw=2.0, label="DL")
    ax.scatter([A[0], B[0]], [A[1], B[1]], color="#000000", s=70, marker="^")
    ax.scatter([C0[0], D0[0]], [C0[1], D0[1]], color="#333333", s=50)
    ax.scatter([P0[0]], [P0[1]], color="#cc2222", s=90, zorder=5, label="P")
    ax.scatter([L0[0]], [L0[1]], color="#2222cc", s=90, zorder=5, label="L")
    for name, pt in zip("ABCDPL", [A, B, C0, D0, P0, L0]):
        ax.annotate(name, pt, textcoords="offset points", xytext=(6, 6))
    ax.set_aspect("equal"); ax.set_title("Linkage at rest (theta = 0)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax = axes[1]
    P_trace = trace_point(linkage, "P", theta_deg_range=theta_deg_range, npts=201)
    L_trace = trace_point(linkage, "L", theta_deg_range=theta_deg_range, npts=201)
    a_P, a_L = curvature_sign(P_trace), curvature_sign(L_trace)
    rms_P = flatness_rms(P_trace)

    ax.plot(P_trace[:, 0], P_trace[:, 1], color="#000000", lw=2.0,
            label=f"P  (RMS={rms_P:.3f} mm, a={a_P:.2e}, {stability_label(a_P)})")
    ax.plot(L_trace[:, 0], L_trace[:, 1], color="#cc4422", lw=1.8,
            label=f"L  (a={a_L:.2e}, {stability_label(a_L)})")
    ax.set_aspect("equal")
    ax.set_title(f"P and L traces, theta in [-{theta_deg_range}, {theta_deg_range}] deg")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    A, B, C0, D0, P0, L0 = geometry_defaults()
    linkage = RobertsLinkageCLD(A, B, C0, D0, P0, L0)

    print("=" * 70)
    print("  Roberts Linkage -- CORRECTED topology (no C-D link;")
    print("  P and L are independent apexes on the same base C-D)")
    print("=" * 70)
    print(f"\nLink lengths:")
    print(f"  AC = BD = {linkage.AC:.4f} mm")
    print(f"  CP = DP = {linkage.CP:.4f} mm")
    print(f"  CL = DL = {linkage.CL:.4f} mm")

    print("\nSelect which section to run:")
    print("  1 -- Interactive explorer (slider)")
    print("  2 -- P and L traces at +/-5 deg (printed values)")
    print("  3 -- P and L traces (static figure, saved)")
    print("  4 -- Stability-crossing search (scan base depth q for a sign change)")
    print("  5 -- Solve for the exact q at the crossing (brentq)")
    print("  all -- Run 2, 3, 4 (no interactive window)")

    choice = input("Enter choice: ").strip().lower()

    if choice == "1":
        interactive_explorer()

    elif choice == "2":
        P_trace = trace_point(linkage, "P", theta_deg_range=5.0, npts=11)
        L_trace = trace_point(linkage, "L", theta_deg_range=5.0, npts=11)
        print("\nP trace:\n", P_trace)
        print("\nL trace:\n", L_trace)
        print(f"\nP: a={curvature_sign(P_trace):.6f}  {stability_label(curvature_sign(P_trace))}")
        print(f"L: a={curvature_sign(L_trace):.6f}  {stability_label(curvature_sign(L_trace))}")

    elif choice == "3":
        plot_traces(save_path=os.path.join(OUTPUT_DIR, "P_and_L_traces.png"))

    elif choice == "4":
        W_here = float(np.linalg.norm(B - A)) / 2.0
        print(f"\nUsing W={W_here:.4f} mm, L rest position={L0.tolist()}.")
        print("Scanning base depth q for a sign change in L's local curvature...")
        result = find_q_crossing(W_here, L0)
        if result is None:
            print("No sign change found in the default scan range "
                  "(q in [50, 3000] mm). Try widening q_range.")
        else:
            q_lo, q_hi = result
            print(f"\nSign change bracketed between q={q_lo:.2f} mm and "
                  f"q={q_hi:.2f} mm.")
            print("Use option 5 (or solve_q_crossing directly) to refine "
                  "this into an exact root.")

    elif choice == "5":
        W_input = input("Enter W (half-width, mm) [default 265]: ").strip()
        W_here = float(W_input) if W_input else 265.0
        print(f"\nUsing W={W_here:.4f} mm, L rest position={L0.tolist()}.")
        print("Finding bracket, then solving for the exact crossing...")
        result = find_q_crossing(W_here, L0)
        if result is None:
            print("No sign change found -- cannot solve. Try option 4 with "
                  "a wider q_range first.")
        else:
            q_star = solve_q_crossing(W_here, L0, result)
            print(f"\nq* = {q_star:.4f} mm")
            print(f"Aspect ratio q*/W = {q_star / W_here:.3f}")
            print("(base depth at which L's local curvature is exactly "
                  "zero -- i.e. the concave/convex transition, for this "
                  "fixed L rest position)")

    elif choice == "all":
        P_trace = trace_point(linkage, "P", theta_deg_range=5.0, npts=201)
        L_trace = trace_point(linkage, "L", theta_deg_range=5.0, npts=201)
        print(f"\nP: RMS={flatness_rms(P_trace):.4f} mm, a={curvature_sign(P_trace):.6f}  "
              f"{stability_label(curvature_sign(P_trace))}")
        print(f"L: a={curvature_sign(L_trace):.6f}  {stability_label(curvature_sign(L_trace))}")
        plot_traces(save_path=os.path.join(OUTPUT_DIR, "P_and_L_traces.png"))

        print("\n--- Stability-crossing search (base depth q) ---")
        W_here = float(np.linalg.norm(B - A)) / 2.0
        result = find_q_crossing(W_here, L0)
        if result is None:
            print("No sign change found in q in [50, 3000] mm.")
        else:
            q_star = solve_q_crossing(W_here, L0, result)
            print(f"Crossing bracket: {result}  ->  q* = {q_star:.4f} mm "
                  f"(aspect ratio q*/W = {q_star / W_here:.3f})")

    else:
        print("Running interactive explorer by default...")
        interactive_explorer()