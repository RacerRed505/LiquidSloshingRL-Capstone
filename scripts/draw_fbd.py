"""
Free-body diagram (FBD) of the single-pivot sloshing tank, drawn from this
project's real config values (second-review item "create FBD, singular
pivot").

Every number on the figure is computed here from config/simulation_config.py,
utilities/housner_utils.py and environment/tank_dynamics.py -- the same
functions the simulator itself uses -- so the diagram cannot drift from the
model.

Panel (a): the whole tank on its single pivot O. The liquid is replaced by
its Housner (1963) / ACI 350.3-06 equivalent: an impulsive mass m_i rigidly
attached to the tank and a convective mass m_c on a pendulum of length L.
Panel (b): the convective mass on its own.

    python -m scripts.draw_fbd                # writes figures/fbd_single_pivot.png/.svg
    python -m scripts.draw_fbd --tilt-deg 12  # change the drawn tilt angle
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Circle, Arc  # noqa: E402

from config.simulation_config import (  # noqa: E402
    EnvironmentConfig,
    FluidConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.tank_dynamics import tank_moment_of_inertia_about_pivot  # noqa: E402
from utilities.housner_utils import compute_housner_parameters  # noqa: E402
from utilities.inertia_utils import compute_composite_mass_properties  # noqa: E402
from utilities.tank_geometry import tank_component_boxes  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#1b2733"
MUTED = "#6b7a88"
TANK = "#2f6f93"
LIQUID = "#cfe6f3"
FORCE = "#c0392b"
MOMENT = "#7d3c98"
PEND = "#1e8449"


def model_values() -> dict:
    tank, fluid, pivot, env, world = TankConfig(), FluidConfig(), PivotConfig(), EnvironmentConfig(), WorldConfig()
    g = abs(world.gravity[1])
    mp = compute_composite_mass_properties(tank_component_boxes(tank))
    h = compute_housner_parameters(tank.length, tank.width, fluid.fill_height, fluid.density, g)
    return {
        "g": g,
        "L_tank": tank.length,
        "H_tank": tank.height,
        "fill": fluid.fill_height,
        "m_t": mp.mass,
        "h_t": mp.center_of_mass[1] + tank.position[1] - pivot.position[1],
        "I_O": tank_moment_of_inertia_about_pivot(tank, pivot),
        "m_i": h.impulsive_mass,
        "h_i": h.impulsive_height,
        "m_c": h.convective_mass,
        "h_c": h.convective_pivot_height,
        "L_p": h.pendulum_length,
        "f_n": h.natural_frequency_hz,
        "zeta": fluid.damping_ratio,
        "c": env.pivot_damping_coefficient,
        "tau_max": env.max_motor_torque_nm,
    }


def rot(x: float, y: float, a: float) -> tuple:
    """Rotate a tank-frame point (x along the tank, y up the tank) by angle a about O."""
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))


def _clip_below(poly, y_s):
    """Sutherland-Hodgman clip of a polygon to the half-plane y <= y_s."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        p_in, q_in = p[1] <= y_s, q[1] <= y_s
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = (y_s - p[1]) / (q[1] - p[1])
            out.append((p[0] + t * (q[0] - p[0]), y_s))
    return out


def _area(poly):
    return 0.5 * abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1])))


def level_liquid(half, height, fill, a):
    """Liquid polygon with a LEVEL (world-horizontal) free surface, holding the
    same area as the untilted fill. Real liquid does not rotate with its tank."""
    interior = [rot(x, y, a) for x, y in [(-half, 0), (half, 0), (half, height), (-half, height)]]
    target = 2 * half * fill
    lo, hi = min(p[1] for p in interior), max(p[1] for p in interior)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _area(_clip_below(interior, mid)) < target:
            lo = mid
        else:
            hi = mid
    return _clip_below(interior, 0.5 * (lo + hi))


def arrow(ax, start, end, color, lw=2.2, ms=16, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=6, **kw))


def curved_moment(ax, center, radius, start_deg, end_deg, color, label, label_xy):
    """Curved arrow for a moment about `center`, drawn counter-clockwise from start to end."""
    ax.add_patch(Arc(center, 2 * radius, 2 * radius, theta1=start_deg, theta2=end_deg - 8,
                     color=color, lw=2.2, zorder=6))
    e = math.radians(end_deg)
    tip = (center[0] + radius * math.cos(e), center[1] + radius * math.sin(e))
    pre = math.radians(end_deg - 10)
    tail = (center[0] + radius * math.cos(pre), center[1] + radius * math.sin(pre))
    arrow(ax, tail, tip, color, ms=14)
    ax.text(*label_xy, label, color=color, fontsize=11, ha="center", va="center", zorder=7)


def draw(tilt_deg: float, phi_deg: float, out_stem: str) -> None:
    v = model_values()
    a = math.radians(tilt_deg)
    phi = math.radians(phi_deg)
    half = v["L_tank"] / 2

    fig = plt.figure(figsize=(14, 9.2))
    ax = fig.add_axes([0.02, 0.27, 0.58, 0.68])
    bx = fig.add_axes([0.62, 0.42, 0.36, 0.53])
    for axis in (ax, bx):
        axis.set_aspect("equal")
        axis.axis("off")

    # ---------------- Panel (a): whole tank on its single pivot ----------------
    ax.set_xlim(-0.36, 0.40)
    ax.set_ylim(-0.14, 0.40)
    ax.set_title("(a)  Tank + liquid on a single pivot O", loc="left", fontsize=13, color=INK, pad=6)

    # Ground and pivot pedestal.
    ax.plot([-0.33, 0.37], [-0.075, -0.075], color=INK, lw=1.4)
    for xh in [i * 0.03 - 0.33 for i in range(24)]:
        ax.plot([xh, xh - 0.015], [-0.075, -0.092], color=MUTED, lw=0.8)
    ax.add_patch(Polygon([(0, 0), (-0.035, -0.075), (0.035, -0.075)], closed=True,
                         facecolor="#e9edf0", edgecolor=INK, lw=1.4, zorder=2))
    ax.add_patch(Circle((0, 0), 0.008, facecolor="white", edgecolor=INK, lw=1.6, zorder=8))
    ax.text(0.014, -0.028, "O", fontsize=12, color=INK, fontweight="bold", zorder=8)

    # Reference vertical through O and the tilt angle theta.
    ax.plot([0, 0], [0, 0.36], color=MUTED, lw=1, ls=(0, (4, 4)))
    tank_axis_top = rot(0, 0.36, a)
    ax.plot([0, tank_axis_top[0]], [0, tank_axis_top[1]], color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=90, theta2=90 + tilt_deg, color=INK, lw=1.2))
    ax.text(-0.045, 0.268, r"$\theta$", fontsize=14, color=INK)

    # Liquid region (fill height, tank frame) and tank walls.
    liquid = level_liquid(half, v["H_tank"], v["fill"], a)
    ax.add_patch(Polygon(liquid, closed=True, facecolor=LIQUID, edgecolor="none", zorder=1))
    surf_y = max(p[1] for p in liquid)
    surf_x = [p[0] for p in liquid if abs(p[1] - surf_y) < 1e-9]
    ax.plot([min(surf_x), max(surf_x)], [surf_y, surf_y], color="#7fb3d5", lw=1.2, zorder=1)
    ax.text(min(surf_x) + 0.01, surf_y + 0.006, "free surface stays level", color="#4a90b8",
            fontsize=9, ha="left")
    walls = [rot(x, y, a) for x, y in [(-half, v["H_tank"]), (-half, 0), (half, 0), (half, v["H_tank"])]]
    ax.plot(*zip(*walls), color=TANK, lw=3.2, zorder=3, solid_capstyle="round")
    lbl = rot(half - 0.005, v["H_tank"] + 0.012, a)
    ax.text(*lbl, "tank (acrylic)", color=TANK, fontsize=10, ha="right")

    # Tank centre of mass G_t and its weight.
    gt = rot(0, v["h_t"], a)
    ax.add_patch(Circle(gt, 0.009, facecolor=TANK, edgecolor="white", lw=1.2, zorder=7))
    arrow(ax, gt, (gt[0], gt[1] - 0.075), FORCE)
    ax.text(gt[0] + 0.01, gt[1] - 0.068, r"$m_t\,g$", color=FORCE, fontsize=12)
    ax.text(gt[0] - 0.016, gt[1] + 0.006, r"$G_t$", color=TANK, fontsize=11, ha="right")

    # Impulsive mass m_i (rigid with the tank) and its weight.
    mi = rot(-0.07, v["h_i"], a)
    ax.add_patch(Polygon([rot(-0.07 + dx, v["h_i"] + dy, a) for dx, dy in
                          [(-0.022, -0.016), (0.022, -0.016), (0.022, 0.016), (-0.022, 0.016)]],
                         closed=True, facecolor="#5d8aa8", edgecolor=INK, lw=1, zorder=7))
    ax.text(mi[0], mi[1], r"$m_i$", color="white", fontsize=10, ha="center", va="center", zorder=8)
    arrow(ax, (mi[0], mi[1] - 0.016), (mi[0], mi[1] - 0.085), FORCE)
    ax.text(mi[0] - 0.01, mi[1] - 0.095, r"$m_i\,g$", color=FORCE, fontsize=12, ha="right")

    # Convective pendulum: bob at height h_c (Housner moment arm), hinge L above it.
    hinge_t = (0.08, v["h_c"] + v["L_p"])
    hinge = rot(*hinge_t, a)
    # Rod direction: tank-frame "down" rotated by the relative swing phi.
    down = (math.sin(phi), -math.cos(phi))
    bob_t = (hinge_t[0] + v["L_p"] * down[0], hinge_t[1] + v["L_p"] * down[1])
    bob = rot(*bob_t, a)
    ax.plot([hinge[0], bob[0]], [hinge[1], bob[1]], color=PEND, lw=2, zorder=5)
    ax.add_patch(Circle(hinge, 0.006, facecolor="white", edgecolor=PEND, lw=1.6, zorder=8))
    ax.add_patch(Circle(bob, 0.016, facecolor=PEND, edgecolor="white", lw=1.2, zorder=8))
    ax.text(bob[0], bob[1], r"$m_c$", color="white", fontsize=9, ha="center", va="center", zorder=9)
    # Tension the rod applies to the tank at the hinge (pulls toward the bob).
    tdir = ((bob[0] - hinge[0]) / v["L_p"], (bob[1] - hinge[1]) / v["L_p"])
    arrow(ax, hinge, (hinge[0] + 0.06 * tdir[0], hinge[1] + 0.06 * tdir[1]), FORCE)
    ax.text(hinge[0] + 0.03, hinge[1] + 0.012, r"$T$", color=FORCE, fontsize=12)
    # Relative swing angle phi, measured from the tank's own "down".
    ref_end = rot(hinge_t[0], hinge_t[1] - 0.07, a)
    ax.plot([hinge[0], ref_end[0]], [hinge[1], ref_end[1]], color=PEND, lw=1, ls=(0, (2, 2)))
    ax.text(hinge[0] + 0.022, hinge[1] - 0.075, r"$\varphi$", color=PEND, fontsize=13)

    # Pivot reactions.
    arrow(ax, (-0.13, 0), (-0.012, 0), FORCE)
    ax.text(-0.135, 0, r"$R_x$", color=FORCE, fontsize=12, ha="right", va="center")
    arrow(ax, (0, -0.135), (0, -0.012), FORCE)
    ax.text(0.01, -0.125, r"$R_y$", color=FORCE, fontsize=12)

    # Motor torque and pivot damping moment about O.
    curved_moment(ax, (0, 0), 0.05, 215, 325, MOMENT, r"$\tau$ (motor)", (0.098, -0.052))
    # Damping drawn clockwise (opposing a CCW theta_dot): arc from 60 down to -20 deg.
    ax.add_patch(Arc((0, 0), 0.07, 0.07, theta1=-12, theta2=60, color=MOMENT, lw=2, zorder=6))
    e = math.radians(-20)
    arrow(ax, (0.035 * math.cos(math.radians(-8)), 0.035 * math.sin(math.radians(-8))),
          (0.035 * math.cos(e), 0.035 * math.sin(e)), MOMENT, ms=12)
    ax.text(0.048, 0.04, r"$c\,\dot\theta$", color=MOMENT, fontsize=11)

    # Heights, measured along the tank's own axis from O.
    def dim(h, label, x_t):
        p0, p1 = rot(x_t, 0, a), rot(x_t, h, a)
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
        mid = rot(x_t + 0.012, h / 2, a)
        ax.text(*mid, label, color=MUTED, fontsize=10, va="center")

    dim(v["h_t"], r"$h_t$", 0.235)
    dim(v["h_c"], r"$h_c$", 0.285)
    dim(v["h_i"], r"$h_i$", -0.25)

    # ---------------- Panel (b): the convective mass on its own ----------------
    bx.set_xlim(-0.16, 0.16)
    bx.set_ylim(-0.21, 0.07)
    bx.set_title(r"(b)  Convective mass $m_c$ isolated", loc="left", fontsize=13, color=INK, pad=6)
    h0 = (0.0, 0.03)
    phi_b = math.radians(22)
    bob_b = (h0[0] + 0.15 * math.sin(phi_b), h0[1] - 0.15 * math.cos(phi_b))
    bx.plot([h0[0], 0], [h0[1], h0[1] - 0.17], color=MUTED, lw=1, ls=(0, (3, 3)))
    bx.plot([h0[0], bob_b[0]], [h0[1], bob_b[1]], color=PEND, lw=1.5, ls=(0, (5, 3)))
    bx.add_patch(Circle(h0, 0.006, facecolor="white", edgecolor=PEND, lw=1.5))
    bx.add_patch(Arc(h0, 0.12, 0.12, theta1=-90, theta2=-90 + 22, color=PEND, lw=1.2))
    bx.text(0.006, -0.06, r"$\varphi$", color=PEND, fontsize=13)
    bx.text(-0.012, 0.042, "hinge (fixed to tank)", color=MUTED, fontsize=9.5, ha="center")
    bx.add_patch(Circle(bob_b, 0.018, facecolor=PEND, edgecolor="white", lw=1.2, zorder=5))
    bx.text(*bob_b, r"$m_c$", color="white", fontsize=10, ha="center", va="center", zorder=6)
    ud = ((h0[0] - bob_b[0]) / 0.15, (h0[1] - bob_b[1]) / 0.15)
    arrow(bx, bob_b, (bob_b[0] + 0.08 * ud[0], bob_b[1] + 0.08 * ud[1]), FORCE)
    bx.text(bob_b[0] - 0.012, bob_b[1] + 0.07, r"$T$", color=FORCE, fontsize=12, ha="right")
    arrow(bx, bob_b, (bob_b[0], bob_b[1] - 0.075), FORCE)
    bx.text(bob_b[0] + 0.01, bob_b[1] - 0.07, r"$m_c\,g$", color=FORCE, fontsize=12)
    tang = (math.cos(phi_b), math.sin(phi_b))
    arrow(bx, bob_b, (bob_b[0] - 0.06 * tang[0], bob_b[1] - 0.06 * tang[1]), FORCE, lw=1.8)
    bx.text(bob_b[0] - 0.105, bob_b[1] - 0.04, r"$c_\varphi\,\dot\varphi$" + "\n(liquid damping)",
            color=FORCE, fontsize=10.5, ha="center")

    # ---------------- Equations and values ----------------
    eq = fig.add_axes([0.62, 0.03, 0.36, 0.37])
    eq.axis("off")
    eq.text(0, 0.98, "Equations of motion (small angles)", fontsize=12, color=INK, fontweight="bold", va="top")
    lines = [
        r"Tank about O:  $I_O\,\ddot\theta = \tau - c\,\dot\theta + (m_t h_t + m_i h_i)\,g\,\theta + M_T$",
        r"$m_c$ pendulum:  $\ddot\varphi = -\frac{g}{L}\,\varphi - 2\zeta\omega_n\,\dot\varphi + (\mathrm{tank\ forcing})$",
        r"Slosh energy:  $E = \frac{1}{2}m_c (L\dot\varphi)^2 + m_c g L (1-\cos\varphi)$",
        r"Natural frequency:  $\omega_n^2 = g/L$",
    ]
    for i, s in enumerate(lines):
        eq.text(0, 0.84 - i * 0.13, s, fontsize=11, color=INK, va="top")
    eq.text(0, 0.30,
            r"$M_T$ = moment of the rod tension $T$ about O.  The $+\,g\,\theta$ term is destabilising:"
            "\n"
            r"$G_t$ sits above O, so the tank behaves like an inverted pendulum."
            "\n"
            "Simulator simplifications: the fast training model omits the gravity term;"
            "\n"
            r"its pendulum is forced by the tank's angular velocity $\dot\theta$ only."
            "\n"
            "The real-fluid (SPH) and Chrono tank simulations include gravity.",
            fontsize=9.5, color=MUTED, va="top")

    tbl = fig.add_axes([0.02, 0.02, 0.58, 0.22])
    tbl.axis("off")
    rows = [
        ("Tank mass", r"$m_t$", f"{v['m_t']:.2f} kg", "Tank inertia about O", r"$I_O$", f"{v['I_O']:.3f} kg m$^2$"),
        ("Tank CoM above O", r"$h_t$", f"{v['h_t'] * 1000:.0f} mm", "Liquid fill height", r"$H$", f"{v['fill'] * 1000:.0f} mm"),
        ("Impulsive mass", r"$m_i$", f"{v['m_i']:.2f} kg", "Impulsive height", r"$h_i$", f"{v['h_i'] * 1000:.1f} mm"),
        ("Convective mass", r"$m_c$", f"{v['m_c']:.2f} kg", "Convective height", r"$h_c$", f"{v['h_c'] * 1000:.0f} mm"),
        ("Pendulum length", r"$L$", f"{v['L_p'] * 1000:.0f} mm", "Slosh frequency", r"$f_n$", f"{v['f_n']:.2f} Hz"),
        ("Liquid damping ratio", r"$\zeta$", f"{v['zeta'] * 100:.1f} %", "Pivot damping", r"$c$", f"{v['c']:.2f} N m s"),
    ]
    tbl.text(0, 1.0, "Values used in the model (water, 400 x 250 x 300 mm acrylic tank)",
             fontsize=12, color=INK, fontweight="bold", va="top")
    for i, r in enumerate(rows):
        y = 0.80 - i * 0.14
        for j, (txt, x, col) in enumerate(zip(r, [0.0, 0.25, 0.32, 0.52, 0.76, 0.83],
                                               [INK, INK, TANK, INK, INK, TANK])):
            tbl.text(x, y, txt, fontsize=10.5, color=col, va="top")

    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_stem}.png and .svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tilt-deg", type=float, default=12.0, help="tank tilt drawn in panel (a)")
    parser.add_argument("--phi-deg", type=float, default=-20.0, help="pendulum swing drawn in panel (a)")
    parser.add_argument("--output", default=os.path.join(_REPO_ROOT, "figures", "fbd_single_pivot"))
    args = parser.parse_args()
    draw(args.tilt_deg, args.phi_deg, args.output)


if __name__ == "__main__":
    main()
