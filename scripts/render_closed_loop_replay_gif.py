"""Render the real closed_loop_finetuned_eval.csv run as an animated GIF.

Illustrative replay, not a physics simulation: draws the tank (real
TankConfig dimensions) rotating by the recorded tank_angle, with a marker
at the recorded fluid_com_x/y offset (tank-local frame, displacement from
the fluid's rest position -- NOT the full particle field; that needs a
run with real per-particle capture enabled, not yet done for this backend).

Mirrors the same real recorded data already used in the HTML report's
interactive replay -- this is the exportable-video-file version of that
same thing, for sending/embedding where an interactive page won't do.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

CSV_PATH = "closed_loop_finetuned_eval.csv"
OUT_PATH = "closed_loop_replay.gif"

# Real TankConfig defaults (config/simulation_config.py)
LENGTH = 0.40
HEIGHT = 0.30
FILL_HEIGHT = 0.20
REST_Y = FILL_HEIGHT / 2  # 0.10 m, nominal fluid COM rest height above floor

steps, angle, torque, slosh, comx, comy = [], [], [], [], [], []
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        steps.append(int(row["step"]))
        angle.append(float(row["tank_angle"]))
        torque.append(float(row["torque_nm"]))
        slosh.append(float(row["slosh_energy"]))
        comx.append(float(row["fluid_com_x"]))
        comy.append(float(row["fluid_com_y"]))

N = len(steps)
angle = np.array(angle)
torque = np.array(torque)
slosh = np.array(slosh)
comx = np.array(comx)
comy = np.array(comy)

# Local (unrotated) tank geometry, pivot at (0,0) = footprint center, floor level
half_len = LENGTH / 2
floor = np.array([[-half_len, 0], [half_len, 0]])
left_wall = np.array([[-half_len, 0], [-half_len, HEIGHT]])
right_wall = np.array([[half_len, 0], [half_len, HEIGHT]])
fill_line = np.array([[-half_len, FILL_HEIGHT], [half_len, FILL_HEIGHT]])


def rotate(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


fig, (ax_tank, ax_data) = plt.subplots(
    1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1, 1.3]}
)
fig.patch.set_facecolor("#f4f7f6")

# --- tank panel ---
ax_tank.set_facecolor("#eef2f1")
ax_tank.set_xlim(-0.32, 0.32)
ax_tank.set_ylim(-0.06, 0.36)
ax_tank.set_aspect("equal")
ax_tank.axis("off")
ax_tank.set_title("Real closed-loop replay", fontsize=11, fontweight="bold", color="#16222c")

(floor_line,) = ax_tank.plot([], [], color="#3d5164", lw=3, solid_capstyle="round")
(left_line,) = ax_tank.plot([], [], color="#3d5164", lw=3, solid_capstyle="round")
(right_line,) = ax_tank.plot([], [], color="#3d5164", lw=3, solid_capstyle="round")
(fill_line_plot,) = ax_tank.plot([], [], color="#2f6f93", lw=1, ls="--", alpha=0.6)
com_dot = Circle((0, REST_Y), 0.012, color="#c98a3f", ec="white", lw=1.5, zorder=5)
ax_tank.add_patch(com_dot)
ax_tank.plot(0, 0, "o", color="#3d5164", ms=6, zorder=5)
ax_tank.annotate("pivot", (0, 0), xytext=(0, -14), textcoords="offset points",
                  ha="center", fontsize=8, color="#6d7f8d", family="monospace")

readout = ax_tank.text(
    0.02, 0.98, "", transform=ax_tank.transAxes, va="top", ha="left",
    fontsize=9.5, family="monospace", color="#16222c",
)
safety_text = ax_tank.text(
    0.5, -0.10, "", transform=ax_tank.transAxes, ha="center", va="top",
    fontsize=9.5, family="monospace", color="#b23b3b", fontweight="bold",
)

# --- data panel: 3 stacked traces ---
ax_data.axis("off")
sub_h = 0.30
sub_gap = 0.04
labels_units = [
    ("TANK ANGLE (rad)", angle, "#c98a3f"),
    ("TORQUE (N·m)", torque, "#2f6f93"),
    ("SLOSH ENERGY (J)", slosh, "#3f8f6b"),
]
sub_axes = []
for idx, (label, data, color) in enumerate(labels_units):
    top = 1 - idx * (sub_h + sub_gap)
    sub = fig.add_axes([0.56, top - sub_h, 0.40, sub_h])
    sub.set_facecolor("#ffffff")
    sub.plot(steps, data, color=color, lw=1.6)
    sub.set_xlim(0, N - 1)
    pad = (max(data) - min(data)) * 0.08 or 0.01
    sub.set_ylim(min(data) - pad, max(data) + pad)
    sub.set_title(label, fontsize=8, family="monospace", loc="left", color="#3d5164")
    sub.tick_params(labelsize=7, colors="#6d7f8d")
    for spine in ("top", "right"):
        sub.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        sub.spines[spine].set_color("#dbe3e1")
    (marker,) = sub.plot([], [], "o", color=color, ms=5, mec="white", mew=1)
    vline = sub.axvline(0, color="#6d7f8d", lw=1, ls=":", alpha=0.6)
    sub_axes.append((marker, vline))
sub_axes[-1][0].axes.set_xlabel("control step", fontsize=8, family="monospace", color="#6d7f8d")

fig.suptitle("")


def init():
    return []


def update(i):
    theta = angle[i]
    floor_r = rotate(floor, theta)
    left_r = rotate(left_wall, theta)
    right_r = rotate(right_wall, theta)
    fill_r = rotate(fill_line, theta)
    floor_line.set_data(floor_r[:, 0], floor_r[:, 1])
    left_line.set_data(left_r[:, 0], left_r[:, 1])
    right_line.set_data(right_r[:, 0], right_r[:, 1])
    fill_line_plot.set_data(fill_r[:, 0], fill_r[:, 1])

    local_com = np.array([comx[i], REST_Y + comy[i]])
    world_com = rotate(local_com[None, :], theta)[0]
    com_dot.center = (world_com[0], world_com[1])

    readout.set_text(
        f"step   {i:>3d} / {N - 1}\n"
        f"angle  {angle[i]:+.3f} rad\n"
        f"torque {torque[i]:+.2f} N·m\n"
        f"slosh  {slosh[i]:.3f} J"
    )
    safety_text.set_text("⚠ safety stop -- |ω| > 1.0 rad/s" if i == N - 1 else "")

    for (marker, vline), (_, data, _) in zip(sub_axes, labels_units):
        marker.set_data([steps[i]], [data[i]])
        vline.set_xdata([steps[i], steps[i]])

    return [floor_line, left_line, right_line, fill_line_plot, com_dot, readout, safety_text]


# hold on the last frame a bit longer so the safety-stop reads clearly
frame_ids = list(range(N)) + [N - 1] * 12

ani = animation.FuncAnimation(
    fig, update, frames=frame_ids, init_func=init, blit=False, interval=70
)
ani.save(OUT_PATH, writer=animation.PillowWriter(fps=14))
print("wrote", OUT_PATH)
