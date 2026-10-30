"""
Builds for_advisor/4_calculation_vs_simulation.xlsx: hand calculations of the
model's key quantities (as live Excel formulas) next to the values the
simulations actually produced, with % error (second-review item
"calculation v simulation ... use excel").

Inputs are read from config/simulation_config.py so the workbook always
matches the code. Simulation-side values are the real numbers the code or a
logged run produced; each row names its source (file or PROJECT_LOG entry).

    python -m scripts.build_calc_vs_sim_xlsx
    # Formulas are verified with pycel (LibreOffice could not open files in the
    # cloud container); the workbook is flagged to recalculate when opened.
"""

from __future__ import annotations

import os

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config.simulation_config import (
    EnvironmentConfig,
    FluidConfig,
    MotorConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.tank_dynamics import tank_moment_of_inertia_about_pivot
from utilities.housner_utils import compute_housner_parameters
from utilities.inertia_utils import compute_composite_mass_properties
from utilities.tank_geometry import tank_component_boxes

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO_ROOT, "for_advisor", "4_calculation_vs_simulation.xlsx")

FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BLACK = Font(name=FONT, color="000000")
GREEN = Font(name=FONT, color="008000")
BOLD = Font(name=FONT, bold=True)
TITLE = Font(name=FONT, bold=True, size=14)
NOTE = Font(name=FONT, italic=True, color="666666", size=9)
HEAD_FILL = PatternFill("solid", fgColor="DCE6F1")
SIM_FILL = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def header(ws, row, labels, widths=None):
    for i, text in enumerate(labels, start=1):
        c = ws.cell(row=row, column=i, value=text)
        c.font = BOLD
        c.fill = HEAD_FILL
        c.border = BOX
        c.alignment = Alignment(wrap_text=True, vertical="center")
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w


def build() -> None:
    tank, fluid, pivot, env, motor, world = (
        TankConfig(), FluidConfig(), PivotConfig(), EnvironmentConfig(), MotorConfig(), WorldConfig())
    g = abs(world.gravity[1])
    mp = compute_composite_mass_properties(tank_component_boxes(tank))
    hous = compute_housner_parameters(tank.length, tank.width, fluid.fill_height, fluid.density, g)
    i_o = tank_moment_of_inertia_about_pivot(tank, pivot)

    wb = Workbook()

    # ------------------------------------------------------------------ Inputs
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"] = "Inputs (blue = values taken from the project's configuration)"
    ws["A1"].font = TITLE
    header(ws, 3, ["Name", "Symbol", "Value", "Unit", "Source"], [34, 10, 12, 10, 60])
    inputs = [
        ("Tank outer length (sloshing direction)", "L", tank.length, "m", "TankConfig.length"),
        ("Tank outer width", "W", tank.width, "m", "TankConfig.width"),
        ("Tank wall height", "h", tank.height, "m", "TankConfig.height"),
        ("Wall thickness", "t", tank.wall_thickness, "m", "TankConfig.wall_thickness (6 mm cast acrylic)"),
        ("Acrylic density", "rho_t", tank.density, "kg/m^3", "TankConfig.density (PMMA)"),
        ("Water fill height", "H", fluid.fill_height, "m", "FluidConfig.fill_height"),
        ("Water density", "rho", fluid.density, "kg/m^3", "FluidConfig.density"),
        ("Gravity", "g", g, "m/s^2", "WorldConfig.gravity"),
        ("Water dynamic viscosity (20 C)", "mu", 1.0e-3, "Pa s", "Standard property; scripts/compute_nondimensional_numbers.py"),
        ("Motor drive amplitude", "A", motor.amplitude_rad, "rad", "MotorConfig.amplitude_rad"),
        ("Motor drive frequency", "f_d", motor.frequency_hz, "Hz", "MotorConfig.frequency_hz"),
        ("Wave tilt used for Reynolds number", "theta_w", 9.5, "deg", "Observed ~9-10 deg in the resonance demo (the project notes, main.py --fluid-viz-resonance)"),
        ("Liquid damping ratio", "zeta", fluid.damping_ratio, "-", "FluidConfig.damping_ratio (Sivy et al. 2017)"),
    ]
    name_row = {}
    for i, (name, sym, val, unit, src) in enumerate(inputs, start=4):
        ws.cell(row=i, column=1, value=name).font = BLACK
        ws.cell(row=i, column=2, value=sym).font = BLACK
        c = ws.cell(row=i, column=3, value=val)
        c.font = BLUE
        ws.cell(row=i, column=4, value=unit).font = BLACK
        ws.cell(row=i, column=5, value=src).font = NOTE
        name_row[sym] = i
    ref = {k: f"Inputs!$C${r}" for k, r in name_row.items()}
    ws.cell(row=len(inputs) + 5, column=1,
            value="Change a blue input and every hand calculation updates. Simulation values (grey cells "
                  "on the comparison sheet) are fixed results from runs and do not change.").font = NOTE

    # ------------------------------------------------------- Hand calculations
    hc = wb.create_sheet("Hand calculations")
    hc["A1"] = "Hand calculations (every number below is an Excel formula)"
    hc["A1"].font = TITLE

    # Tank boxes: mass, centroid height, inertia about pivot z-axis.
    hc["A3"] = "1. Tank mass and inertia from its five acrylic plates"
    hc["A3"].font = BOLD
    header(hc, 4, ["Plate", "dim x (m)", "dim y (m)", "dim z (m)", "centre x (m)", "centre y (m)",
                   "mass (kg)", "I about own centre (kg m^2)", "I about pivot O (kg m^2)"],
           [30, 12, 12, 12, 13, 13, 12, 18, 18])
    L, W, h, t = ref["L"], ref["W"], ref["h"], ref["t"]
    plates = [
        ("Base", f"={L}", f"={t}", f"={W}", "=0", f"=-{t}/2"),
        ("Side wall (+z)", f"={L}", f"={h}", f"={t}", "=0", f"={h}/2"),
        ("Side wall (-z)", f"={L}", f"={h}", f"={t}", "=0", f"={h}/2"),
        ("End wall (+x)", f"={t}", f"={h}", f"={W}", f"={L}/2-{t}/2", f"={h}/2"),
        ("End wall (-x)", f"={t}", f"={h}", f"={W}", f"=-({L}/2-{t}/2)", f"={h}/2"),
    ]
    for r, (name, dx, dy, dz, cx, cy) in enumerate(plates, start=5):
        hc.cell(row=r, column=1, value=name).font = BLACK
        for col, v in zip(range(2, 7), (dx, dy, dz, cx, cy)):
            hc.cell(row=r, column=col, value=v).font = GREEN
        hc.cell(row=r, column=7, value=f"={ref['rho_t']}*B{r}*C{r}*D{r}").font = BLACK
        hc.cell(row=r, column=8, value=f"=G{r}*(B{r}^2+C{r}^2)/12").font = BLACK
        hc.cell(row=r, column=9, value=f"=H{r}+G{r}*(E{r}^2+F{r}^2)").font = BLACK
    hc["A10"] = "Total"
    hc["A10"].font = BOLD
    hc["G10"] = "=SUM(G5:G9)"
    hc["I10"] = "=SUM(I5:I9)"
    hc["A11"] = "Tank centre of mass height above pivot O, h_t (m)"
    hc["G11"] = "=SUMPRODUCT(G5:G9,F5:F9)/G10"
    hc["A12"] = ("Inertia about O: I_plate = m(a^2+b^2)/12 + m*(x^2+y^2)  "
                 "(parallel-axis theorem; rotation axis z through O)")
    hc["A12"].font = NOTE

    # Housner.
    hc["A14"] = "2. Housner (1963) / ACI 350.3-06 equivalent model of the water"
    hc["A14"].font = BOLD
    header(hc, 15, ["Quantity", "Formula", "Value"], None)
    rows = [
        ("Total water mass m (kg)", "rho*L*W*H", f"={ref['rho']}*{L}*{W}*{ref['H']}"),
        ("Aspect ratio L/H (-)", "L/H", f"={L}/{ref['H']}"),
        ("Impulsive mass m_i (kg)", "tanh(0.866 L/H)/(0.866 L/H) * m", "=TANH(0.866*C17)/(0.866*C17)*C16"),
        ("Convective mass m_c (kg)", "0.264 (L/H) tanh(3.16 H/L) * m", f"=0.264*C17*TANH(3.16*{ref['H']}/{L})*C16"),
        ("Impulsive height h_i (m)", "(0.5 - 0.09375 L/H) H  [L/H >= 1.333]",
         f"=IF(C17>=1.333,0.5-0.09375*C17,0.375)*{ref['H']}"),
        ("Convective height h_c (m)", "[1 - (cosh y - 1)/(y sinh y)] H, y = 3.16 H/L",
         f"=(1-(COSH(3.16*{ref['H']}/{L})-1)/(3.16*{ref['H']}/{L}*SINH(3.16*{ref['H']}/{L})))*{ref['H']}"),
        ("Sloshing natural frequency omega_n (rad/s)", "sqrt(3.16 g/L tanh(3.16 H/L))",
         f"=SQRT(3.16*{ref['g']}/{L}*TANH(3.16*{ref['H']}/{L}))"),
        ("Sloshing natural frequency f_n (Hz)", "omega_n / 2 pi", "=C22/(2*PI())"),
        ("Equivalent pendulum length L_p (m)", "g / omega_n^2", f"={ref['g']}/C22^2"),
        ("Modelled mass fraction (m_i+m_c)/m (-)", "", "=(C18+C19)/C16"),
    ]
    for r, (q, f, val) in enumerate(rows, start=16):
        hc.cell(row=r, column=1, value=q).font = BLACK
        hc.cell(row=r, column=2, value=f).font = NOTE
        hc.cell(row=r, column=3, value=val).font = BLACK
    hc.column_dimensions["B"].width = 40

    # Exact linear potential theory, inner length.
    hc["A27"] = "3. Exact linear potential-flow sloshing frequency (alternative analytical check)"
    hc["A27"].font = BOLD
    hc["A28"] = "Inner tank length L_in (m)"
    hc["C28"] = f"={L}-2*{t}"
    hc["A29"] = "omega_1 (rad/s)"
    hc["B29"] = "sqrt(pi g / L_in * tanh(pi H / L_in))"
    hc["B29"].font = NOTE
    hc["C29"] = f"=SQRT(PI()*{ref['g']}/C28*TANH(PI()*{ref['H']}/C28))"
    hc["A30"] = "f_1 (Hz)"
    hc["C30"] = "=C29/(2*PI())"

    # Non-dimensional numbers.
    hc["A32"] = "4. Non-dimensional numbers (flow regime)"
    hc["A32"].font = BOLD
    hc["A33"] = "Peak tank angular velocity A*2*pi*f_d (rad/s)"
    hc["C33"] = f"={ref['A']}*2*PI()*{ref['f_d']}"
    hc["A34"] = "Froude number Fr = theta_dot_max*(L/2)/sqrt(g H)"
    hc["C34"] = f"=C33*({L}/2)/SQRT({ref['g']}*{ref['H']})"
    hc["A35"] = "Wave amplitude eta_max = L_p sin(theta_w) (m)"
    hc["C35"] = f"=C24*SIN(RADIANS({ref['theta_w']}))"
    hc["A36"] = "Reynolds number Re = rho*omega_n*eta_max*L/mu"
    hc["C36"] = f"={ref['rho']}*C22*C35*{L}/{ref['mu']}"
    hc["A37"] = "Fr << 0.2 means small, smooth (linear) sloshing; Re ~1e4-1e5 is typical of lab-scale water sloshing."
    hc["A37"].font = NOTE
    hc.column_dimensions["A"].width = 52

    # --------------------------------------------------- Comparison sheet
    cs = wb.create_sheet("Calc vs Simulation", 0)
    cs["A1"] = "Calculation vs simulation"
    cs["A1"].font = TITLE
    cs["A2"] = ("Hand calculation = Excel formula (see 'Hand calculations' sheet). Simulation = value the "
                "code or a logged run actually produced (grey). Error = (simulation - calculation) / calculation.")
    cs["A2"].font = NOTE
    header(cs, 4, ["Quantity", "Unit", "Hand calculation", "Simulation", "Error (%)",
                   "What the simulation value is", "Comment"],
           [40, 9, 16, 14, 11, 52, 60])
    H_ = "'Hand calculations'!"
    comp = [
        ("Tank mass m_t", "kg", f"={H_}G10", mp.mass,
         "Mass the simulator gives the tank body (utilities/inertia_utils.py)",
         "Same plate model, so this checks the code's arithmetic."),
        ("Tank centre-of-mass height above pivot h_t", "m", f"={H_}G11", mp.center_of_mass[1],
         "Centre of mass used by the simulator", "Positive = above the pivot: the tank is an inverted pendulum."),
        ("Tank inertia about pivot I_O", "kg m^2", f"={H_}I10", i_o,
         "environment/tank_dynamics.py::tank_moment_of_inertia_about_pivot", "Parallel-axis theorem, 5 plates."),
        ("Impulsive water mass m_i", "kg", f"={H_}C18", hous.impulsive_mass,
         "utilities/housner_utils.py output", "Housner / ACI 350.3-06."),
        ("Convective (sloshing) water mass m_c", "kg", f"={H_}C19", hous.convective_mass,
         "utilities/housner_utils.py output", "Mass of the equivalent pendulum."),
        ("Equivalent pendulum length L_p", "m", f"={H_}C24", hous.pendulum_length,
         "PendulumBackend pendulum length", ""),
        ("Sloshing frequency: pendulum model (code)", "Hz", f"={H_}C23", hous.natural_frequency_hz,
         "Frequency the fast pendulum model is built with", "Formula vs code: should match exactly."),
        ("Sloshing frequency: pendulum model (measured)", "Hz", f"={H_}C23", 1.30,
         "FFT of the pendulum model's free oscillation (PROJECT_LOG Entry 88)",
         "Within one FFT bin (0.1 Hz) of the calculation."),
        ("Sloshing frequency: real fluid (SPH) vs Housner", "Hz", f"={H_}C23", 1.4218,
         "Peak-to-peak period of the real SPH fluid after a tilt kick (PROJECT_LOG Entry 95)",
         "Real fluid sloshes ~6% faster than the Housner pendulum predicts."),
        ("Sloshing frequency: real fluid (SPH) vs potential theory", "Hz", f"={H_}C30", 1.4218,
         "Same SPH measurement (Entry 95)", "Exact linear theory with the inner length; within ~4%."),
        ("LQR closed-loop sloshing frequency", "rad/s", f"={H_}C22", 8.4406,
         "Imaginary part of the LQR closed-loop poles (python -m control.lqr_controller)",
         "Feedback barely moves the sloshing frequency, as expected."),
        ("Froude number Fr", "-", f"={H_}C34", 0.0538,
         "scripts/compute_nondimensional_numbers.py output", "Fr << 0.2: linear sloshing regime."),
        ("Reynolds number Re", "-", f"={H_}C36", 7.676e4,
         "scripts/compute_nondimensional_numbers.py output", "Within the 1e4-1e5 lab-scale range."),
    ]
    for r, (q, unit, calc, sim, what, comment) in enumerate(comp, start=5):
        cs.cell(row=r, column=1, value=q).font = BLACK
        cs.cell(row=r, column=2, value=unit).font = BLACK
        c = cs.cell(row=r, column=3, value=calc)
        c.font = GREEN
        s = cs.cell(row=r, column=4, value=sim)
        s.font = BLUE
        s.fill = SIM_FILL
        e = cs.cell(row=r, column=5, value=f"=IF(C{r}=0,\"-\",(D{r}-C{r})/C{r})")
        e.font = BLACK
        e.number_format = "0.0%;-0.0%;0.0%"
        cs.cell(row=r, column=6, value=what).font = NOTE
        cs.cell(row=r, column=7, value=comment).font = NOTE
        for col in range(1, 8):
            cs.cell(row=r, column=col).border = BOX
            cs.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="center")
        fmt = "0.0000" if abs(sim) < 100 else "0.00E+00"
        cs.cell(row=r, column=3).number_format = fmt
        cs.cell(row=r, column=4).number_format = fmt
    last = 4 + len(comp)
    cs.cell(row=last + 2, column=1, value="How to read this").font = BOLD
    notes = [
        "Rows 1-7 check that the code implements the textbook formulas correctly (errors should be ~0%).",
        "Rows 8-11 are the real test: they compare the theory against what the simulations measured.",
        "The real-fluid (SPH) frequency is 4-6% above both analytical models: good agreement for an equivalent model.",
        "Colour key: blue = fixed numbers from runs/config, green = links to the hand-calculation sheet, black = formulas.",
    ]
    for i, n in enumerate(notes, start=last + 3):
        cs.cell(row=i, column=1, value=n).font = NOTE
    cs.freeze_panes = "A5"
    cs["D4"].comment = Comment("Grey cells: values produced by the code or by logged simulation runs. "
                               "Each row names its source in column F.", "LiquidSloshingRL team")

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.font is None or cell.font.name != FONT:
                    f = cell.font
                    cell.font = Font(name=FONT, bold=f.bold, italic=f.italic, color=f.color, size=f.size)

    # No cached values are stored (openpyxl cannot write them), so ask Excel /
    # LibreOffice to compute every formula when the file is opened.
    wb.calculation.fullCalcOnLoad = True

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
