# Milestone 2: Demo v2 (draw any shape, live animated flow, aerospace-corporate look)

Status: **complete** (Plan B, tasks B0-B9). Branch `plan-b-demo-v2`. Labels: **measured** = produced by code or tests in
this repo; **quoted** = copied from another document. The flow solver is still the classical panel method (a stand-in
for the PINN, which comes in Milestone 3).

## 1. What is new

- **Draw any 2D shape** with the mouse or a finger: the outline is smoothed, auto-closed and solved when you lift the
  pen. Mirror drawing, undo, clear, and `.dat` (UIUC) import and export. A self-crossing outline is refused with a clear
  message ("The outline crosses itself").
- **Shape library:** circle, ellipse, teardrop, flat plate, bird wing, sedan and sports car (idealized, see limits).
- **Road under the shape** (ground effect, method of images) with a gap slider; cars use it.
- **Live flow view:** colour map, isobar lines, animated particles that follow the real velocity field, a hover probe,
  surface force arrows and a lift arrow, layer toggles, pause, and a reduced-motion mode.
- **Lift polar** (Cl against angle of attack), **pin and compare** (a ghost of the earlier result and a delta-Cl line),
  **PNG and CSV export**, **shareable URL state**, and a **`?present` mode** that cycles examples for a live talk.
- **New look** following `docs/DESIGN_BRIEF.md`: compressed uppercase headings, mono captions, a sticky navy bar, a
  full-bleed live-flow hero, 2 px-radius buttons, a pale card band, hairline panels. Fonts are self-hosted, so it works
  offline. Style only: no third-party logo, photos, text or licensed fonts.
- **Backend:** `solve_xy` (generic closed outline), `/api/solve_shape`, `/api/polar`, `/api/shapes`,
  `/api/shapes/parse_dat`, and velocity (u, v) in the field.

## 2. Run it

```bash
cd app/frontend && npm install && npm run build      # once
cd ../backend && .venv/Scripts/python -m uvicorn aeronet.main:app --port 8000   # then open http://localhost:8000
```
Presenter mode: `http://localhost:8000/?present`. Light theme: `?light`.

## 3. Screenshots (`docs/milestones/img/`)

| What | File |
|---|---|
| Hero (desktop, phone) | `b9_hero_1440.png`, `b9_hero_phone_390.png` |
| Explore, particles and isobars | `b9_explore_canvas_1440.png` |
| Hand-drawn blob solved | `b8_draw_blob_1440.png` |
| Live stroke with the closing line | `b8_draw_stroke_1440.png` |
| Sedan with road and the "idealized flow" note | `b8_draw_sedan_road_1440.png` |
| Pinned ghost and delta-Cl | `b9_explore_pinned_1440.png` |
| Cp curve and polar card | `b9_cp_polar_1440.png` |
| Phone: Draw, polar | `b8_draw_phone_390.png`, `b9_polar_phone_390.png` |
| Presenter mode | `b9_present_1440.png` |
| Earlier build steps | `b6b_*.png`, `b7_*.png` |

## 4. Two-minute live demo script

1. **(15 s) Explore NACA 0012.** "Real panel-method physics, instant." Move the angle slider; the pressure picture and
   the particles update. Hover to read Cp.
2. **(20 s) Sweep alpha** and show the lift arrow and the polar card (Cl against alpha).
3. **(25 s) Draw a blob** on the Draw section. Lift off: it solves. Turn Lift off for a blunt body (Cl is about 0).
4. **(15 s) Mirror-draw a wing** (draw the upper half only), turn Lift on, and read Cl.
5. **(25 s) Sedan with road.** Show the downforce (negative Cl) and the amber note: "idealized flow, drag not shown".
6. **(10 s) Pin and compare:** pin the sedan, move the road gap, read the delta-Cl.
7. **(10 s) Close:** "Milestone 3 adds the trained neural network next to this; same page, same controls."

## 5. Numbers (all **measured**, from the plan's tests and my re-run on 2026-11-08)

| Check | Result |
|---|---|
| Circle, non-lifting, 0 deg | Cp max 0.9989, Cp min -2.9989 (exact: 1 and -3), Cl and Cd about 1e-16 |
| NACA 0012 outline given in reverse order, 4 deg, Mach 0.4 | Cl 0.5247 against 0.5263 from the code-path solve |
| Ground effect: road far away | Cl 0.4821 against 0.4822 in free air |
| Ground effect: speed on the road line | exactly 0 (no flow through the road) |
| NACA 0012 at 4 deg, Cl by gap 1.0 / 0.3 / 0.1 / 0.05 (80 half-panels) | 0.4898 / 0.5300 / 0.5321 / 0.4511 |
| Preset Cl (0 deg for blunt, 4 deg for lifting) | teardrop 0.508, plate 0.457, bird wing 1.204, sedan -0.496 (gap 0.03), sports car -0.369 |
| Solve time per preset (200 panels) | 6-18 ms |
| NACA 0012 field at width 170 | about 105 ms |
| Sedan solve with field 170 and road | about 211 ms, 289 KB of JSON |
| Polar of 23 angles (sedan with road) | about 216 ms |
| Frame rate (headless desktop Chrome, grid 170, 5 s) | hero 137 fps, explorer 149 fps; app's own counter about 165 idle and about 100 while redrawing; worst single frame about 55 fps |
| Test suites | backend **68 passed** (29 old + 39 new); frontend **39 passed** (7 old + 32 new); `npm run build` succeeds |

The frame rates may be capped by the display refresh and come from a fast machine; they were **not** measured on a
low-end device. Lighthouse and a formal contrast audit were not run. Dark mode with the final particle palette was not
checked.

## 6. Limits and open items (be candid)

- **Inviscid flow:** no separation or wake. For cars the rear pressure is not realistic, drag is about zero by
  d'Alembert's paradox, and the app **never shows a car Cd**. Lift on a car is an indicator of downforce only.
- **Hand-drawn shapes inherit the Lift setting** of the previous shape. A blob drawn after the bird wing triggers an
  amber "rear end is rounded" warning until Lift is switched off. Honest, but the first solve can look odd.
- **Freehand outlines are not stored in the URL** (library shapes and settings are).
- **Hero particles** are faint and slightly blurred on the dark hero; left as is.
- **Not a tab layout:** Draw is a page section, not a tab (a deliberate deviation from the plan; the URL `tab` value
  means "the section last used").
- **One unexplained moment:** during early browser testing the Draw controls changed by themselves (sports car, alpha 1
  deg). It did not recur after a clean reload; the likely cause is another person or tool using the shared browser
  window, but this is unproven.
- **The solver is still the panel method.** The PINN Lab section is a placeholder until Milestone 3.
