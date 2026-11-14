# Live demo script (5 minutes)

Replaces the older script in `AeroNet_7_Day_Plan.md` section 7 and the two-minute one in
[`milestones/M2-demo-v2.md`](milestones/M2-demo-v2.md). Written for the current app (tabs: Home, Explore, Draw, PINN Lab, Results).
Every number said aloud is in a file; where it comes from is given in brackets. Say "measured" or "quoted" every time.

## Before you start (2 minutes, not part of the 5)

1. Start the app: `app\start.ps1`, then open http://localhost:8000 in a full-screen browser at 1440 px wide or more. Use the light or dark
   theme you like (the toggle at the top right; `?theme=light` forces light).
2. Open the page once with the backend running and wait for the hero flow to move. This also loads the saved answers for the offline
   fallback (see "If something fails").
3. Click through the PINN Lab once so the case files are warm; return to Home.
4. Close other programs and plug the laptop in (timings on the Results page were measured on an idle CPU).
5. Have two spare tabs ready: `http://localhost:8000/?present` and `http://localhost:8000/?pinn=naca0012_50x25_a4_m04&round=3#pinn`.

## The script

| Time | Where | Click path | Say |
|---|---|---|---|
| 0:00 (15 s) | **Home** | The hero shows live flow around NACA 0012. Do not click yet. | "AeroNet shows how air moves around a shape. This flow is computed live by a classical panel method. Beside it, I will show a neural network, a PINN, that solves the same physics, and I will say plainly where it wins and where it does not." |
| 0:15 (45 s) | **Explore** | Header: **Explore**. Preset **NACA 0012** is selected. Drag **Angle of attack** from 4 to 10, back to 4, then to 0 (the field and lift arrow follow). Hover the field to read Cp. Click the **Forces** layer to show the lift arrow. | "Real panel-method physics, a few milliseconds per solve (the page shows the time, measured on this laptop). Suction on top, stagnation at the nose. At zero degrees a symmetric airfoil gives zero lift, as it should. This is inviscid and subsonic: no stall, no friction." |
| 1:00 (60 s) | **Draw** | Header: **Draw**. Mouse down in the drawing area, sketch a closed blob, lift the mouse: it solves. Untick **Lift (Kutta condition)** if the warning about the rounded rear appears. Then click **Sedan** in the shape library; the road and the amber note appear. Read the note aloud. Optionally move **Gap to the road**. | "Any closed outline is solved when you lift the pen. A blunt blob has Cl about zero. The sedan sits on a road (ground effect by images) and shows downforce, a negative Cl. Honest note, in amber on screen: this is idealized potential flow, there is no wake, so the pressure at the rear is not realistic and the app never shows a car's drag." |
| 2:00 (90 s) | **PINN Lab** | Header: **PINN Lab**. In **Pick a trained case** choose **NACA 0012, 100x50, 4 deg, Ma 0.4, grid-seq., positivity on**: read Cp RMS and Cl in the metrics row, look at the wall Cp against FVM and panel. Then choose **NACA 0012, 50x25, 4 deg, Ma 0.4, positivity on**. Under **Training replay: watch it learn** press **Play** (or drag the slider) and stop near rounds 3 to 8, then let it run to the end. | "Each case is one trained network for one shape, Mach number and angle. This 100x50 one agrees with the finite-volume reference to Cp RMS 0.0162 and Cl 0.502 against 0.519 (measured; the FVM on the same mesh is at 0.0460). Now the 50x25 one. Watch the training: around round 3 it is accurate (RMS 0.030), then at round 8 it jumps to 0.068 and ends at 0.073 while the loss keeps falling. That late drift is unsolved. What works is stopping at rounds 3 to 5 [M1 section 4(b), M3 sections 3 and 9]." |
| 3:30 (45 s) | **Results** | Header: **Results**. Point at the four headline tiles (3.2x, 2.0x, 0.6%, 3.75 ms), then scroll to **What did not work** (0 of 8, 1 of 3, 5 of 6, NACA 2412 4 degrees). | "On the same sparse mesh the PINN has 3.2 times lower Cp error than the FVM at 100x50 and 2.0 times at 200x100, lift error 0.6 percent (all measured, one seed per cell). 3.75 ms is evaluating a trained network, not a new solve: a new condition means retraining. And here is what did not work: the stop rule stopped none of 8 drifted runs, a cold start collapsed in 1 of 3 seeds, 5 of 6 runs drift, and one NACA 2412 run collapsed and is kept in the data." |
| 4:15 (30 s) | **Results**, bottom, or back to **Home** | Open **What is real, what is a stand-in** at the bottom of Results. | "Limits: inviscid only; cars are idealized; one network per condition; mostly single seeds on one laptop CPU; the FVM times are quoted from the paper. Next: a Kutta-type fix for the drift, a parametric angle of attack, training on a drawn shape, a viscous extension. NNfoil is by Cao, Song and Zhang; we verified it, ran it on a CPU and measured it." |
| 4:45 | | Spare 15 s. Questions. | |

Sources of the numbers: tiles and tags from `GET /api/results` (rows of `results/summary.csv`); 0.0162, 0.0460, Cl 0.5024 and 0.5193: `summary.csv`
E2 and B0; round values 0.030, 0.068, 0.073: the 50x25 positivity-on run, `results/runs/naca0012_50x25_a4_m04/log.json` and M1 section 4(b);
"0 of 8", "1 of 3", "5 of 6": `results/e6_stop_rule.md`, `results/seed_check.csv`, `results/e1_rounds.csv`.
The replay may show only a subset of rounds (snapshots), so read the round shown on screen rather than trusting the numbers above to the digit.

## If something fails

- **The browser or the page is slow:** wait for the hero flow before starting. If the animation stutters, press **Pause** on the flow card; the
  pressure picture and every number stay.
- **The backend dies or does not answer while the page is open:** the page switches to saved results and shows a banner with the date they were
  saved (`frontend/public/cache/bundle.json`). Presets at their starting angle and Mach number, the PINN Lab cases kept in the bundle (the 100x50
  grid-sequenced case, the two 50x25 cases, the NACA 2412 collapsed run and its retry), the Results page and its charts still work; the sliders,
  drawing and new cases do not. Say so: "the live solver is down; these are saved results from this morning". Restart with `app\start.ps1` and
  press **Try again** on the banner. This only works if the page was loaded once while the backend was up (step 2 above).
- **`?present` mode:** `http://localhost:8000/?present` cycles Explore (NACA 0012, NACA 2412), the bird wing, the sedan and the circle, 8 seconds
  each, until the first click, key press or scroll. Use it as a kiosk loop while people arrive, or to carry the talk if you lose your voice. It
  needs the live backend; whether it runs from the saved results alone was not checked for this script.
- **The network fails:** nothing in the demo needs it. The fonts are self-hosted and the solver is local. Only the first `npm install` and
  `pip install` need the internet.
- **The laptop fails:** clone the repository on another machine and follow the root README quick start (needs the internet once). Otherwise
  show the screenshots in `docs/milestones/img/`: Explore `b9_explore_canvas_1440.png`, blob `b8_draw_blob_1440.png`, sedan with road
  `b8_draw_sedan_road_1440.png`, the 100x50 case `c8_lab_gs100_1440.png`, the 50x25 replay `c8_replay_drift_50x25_1440.png`, Results
  `results_1440_light.png`. A recorded backup video is a Plan D5 step; none has been recorded yet.

## Likely questions (two-line answers, with where to check)

1. **Why does a PINN need no labelled data?** It is trained on the Euler equations themselves: the loss is the equation residual at points of the
   mesh plus the wall and far-field conditions (`docs/verification.md`, rows "Euler e1 to e4" and "Loss composition"). The FVM data is only used afterwards to score it.
2. **Why is the leading edge hard?** The flow accelerates sharply there (the paper's abstract), so pressure changes very fast over a tiny distance, and a smooth network
   resolves that badly; the paper maps the airfoil onto a body-fitted mesh to put resolution there, and the related PINN of Sun et al. also reports its largest errors at the leading edge (quoted from our plan, paper not re-read here).
3. **Why is drag about zero?** For steady inviscid subcritical flow the exact drag is zero (d'Alembert's paradox), so any Cd the solver reports is numerical
   error (`docs/verification.md`, force row). That is why the app never shows a car's drag.
4. **How is this different from NeuralFoil and from Sun et al.?** NeuralFoil is a network trained on XFOIL results: viscous, fast (1.16 ms per call, measured here) but a
   different problem and not comparable. Sun et al. use a PINN to optimise shapes and report 18 to 75 hours on a GPU (quoted from `AeroNet_7_Day_Plan.md`); NNfoil solves one condition per network, and we measured it at 334 to 578 s per run on a CPU (M1 section 3).
5. **Which speed numbers are measured and which are quoted?** Measured here: evaluating a trained PINN (1.27, 3.75 and 16.77 ms at 50x25, 100x50 and 200x100,
   8 threads), the panel solve (3.64 ms), NeuralFoil, and the training times. Quoted from the paper's Table 1 (another machine, "not meant as a rigorous comparison"): the FVM times 9, 24, 53 and 285 s (`results/e4_speed.md`).
6. **What bugs did you find and how do you know?** Eleven findings, for example CUDA forced on, one CPU thread, forces never computed, a flow condition hard-coded, an
   unconstrained pressure that is divided by, a mesh-smoothing coefficient four times too small, wrong-sign Jacobian cells at the seam. Each is in `docs/verification.md` with a test (the Euler residuals match an independent conservative evaluation) or a CSV in `results/`.
7. **Where does it fail?** The stop rule failed (0 of 8 drifted runs stopped), 50x25 training drifts after rounds 3 to 5, a cold start collapsed once in 3 seeds without positivity, and the PINN loses to the
   FVM at 200x100 on the cylinder (0.1337 to 0.1403 against 0.0790). Also: inviscid only, no shocks (the paper's Mach 0.68 case is not captured, quoted), one network per condition.
8. **What next?** Fix the drift (a Kutta-type constraint, periodic input, a panel-method prior: none tried yet), a parametric angle of attack (the paper trains one network over alpha), random airfoils,
   training on a drawn shape, a viscous extension and 3D (M3 section 10).
