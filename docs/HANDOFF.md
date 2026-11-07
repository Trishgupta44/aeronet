# Handoff: where AeroNet v2 stands (read this first if a session was cut off)

*Updated 2026-11-06 by the main session. Read `docs/AeroNet_v2_Design_Report.md` for the why and
`docs/superpowers/plans/2026-11-05-aeronet-v2-roadmap.md` for the interfaces.*

## Done and merged to `main`
- **Plan A (NNfoil verified on CPU), Milestone 1 complete.** Report: `docs/milestones/M1-nnfoil-verified.md`.
  Seed check: `results/seed_check.csv`. 63 tests: `research/.venv/Scripts/python -m pytest research/tests -q`.

## In progress
- **Plan C, research side**, on branch `plan-c-pinn-results` (not merged). Order: C1-C3 code (run helpers,
  E1/E2/E3 commands, `research/ogrid.py`) -> run the printed experiment commands as ONE sequential chain on an
  idle CPU (never parallel, timings!) -> `experiments.py collect` -> C4 (speed table), C5 (cylinder),
  C6 (export), C11 (ablation flags), C9 (figures + `docs/milestones/M3-pinn-results.md`).
- Held until Plan B is merged: C7, C8, C10 (they need Plan B's backend and `FlowCanvas`).

## State when the laptop was closed (2026-11-06 about 11:20)
- Branch `plan-c-pinn-results` (NOT merged into `main`; `main` has Plan A only). Everything below is committed here.
- **E1 done and committed** (`results/e1_rounds.csv`, roadmap finding 8h): all six 50x25 runs reach Cp RMS
  0.027-0.032 by round 3-5, then 5 of 6 degrade (RMS 0.12-0.22). Neither positivity setting wins. Recipe: stop at
  3-5 rounds; test the C11 remedies.
- **Experiment chain `results/run_plan_c.sh` was still running** (E2 200x100 grid-sequenced run at round 9 of 20;
  E3 NACA 2412 not started). Closing the laptop freezes it (Modern Standby); it resumes when the machine wakes
  if the processes survive. If the python processes are gone, resume like this (repo root, Git Bash):
  - E2: `research/.venv/Scripts/python NNfoil/run_case.py --mesh NNfoil/mesh/NACA0012_200_100.x --shape NACA0012
    --mach 0.4 --alpha 4 --rounds <remaining> --threads 8 --out results/runs/naca0012_200x100_a4_m04_gs_nopos
    --no-positivity --resume`
  - E3: `research/.venv/Scripts/python research/experiments.py commands e3 > /tmp/e3.sh; bash /tmp/e3.sh`
  - then `research/.venv/Scripts/python research/experiments.py collect e2` (and `e3`), commit
    `results/summary.csv`, `results/e*_rounds.csv` and the small run files (`config.json`, `log.json`,
    `snapshots.json`); never commit `model.pt` or `field.npz`.
- Only one training process at a time (timings). Check: `Get-Process python`.
- Weekly Claude limit is nearly used up (extra usage disabled). Priority after the chain: C4 (speed table, needs an
  idle CPU), C5 (cylinder), C9 (figures + `docs/milestones/M3-pinn-results.md`); then Plan B core
  (B1-B3, B7-B8); then C6-C8, C11.
- For unattended runs set the lid action to "Do nothing" (plugged in) and sleep to "Never".

## Design direction (user request, 2026-11-07)
- The user wants the frontend to look like an aerospace corporate site in the style of embraer.com/en (they are
  making a sibling aero website). Brief with measured reference styles: `docs/DESIGN_BRIEF.md`. Plan B has a new
  task B6b (design system and page shell) before B7; B9 defers its "Look" to the brief. Style only: no logo,
  photos, text or licensed fonts copied.
- Finding from the improvements session (E1 data, 6 runs): lower training loss goes with a WORSE Cp error in 5 of 6
  runs (rank correlation -0.48 to -0.98). So the loss is not a usable stopping criterion; stop at 3-5 rounds, and
  test a Kutta/circulation fix in C11. Fold into the M3 brief.

## Not started
- **Plan B (demo v2: draw any shape, animated flow, cars)**, `docs/superpowers/plans/2026-11-05-plan-B-demo-v2.md`.
  It is independent of Plan A and can be run in its own session. Plan D (report, slides) comes last.

## Rules that matter
- No fake numbers: every number comes from a file in `results/` and is labelled measured / quoted / fitted.
- Timed runs need an idle CPU and an awake machine (a slept laptop ruins wall-clock timings).
- Keep the repo private: NNfoil has no licence (credit Cao, Song, Zhang, arXiv 2401.08705).
- Nothing has been pushed to GitHub.

## Open questions only a human can answer
- The Mach number of the NACA 2412 sweep files (fitted about 0.25; paper unconfirmed; 0.3 quoted in the
  improvements doc).
- Do the original deliverables (report, slides, synopsis) exist outside the repo? (README lists `deliverables/`.)
