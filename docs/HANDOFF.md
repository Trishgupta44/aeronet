# Handoff: where AeroNet v2 stands (read this first if a session was cut off)

*Updated 2026-11-06 by the main session. Read `docs/AeroNet_v2_Design_Report.md` for the why and
`docs/superpowers/plans/2026-11-05-aeronet-v2-roadmap.md` for the interfaces.*

## Merge state (2026-11-08)
- `main` = Plan A + Plan B (demo v2, Milestone 2) + the 7 extra demo features from the improvements session
  (`features-v3`: showdown, critical-Mach warning, NeuralFoil viscous check, evidence card, tradeoff chart,
  offline fallback, critical-Mach optimiser). Fast-forwarded locally (head `cbdab0c`), tests 114 backend and
  48 frontend, NOT pushed. Worktrees `../Aero-Net-planB` and `../Aero-Net-features` can be removed
  (`git worktree remove <path>`) once nobody needs them; the branches stay.
- `plan-c-pinn-results` (this branch) is NOT merged into `main`. Merging it later: expect the showdown's stopgap
  loader `app/backend/aeronet/pinn_data.py` to be replaced by Plan C's PINN-case export (C6-C8).

## Done and merged to `main` (Plan A)
- **Plan A (NNfoil verified on CPU), Milestone 1 complete.** Report: `docs/milestones/M1-nnfoil-verified.md`.
  Seed check: `results/seed_check.csv`. 63 tests: `research/.venv/Scripts/python -m pytest research/tests -q`.

## In progress
- **Plan C, research side**, on branch `plan-c-pinn-results` (not merged). Order: C1-C3 code (run helpers,
  E1/E2/E3 commands, `research/ogrid.py`) -> run the printed experiment commands as ONE sequential chain on an
  idle CPU (never parallel, timings!) -> `experiments.py collect` -> C4 (speed table), C5 (cylinder),
  C6 (export), C11 (ablation flags), C9 (figures + `docs/milestones/M3-pinn-results.md`).
- Held until Plan B is merged: C7, C8, C10 (they need Plan B's backend and `FlowCanvas`).

## STOPPED BY THE USER (2026-11-08, about 02:10): exact state and how to resume
Nothing is running (training processes, the chain and the keep-awake helper were all stopped). Everything finished is committed.

- **Branches:** `main` = Plan A + Plan B + the 7 extra demo features (local only, not pushed). `plan-c-pinn-results`
  (checked out in the main folder, NOT merged into `main`) holds the research: E1-E5 results, E6 work in progress,
  diagnostics code, corrections. Extra worktrees `../Aero-Net-planB` and `../Aero-Net-features` are no longer
  needed (`git worktree remove <path>`).
- **E6 (stop-rule experiment) is unfinished:** 6 held-out runs (seeds 3, 4, 5, positivity on and off, 20 attempts each,
  folders `results/runs/naca0012_50x25_a4_m04[_nopos]_s{3,4,5}_e6`) and one online confirmation run
  (`..._s3_e6_online`) are done and committed (`af5f51f`). Note two runs had many failed rounds
  (`s3_e6`: 14 good + 6 failed, `nopos_s3_e6`: 11 good + 9 failed): count them as they are. The second
  online run (`nopos_s4_e6_online`) was cut off and is NOT committed: delete it.
  Still to do: run the replay on the six new runs (`research/stop_rule_eval.py`, tests pass: 153 total), write
  `results/e6_stop_rule.md` (development set = E1, held-out = everything else; thresholds dCl 0.002, dCp 0.005,
  window 2 were set after looking at E1; verdict with small-n caveat), add the `E6` rows to `results/summary.csv`.
- **Next after E6 (Plan C):** C9 figures + `docs/milestones/M3-pinn-results.md` (`research/make_figures.py`,
  `report_tables.py`); then the app side: C6 export of runs to case JSON, C7 backend, C8 PINN Lab frontend (replaces the
  showdown's stopgap loader `app/backend/aeronet/pinn_data.py`), C10 train-on-your-shape; C11 ablations (stretch).
  Then Plan D (README, report .docx, slides .pptx, offline test, `v1.0` tag; ask the user whether the original
  deliverables exist outside the repo).
- **Merging `plan-c-pinn-results` into `main`:** non-fast-forward; docs conflicts are possible in this HANDOFF and the
  roadmap (`main` has none of these edits). Review before merging; ask the user.
- **Budget:** about 17% of the weekly Claude limit used on 2026-11-08; extra usage is disabled (no charges possible).
- **Only the user can answer:** do the original deliverables (report, slides, synopsis) exist outside the repo? Should
  anything be pushed to GitHub (keep it private: NNfoil has no licence)?
- **Timing rules:** timed runs need an idle, awake machine (a laptop sleep ruined the 200x100 run's time; withheld).
  Run one training process at a time. For unattended runs set the lid action to "Do nothing" and sleep to "Never".

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
