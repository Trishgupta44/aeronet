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

## Running right now (started 2026-11-06 01:14)
- `results/run_plan_c.sh` runs E1 -> E2 (probe, resume) -> E3 sequentially, logging to
  `results/runs/planc_progress.log` ("ALL DONE" when finished; about 3-5 h). A hidden PowerShell holds the laptop
  awake until then. When done: `research/.venv/Scripts/python research/experiments.py collect e1` (then e2, e3),
  commit `results/summary.csv`, `results/e*_rounds.csv` and the small run files, then do C4, C5, C9.
- Budget note: the weekly Claude limit is nearly used up (extra usage is disabled), so priority order is
  C4, C5, C9 (numbers and the M3 brief), then Plan B core (B1-B3, B7-B8), then C6-C8.

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
