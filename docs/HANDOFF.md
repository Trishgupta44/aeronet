# Handoff: where AeroNet v2 stands (read this first if a session was cut off)

*Updated 2026-11-14 by the main session. Read `docs/AeroNet_v2_Design_Report.md` for the why and
`docs/superpowers/plans/2026-11-05-aeronet-v2-roadmap.md` for the interfaces.*

## CURRENT STATE (updated 2026-11-14, after merging everything into `main`)

**Everything is on `main` (local only, NOT pushed).** Tests: frontend 106, backend 153 (+2 skipped: NeuralFoil is optional),
research 228. Nothing is running. Extra worktrees `../Aero-Net-planB` and `../Aero-Net-features` are no longer needed
(`git worktree remove <path>`); the branches `plan-a-nnfoil-verified`, `plan-b-demo-v2`, `features-v3` and
`plan-c-pinn-results` can be deleted when nobody needs them.

What exists:
- **Research (Milestones 1 and 3):** verified NNfoil on CPU (`NNfoil/`, `docs/verification.md`), experiments E1-E6 in
  `results/` (`summary.csv`, figures `results/figures/`), reports `docs/milestones/M1-nnfoil-verified.md` and
  `docs/milestones/M3-pinn-results.md`. Key honest findings: PINN beats the FVM on the same mesh (about 3x at
  100x50, 2x at 200x100); 50x25 training drifts after round 3-5 whatever the positivity setting; the plateau stop rule
  FAILS on held-out runs (fixed early stop at rounds 3-5 is the practice); one cold-start collapse in 3 seeds without
  positivity; grid sequencing (our addition) is reliable.
- **App (Milestone 2 + PINN Lab):** `app/` (FastAPI + React): drawing any shape, animated flow, cars with a road,
  polar, tabs and dark mode, PINN Lab (14 exported trained cases, training replay, numbers dashboard), redesigned
  Results page, 7 extra features from the improvements session (showdown, critical-Mach warning, NeuralFoil check
  (optional), evidence card, trade-off chart, offline fallback, critical-Mach optimiser). Look follows
  `docs/DESIGN_BRIEF.md` (aerospace-corporate style; no third-party logo, text or fonts copied).
- **Docs:** `docs/AeroNet_v2_Design_Report.md` (design decisions), the roadmap and plans in
  `docs/superpowers/plans/`, milestone reports in `docs/milestones/`.

## DONE: Plan D (final package) and the clean-copy test (2026-11-14)
- Built and committed: README overhaul and `docs/demo_script.md`; the project report in the institute's capstone template
  (`deliverables/2_Project_Report_AeroNet.docx`, 54 pages, PDF in `deliverables/pdf/`); the 17-slide deck with speaker notes
  (`deliverables/3_Panel_Presentation.pptx`); progress report, synopsis, related-papers guide and viva sheet in `deliverables/`.
  Sources live in `docs/deliverables_src/`; rebuild with `research/.venv/Scripts/python research/docs_build/build_all.py`,
  `build_pptx.py` and `build_misc_docs.py` (each has `--check`: 0 missing numbers). People-specific details come from
  `docs/deliverables_src/fields.json` (filled from the team's own documents in `docs/DOCS NEEDED FOR MAJOR/`; that folder is
  untracked on purpose).
- Clean-copy test: a fresh `git clone` installs and builds the frontend in 9 s, passes 153 backend and 106 frontend tests, and the
  app serves every endpoint (14 PINN cases, results, showdown, trade-off) with `NNfoil/` and `results/runs/` deleted. The solver
  `POST /api/solve` with `solver=pinn` returns 409 by design.
- Final test counts: research 228, backend 153 (+2 skipped: NeuralFoil is optional), frontend 106. Link check: 0 broken links.

## STILL OPEN (only the user can do these; see `docs/deliverables_src/FILL_IN_CHECKLIST.md`)
- HOD name, dates, institute logo image (`docs/deliverables_src/jiit_logo.png`), PSO list, Turnitin summary, proof of outcome,
  supervisor remarks and signatures, thank-you names, optional laptop price. Read every reference before submission (they were
  checked by metadata, not read in full). Confirm the team-role wording and the degree wording taken from the team's documents.
- Update the Word contents field (right-click, Update field) after the last edit, then print the report; binding is navy blue with
  gold embossing, no spiral (template).
- Decide whether to push to GitHub (keep the repository private: NNfoil has no licence) and which version of the report and slides
  to submit (the team's own `AeroNet_Major_Project_*` files or these).
- Optional extras not done: train a PINN on a drawn shape (C10), ablations that might fix the drift (C11), a 400x200 PINN run,
  random-airfoil runs, a recorded backup video of the demo. The `features-v3` branch has extra deployment work (public mode without
  NNfoil's data) that is not merged.

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
- Older drafts of the report, slides, synopsis or progress report outside the repo? (The user supplied the institute's report template and cover sample in `docs/DOCS NEEDED FOR MAJOR/`; the report follows it. `deliverables/` is generated by the Plan D builders.)
