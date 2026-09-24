# AeroNet: Major Project

A physics-informed neural network (PINN) airfoil solver, built by verifying and presenting NNfoil (Cao, Song & Zhang, 2024).

## What is in this folder

| Path | What it is |
|---|---|
| `docs/AeroNet_7_Day_Plan.md` | **The plan to follow**: 7 days, checkpoints, CPU-only strategy, demo script |
| `docs/superpowers/plans/2026-09-24-aeronet-one-week-plan.md` | Earlier, longer plan (report outline, viva questions). The 7-day plan above supersedes it |
| `deliverables/1_Progress_Report.docx` | 3-page progress report |
| `deliverables/2_Project_Report_AeroNet.docx` | Project report (JIIT format: cover to references, 20 pages: 6 front matter + 14 body) |
| `deliverables/3_Panel_Presentation.pptx` | 17-slide panel presentation with speaker notes |
| `deliverables/4_Synopsis.docx` | 3-page synopsis |
| `deliverables/5_Related_Papers_Guide.docx` | 2-page plain-language guide to 4 related papers (viva / interview) |
| `deliverables/pdf/` | PDF copies of the documents; `deliverables/assets/` holds the figures |
| `app/` | **The proper project:** React (Vite) frontend + Python (FastAPI/NumPy) solver backend, same design and behaviour as the prototype. See `app/README.md`; start with `app/start.ps1` |
| `prototype/` | Original single-file demo (plain HTML/CSS/JS), kept as the reference the app was ported from |
| `prototype/solver.js` | Solver, geometry, and reference-file comparison |
| `prototype/test_solver.js` | Sanity tests for the solver |
| `prototype/reference_naca0012.js` | Sample of NNfoil's bundled FVM reference data (NACA0012, 200x100) |

## Run the prototype

```
python -m http.server 8765 --directory prototype
```

Then open http://localhost:8765 in a browser.

## Run the solver tests

```
node prototype/test_solver.js
```

## Notes

- The prototype's solver is a classical panel method, not a neural network. The PINN replaces it on Day 5 of the plan.
- NNfoil (https://github.com/Cao-WenBo/NNfoil) has no license file. Keep any fork private and credit the authors.
