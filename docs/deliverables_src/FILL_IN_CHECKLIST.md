# Fill-in checklist for the project report

The report (`deliverables/2_Project_Report_AeroNet.docx`, source `docs/deliverables_src/report.md`) is built in the institute's
capstone template. Everything only the team can supply is a yellow `[FILL IN: ...]` field. Nothing about people, dates or
signatures was invented. To list what is still open, run

    research/.venv/Scripts/python research/docs_build/build_all.py --check

(it prints `0 missing numbers` and every remaining field). The slides and the four small documents have the same check:
`build_pptx.py --check` and `build_misc_docs.py --check`. After editing a source, rebuild with
`research/.venv/Scripts/python research/docs_build/build_all.py` (add `--pdf` to get the PDF and the page numbers of the
contents lists; needs LibreOffice), then `build_pptx.py` and `build_misc_docs.py`. The institute documents in
`docs/DOCS NEEDED FOR MAJOR/` are not committed on purpose.

## 0. Already filled, from `docs/deliverables_src/fields.json`

Names, enrolment numbers, supervisor, department line, course, programme, semester, academic year and the lead area and
contribution of each student are typed once in `fields.json` (read by every builder through `{{field:key}}` placeholders)
and come from the team's own documents of 14 November 2026 (report, synopsis, log book, presentation), cross-checked with the
mid-term report. To change one, edit `fields.json` and rebuild; an empty value turns back into a highlighted `[FILL IN]`.
Check these once, because the builders cannot: the order of the three students on the cover (Arjun Awasthi, Pratyaksh
Krishnna, Trish Gupta), the spelling of the supervisor's name (Ms Riya Goyal, Assistant Professor), the degree wording in
the certificate ("Bachelor of Technology in Computer Science & Engineering", expanded from "B.Tech.") and the team roles
(taken from the team's own report, Table 1.1; the test counts in that table were left out because they are older than the
numbers in this report).

## 1. Still missing (highlighted `[FILL IN]` in the documents)

| Where | Field | Notes |
|---|---|---|
| Report, certificate | Name of the Head of the Department | Not named in any of the team's documents (they only say "Head of Department") |
| Report, certificate and declaration | Dates | Written when the page is signed. Signature lines stay blank on purpose |
| Report, cover | Institute logo | The team's own report has no logo image. Save the logo as `docs/deliverables_src/jiit_logo.png` and rebuild; a labelled box stands in until then |
| Report, declaration | AI-assistance wording | The mid-term report declared the use of an AI coding assistant; confirm the wording with the supervisor |
| Report, acknowledgement | Anyone else to thank | Delete the field if nobody |
| Report, section 5.3 | Purchase price of the laptop | Only if the department wants it counted; otherwise delete the field |
| Report, Annexure B | Programme specific outcomes (PSOs) and how the project meets each | The team's own report also leaves them blank. The PO table is a **draft** of the standard 12 graduate attributes: confirm against your department's list |
| Report, Annexure A, C | Turnitin summary and proof of outcome | See section 2 |
| Report, references | "Read every reference" reminder | See section 4 |
| Synopsis, progress report | Date of submission, reporting period | |
| Synopsis, progress report | Signatures and supervisor remarks | Signature sections are left for the printed copy |
| Viva sheet | Panel or examiner, who answers which question | A plan the team has to decide (the log book says who led which area, not who answers which question) |

The title "AeroNet: A Physics-Informed Neural Network Solver for Airfoil Aerodynamics" is the same in the synopsis, report,
log book, presentation and mid-term report, so it is not listed.

## 2. Attachments

- **Turnitin similarity summary** (Annexure A): run the finished report through the institute's check, paste the one-page
  summary and write the percentage. No percentage was measured by this tool.
- **Proof of outcome** (Annexure C): paper, journal or conference acceptance, accepted abstract, patent or product link,
  or write "none".

## 3. Do before printing

1. In Word, right-click the Table of Contents and choose Update field (F9, "entire table"). The lists of figures and
   tables are plain text with page numbers computed from a LibreOffice render; Word's pagination may differ by a page, so
   re-check those numbers after the final edit (or rebuild with `--pdf` and read the numbers from that PDF).
2. Check the page numbers: the cover has none; pages ii to x are Roman; Chapter 1 starts at 1.
3. Print on A4, white paper. Binding: navy blue with gold embossing, no spiral (a physical step, not in the file).
4. Read every reference (next section).

## 4. References: read before submission

All 28 entries were checked to exist, and their authors, year, venue and identifiers were confirmed from Crossref
(by DOI) or the arXiv listing. **None was read in full for this report**: where the text says what a paper does, it relies
on the title, the abstract or what the team's own earlier documents recorded. Please read the papers and fix any claim in
Chapters 1 and 2 that the full text does not support. Pay most attention to:

- [3] Sun, Sengupta and Juniper (2023): the summary table says only what the title says (a PINN surrogate with design
  variables as inputs, with PDE-constrained shape optimisation); the "slow to train" remark comes from the team's synopsis
  and must be confirmed.
- [4] NeuralFoil: described as a viscous airfoil-analysis tool; the speed was measured by us, but how it was trained was not
  checked.
- [11] Wassing, Langer and Bekemeyer (2025): described from the abstract (transonic, artificial dissipation).
- [13] to [16] (training difficulties of PINNs): only cited as context from their titles and abstracts; [15] is cited as a
  NeurIPS 2021 paper from its arXiv record (no volume or pages), [16] as an arXiv preprint.
- [25] XFOIL (Drela, 1989): the DOI and chapter pages were confirmed; the book title "Low Reynolds Number Aerodynamics" is
  from memory of the series and should be checked on the copy you cite.
- [23] Katz and Plotkin and [24] Anderson (textbooks), [21] Thompson et al. (book): existence confirmed from library
  records; edition, publisher and year should be checked against the copies the team holds (listings for [24] differ
  between 2016 and 2017).
- [28] The NNfoil repository link comes from the project's own records (no access date is given because the page was not
  re-fetched for this report).

Reference statistics (counted from the list in `report.md`): 28 entries; 16 are from 2021 or later (57 %); the
requirement is at least 50 % from the last five years.

## 5. Numbers that are not from `results/summary.csv`

Everything in the tables and result sentences comes from `results/summary.csv`, `results/seed_check.csv` or
`results/e1_rounds.csv` through placeholders. A few numbers are typed as text and were checked by hand against the files,
with the file named in the sentence: the E6 stop-rule counts and fixed-round ranges (`results/e6_stop_rule.md`, cross-checked
against `results/e6_replay.csv`), the rank correlations (`results/drift_step1.csv`), the quoted paper times (`results/e4_speed.md`),
and the test counts (research and front end re-run for this report; the back-end count is from `docs/HANDOFF.md`).
