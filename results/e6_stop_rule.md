# E6: can a reference-free plateau rule stop NNfoil training near the good answer?

*Branch `plan-c-pinn-results`. Everything below is **measured** (computed from files in `results/runs/`; tables in `results/e6_replay.csv` and
`results/summary.csv`, experiment `E6`) unless marked otherwise. Reproduce: `research/.venv/Scripts/python research/stop_rule_eval.py replay --summary`,
then `new-runs --online`, then `tables`. NNfoil is by Cao, Song and Zhang (arXiv 2401.08705); the stop rules are AeroNet's own addition.*

## (a) Question and honesty statement

**Question.** On the 50x25 mesh, training is best at rounds 3 to 5 and then often drifts (Cp RMS 0.027-0.032 early, 0.12-0.22 later in
5 of 6 E1 runs) while the loss keeps falling (roadmap findings 8e, 8h). The FVM answer must not be used to pick the stop round (test-set leakage).
Does the plateau rule, **stop when |dCl| < 0.002 and the wall-Cp RMS change < 0.005 for 2 rounds in a row**, stop near the good answer without the
FVM reference?

**Honesty.**
- The thresholds (0.002, 0.005, window 2) were set after looking at the E1 runs. **E1 is the DEVELOPMENT set** and supports no claim. Only held-out
  runs (the six new seeds run for this experiment, and every other finished run with at least 6 good rounds) support one.
- The thresholds were fixed before this analysis and **were not changed after seeing the results**. Other thresholds appear only in the exploratory
  sensitivity table at the end: tuned on held-out data, **not a claim**.
- The FVM 800x400 file is used only to JUDGE a stop round, never to find it. "Best round" is chosen with the FVM error: **post-hoc, FVM-informed,
  reference point only**.
- Two held-out runs (`naca0012_100x50_a4_m04`, `naca0012_200x100_a4_m04_gs_nopos`) were already replayed in drift Step 1
  (`docs/nnfoil_improvements.md` 3c) with the same thresholds: not tuned on, but seen.
- n is small, so counts are shown. The replay calls `run_case.round_changes` and `run_case.plateau_reached`, the functions the training script
  uses for `--stop-rule plateau`.

## (b) Per-run tables

"Rounds good + failed" counts failed rounds as they are (a failed round = the optimiser restarted from the previous model; it is not a logged
round). "Stop round" is a good-round number. "Cp RMS at stop" is the RMS error against FVM 800x400 at the stop round (the final round when the rule
never fires). "Training s saved" = cumulative training seconds from the stop round to the end of the logged run; a run where one round took more than
5x the median round time and more than 300 s is a laptop sleep, and its time is discarded. "Closest approach to the rule" = the smallest value over
2-round windows of max(dCl/0.002, dCp/0.005); the rule fires exactly when it is below 1.

### DEVELOPMENT (E1: thresholds set on these; no claim)

| run | pos | seed | rounds good + failed | stop round | Cp RMS at stop | Cp RMS final | best round (post-hoc, FVM-informed, reference only) | best RMS (post-hoc) | Cl at stop | Cl final | Cl ref | training s saved | closest approach to the rule | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naca0012_50x25_a4_m04 | on | 0 | 19 + 1 | never | 0.0734 | 0.0734 | 4 | 0.0275 | 0.4119 | 0.4119 | 0.5193 | - | 2.40 | never stops (drifted) |
| naca0012_50x25_a4_m04_nopos | off | 0 | 18 + 2 | never | 0.0325 | 0.0325 | 7 | 0.0262 | 0.4829 | 0.4829 | 0.5193 | - | 1.34 | never stops (no drift) |
| naca0012_50x25_a4_m04_nopos_s1_r20 | off | 1 | 19 + 1 | never | 0.1834 | 0.1834 | 3 | 0.0278 | 0.2226 | 0.2226 | 0.5193 | - | 1.35 | never stops (drifted) |
| naca0012_50x25_a4_m04_nopos_s2_r20 | off | 2 | 19 + 1 | never | 0.2218 | 0.2218 | 4 | 0.0295 | 0.1680 | 0.1680 | 0.5193 | - | 6.72 | never stops (drifted) |
| naca0012_50x25_a4_m04_s1_r20 | on | 1 | 20 + 0 | never | 0.1539 | 0.1539 | 4 | 0.0276 | 0.3168 | 0.3168 | 0.5193 | - | 5.00 | never stops (drifted) |
| naca0012_50x25_a4_m04_s2_r20 | on | 2 | 20 + 0 | never | 0.1621 | 0.1621 | 5 | 0.0271 | 0.2984 | 0.2984 | 0.5193 | - | 1.11 | never stops (drifted) |

### HELD-OUT, six new 50x25 seeds (E6; `--diagnostics`, 20 attempts each, no stop rule in training)

| run | pos | seed | rounds good + failed | stop round | Cp RMS at stop | Cp RMS final | best round (post-hoc, FVM-informed, reference only) | best RMS (post-hoc) | Cl at stop | Cl final | Cl ref | training s saved | closest approach to the rule | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naca0012_50x25_a4_m04_nopos_s3_e6 | off | 3 | 11 + 9 | never | 0.0287 | 0.0287 | 7 | 0.0267 | 0.4889 | 0.4889 | 0.5193 | - | 2.21 | never stops (no drift) |
| naca0012_50x25_a4_m04_nopos_s4_e6 | off | 4 | 20 + 0 | 8 | 0.0447 | 0.0532 | 3 | 0.0326 | 0.4744 | 0.4546 | 0.5193 | 92.4 | 0.95 | good stop (no drift in this run) |
| naca0012_50x25_a4_m04_nopos_s5_e6 | off | 5 | 20 + 0 | never | 0.1139 | 0.1139 | 3 | 0.0267 | 0.3900 | 0.3900 | 0.5193 | - | 1.84 | never stops (drifted) |
| naca0012_50x25_a4_m04_s3_e6 | on | 3 | 14 + 6 | never | 0.0350 | 0.0350 | 4 | 0.0268 | 0.4892 | 0.4892 | 0.5193 | - | 2.74 | never stops (no drift) |
| naca0012_50x25_a4_m04_s4_e6 | on | 4 | 20 + 0 | never | 0.0990 | 0.0990 | 2 | 0.0288 | 0.3468 | 0.3468 | 0.5193 | - | 2.39 | never stops (drifted) |
| naca0012_50x25_a4_m04_s5_e6 | on | 5 | 20 + 0 | never | 0.1658 | 0.1658 | 3 | 0.0297 | 0.2960 | 0.2960 | 0.5193 | - | 1.64 | never stops (drifted) |

Two of the six had many failed rounds: `s3_e6` (positivity on) 14 good + 6 failed, `nopos_s3_e6` 11 good + 9 failed. They are counted as they are.
Both stayed near the good answer (final RMS 0.035 and 0.029) and the rule did not fire in either.

### HELD-OUT, other finished runs (existing)

| run | pos | seed | rounds good + failed | stop round | Cp RMS at stop | Cp RMS final | best round (post-hoc, FVM-informed, reference only) | best RMS (post-hoc) | Cl at stop | Cl final | Cl ref | training s saved | closest approach to the rule | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cylinder_50x25_a0_m04 | on | 0 | 10 + 0 | never | 0.1403 | 0.1403 | 10 | 0.1403 | -0.0044 | -0.0044 | 0.0000 | - | 23.21 | never stops (no drift) |
| cylinder_50x25_a0_m04_nopos | off | 0 | 10 + 0 | never | 0.1337 | 0.1337 | 6 | 0.1326 | 0.0418 | 0.0418 | 0.0000 | - | 16.99 | never stops (no drift) |
| naca0012_100x50_a4_m04 | on | 0 | 20 + 0 | 7 | 0.0144 | 0.0145 | 18 | 0.0144 | 0.5172 | 0.5163 | 0.5193 | 230.5 | 0.01 | good stop (no drift in this run) |
| naca0012_200x100_a4_m04_gs_nopos | off | 0 | 20 + 0 | 5 | 0.0094 | 0.0092 | 13 | 0.0089 | 0.5159 | 0.5118 | 0.5193 | discarded (laptop sleep) | 0.00 | good stop (no drift in this run) |
| naca2412_100x50_a0_m03 | on | 0 | 10 + 0 | 10 | 0.0157 | 0.0157 | 6 | 0.0156 | 0.2468 | 0.2468 | 0.2651 | 0.0 | 0.80 | good stop (no drift in this run) |
| naca2412_100x50_a4_m03_nopos | off | 0 | 10 + 0 | never | 0.0171 | 0.0171 | 9 | 0.0170 | 0.7412 | 0.7412 | 0.7605 | - | 5.89 | never stops (no drift) |
| naca2412_100x50_a4_m03_s1 | on | 1 | 10 + 0 | never | 0.0194 | 0.0194 | 10 | 0.0194 | 0.7364 | 0.7364 | 0.7605 | - | 2.57 | never stops (no drift) |
| naca2412_100x50_a8_m03 | on | 0 | 10 + 0 | never | 0.0439 | 0.0439 | 10 | 0.0439 | 1.1912 | 1.1912 | 1.2516 | - | 2.77 | never stops (no drift) |
| naca2412_100x50_am4_m03 | on | 0 | 10 + 0 | never | 0.0195 | 0.0195 | 9 | 0.0190 | -0.2318 | -0.2318 | -0.2301 | - | 1.87 | never stops (no drift) |

`naca0012_200x100_a4_m04_gs_nopos`: training time discarded (round 6 took about 34,000 s against a median of 103 s: a laptop sleep). `naca2412_100x50_a0_m03`: the rule
fires at round 10 of 10, so nothing is saved. The cylinder runs never reach a good answer (final RMS 0.13-0.14 against 0.009-0.044 for the non-drifting airfoil runs; best round 10 and 6),
so they say nothing about drift.

**Excluded (fewer than 6 good rounds, 21 run folders):** the seed-check and grid-sequenced first-stage runs (`*_s1`, `*_s2`, `*_s3`, `*_gs*` on 50x25 and
100x50) and `naca2412_100x50_a4_m03` (3 good, 7 failed), with 0 to 5 good rounds each. With a window of 2 they cannot show a drift that starts at round 6
to 17. The online run is excluded from the replay because it has a stop rule (see (d)).

## (c) Aggregates

| group | runs | drifted (final RMS > 2x best) | rule stopped | never stopped | stopped at a good answer (RMS <= 1.5x best) | bad stop, too early | bad stop, after the jump | mean RMS at stop (stopped runs) | mean RMS final (same runs) | mean RMS best (post-hoc, all runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| DEVELOPMENT (E1 runs; thresholds set after looking at them) | 6 | 5 | 0 | 6 | 0 | 0 | 0 |  |  | 0.0276 |
| HELD-OUT, six new 50x25 seeds (E6, drift regime) | 6 | 3 | 1 | 5 | 1 | 0 | 0 | 0.0447 | 0.0532 | 0.0286 |
| HELD-OUT, other finished runs (cylinder, NACA2412, NACA0012 100x50 and 200x100) | 9 | 0 | 3 | 6 | 3 | 0 | 0 | 0.0131 | 0.0131 | 0.0457 |
| HELD-OUT, all | 15 | 3 | 4 | 11 | 4 | 0 | 0 | 0.0210 | 0.0231 | 0.0388 |

"Drifted" = final Cp RMS above 2x the best-round RMS. "Good answer" = RMS at the stop round at most 1.5x the best-round RMS. "Mean RMS at stop / final" are
averaged over the stopped runs only; the best-round mean is over all runs of the group (post-hoc). Groups mix meshes and shapes: read the counts.

| | development (6) | held-out new seeds (6) | held-out other (9) | held-out all (15) |
|---|---|---|---|---|
| runs that drifted late | 5 | 3 | 0 | 3 |
| drifted runs the rule stopped before the jump | 0 of 5 | 0 of 3 | n/a | 0 of 3 |
| runs where the rule never triggers | 6 of 6 | 5 of 6 | 6 of 9 | 11 of 15 |
| runs stopped | 0 | 1 (`nopos_s4`, round 8 of 20) | 3 | 4 |
| stopped too early (RMS at stop > 1.5x best) | 0 | 0 | 0 | 0 of 4 |
| mean Cp RMS at stop vs final, stopped runs only | n/a | 0.0447 vs 0.0532 | 0.0131 vs 0.0131 | 0.0210 vs 0.0231 |
| mean Cp RMS with the rule (stop, else final) vs final, all runs | 0.1379 vs 0.1379 | 0.0812 vs 0.0826 | 0.0459 vs 0.0459 | 0.0600 vs 0.0606 |
| mean Cp RMS of the post-hoc best round | 0.0276 | 0.0286 | 0.0457 | 0.0388 |
| training time saved (valid timings) | 0 s | 92.4 s (1 run) | 230.5 s (1 run; 1 run discarded, 1 saves 0) | 322.9 s (2 runs) |

In all three new-seed runs that drifted late (`s4` pos, jump at round 9; `s5` pos, round 6; `nopos_s5`, round 12) the final RMS is 3.4x to 5.6x the best,
and the closest approach to the rule anywhere in the run was 1.64 to 2.39: the changes never stayed below the thresholds for two rounds in a row, though
single rounds with a small change do occur. Pooled with E1, **0 of 8 drifted 50x25 runs were stopped**. The one 50x25 stop (`nopos_s4`, closest approach
0.95, round 8) was in a run whose error had already risen from 0.033 (round 3) to 0.045 and then crept to 0.053; stopping there saved 92 s and 0.0085 of RMS.
All four stops happen in runs that do not drift much, so the one demonstrated effect of the rule is to save time where the answer was already fine.

**Side reference (not the question): a fixed stop at round 3, 4 or 5.** The E1 recipe (finding 8h) is "stop at 3-5 rounds"; N was not tuned on the new seeds.
On the six new seeds the Cp RMS at round 3 / 4 / 5 is 0.027-0.033 (mean 0.030) / 0.027-0.059 (mean 0.035) / 0.027-0.058 (mean 0.036), against
0.029-0.166 at the final round. It needs no reference either and is closer to the best round than the plateau rule on these runs. It is not risk-free:
on the E1 development set round 4 already had one run at 0.101 (a jump at round 4) and round 5 had 0.147. Not tested on other meshes or shapes. (measured)

## (d) Online confirmation run: `naca0012_50x25_a4_m04_s3_e6_online`

Trained with `--stop-rule plateau --diagnostics` (seed 3, positivity on), 20 attempts planned.
**Result: inconclusive. The run was cut off after 12 of 20 planned attempts (11 good rounds, 1 failed). `log.json["stop"]` is absent: the rule did not fire in
those 11 rounds.** The replay of the finished full run of the same seed (`s3_e6`, 14 good rounds) also says "never", so the two agree, and the online run's
Cl, Cp RMS, dCl and dCp are identical (to 4 decimals) to the first 11 rounds of the full run. That shows the online code path and the replay compute the same
thing. It does **not** show that the rule stops a drifting run: this seed does not drift (RMS 0.027-0.038 in rounds 2-14), so "never" is the right answer
for it and tests nothing. Last round: RMS 0.0311, Cl 0.5072 (reference 0.5193). Per the handoff, a second online run (`nopos_s4`, the one case where the
replay predicts a stop, at round 8) was cut off and is not in the repo, so the one stop the replay found has no online test. Summary row:
`E6, NACA0012_a4_s3, pinn_stoprule_online`, marked cut off.

## (e) Verdict

**The plateau rule does not do what it was meant to do.** On the held-out 50x25 seeds it never fired in the 3 runs that drifted late (0 of 3; 0 of 8 counting E1):
the wall Cp and Cl never settle for two rounds in a row within the thresholds, even in the good phase, so the rule cannot see the good answer. It fired in
4 of 15 held-out runs, all runs that were not drifting (100x50, 200x100, NACA2412 at 0 degrees, and one mildly rising 50x25 run), where it did no harm (0 bad
stops of 4) and saved 322.9 s of valid timing in total, but was not needed. The practical stop for the demo and report remains a fixed round 3 to 5 plus
reporting the whole trajectory.

## (f) Caveats

- The drift regime is one case family: NACA0012, 4 degrees, Mach 0.4, 50x25 (12 runs: 6 development, 6 held-out, differing only by seed and positivity).
  The other held-out runs are other meshes and shapes that do not drift, so they cannot test the rule's main job. 15 held-out runs is small.
- Within a run, rounds are not independent: counts are runs, not rounds.
- "Drifted" (final RMS > 2x best) and "bad stop" (> 1.5x best) are definitions fixed in `research/stop_rule_eval.py` before the numbers were produced.
- The online check is inconclusive (see (d)). The 200x100 stop was seen in Step 1 and its time is withheld.
- Summary rows: `pinn_stoprule_replay` for every replayed run (case = run folder name); for the six new seeds also `pinn` / `pinn_nopos` (full-run final
  round) and `pinn_stoprule` / `pinn_nopos_stoprule` (numbers at the replayed stop, equal to the full-run row when the rule never fires), case
  `NACA0012_a4_s<seed>`, experiment `E6`. E1 rows of the same case are untouched (different experiment id).
- |dCl| and the wall-Cp change do not track accuracy in the drift regime (drift Step 1: rho -0.66 to +0.59). Whether the off-grid residual diagnostics
  (`--diagnostics`, drift Step 3) do is not tested here.

## Exploratory sensitivity table (tuned on held-out data: NOT a claim)

Other thresholds applied to the same finished runs, to show how far the fixed rule is from firing. "Bad" = RMS at stop above 1.5x the best-round RMS.
Looser thresholds stop more E1 runs but with bad stops. None of this was set in advance: do not quote it as a result.

| dCl < | dCp < | window | DEV runs stopped (of n) | DEV bad stops | DEV mean RMS (stop, else final) | DEV mean RMS final | HELD-OUT runs stopped (of n) | HELD-OUT bad stops | HELD-OUT mean RMS (stop, else final) | HELD-OUT mean RMS final |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 | 0.005 | 2 | 0 of 6 | 0 | 0.1379 | 0.1379 | 4 of 15 | 0 | 0.0600 | 0.0606 |
| 0.005 | 0.01 | 2 | 4 of 6 | 2 | 0.1076 | 0.1379 | 9 of 15 | 1 | 0.0502 | 0.0606 |
| 0.01 | 0.02 | 2 | 4 of 6 | 1 | 0.1008 | 0.1379 | 12 of 15 | 1 | 0.0502 | 0.0606 |
| 0.02 | 0.03 | 2 | 6 of 6 | 2 | 0.0735 | 0.1379 | 13 of 15 | 1 | 0.0420 | 0.0606 |
| 0.002 | 0.005 | 3 | 0 of 6 | 0 | 0.1379 | 0.1379 | 2 of 15 | 0 | 0.0606 | 0.0606 |
| 0.005 | 0.01 | 3 | 2 of 6 | 1 | 0.1350 | 0.1379 | 5 of 15 | 1 | 0.0597 | 0.0606 |
| 0.01 | 0.02 | 3 | 3 of 6 | 1 | 0.1255 | 0.1379 | 10 of 15 | 2 | 0.0521 | 0.0606 |
