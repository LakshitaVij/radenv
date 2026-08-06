# One-pager: documentation-loss fix, regraded on the same 10 episodes

**What changed**: Accuracy grading previously only read *saved* OpenEMR
notes. If the model typed correct findings/impressions but never clicked
Save, that content was invisible to grading and scored identically to
writing nothing at all. Fixed: grading now falls back to the model's
actual typed text when nothing was saved. Process is untouched — the
-0.25 penalty for not saving still applies. Same 10 patients, same 10
visits, same 10 conditions as the original batch — no new agent runs,
just re-grading.

**Fix applied to 6 of 10 episodes** (the exact 6 with typed-but-never-saved
content, confirmed by the regrade log) — the other 4 were unaffected
(1 already had a real save, 3 never typed anything substantial).

## Process: unchanged, as expected

Verified identical, episode-by-episode, before vs. after — confirms the
fix only touched Accuracy grading, not Process:

| Episode | Process score (both versions) |
|---|---|
| GRDN004RP5BFHE0T / 2011-09-01 | 10.75/13 |
| GRDN004RP5BFHE0T / 2009-06-01 | 8.50/13 |
| GRDN00BKCEOKC7S1 / 2024-01-12 | 7.75/13 |
| GRDN004RP5BFHE0T / 2009-06-15 | 7.25/15 |
| GRDN006VFINFR877 / 2013-03-25 | 6.75/12 |
| GRDN00BE97VCHW4E / 2025-01-05 | 4.75/13 |
| GRDN00BJ2R8QRO0L / 2016-03-29 | 2.00/11 |
| GRDN00BKCEOKC7S1 / 2025-02-12 | 1.50/13 |
| GRDN00BJ2R8QRO0L / 2013-04-02 | 1.25/11 |
| GRDN006MCYEVYFE8 / 2011-07-25 | -0.50/12 |

## Accuracy: real, meaningful movement on A1 - not on A2 or A3

| Axis | Before: Sensitivity / Precision | After: Sensitivity / Precision | Before: matched items | After: matched items |
|---|---|---|---|---|
| A1 Findings | 1.5% / 33.3% | **17.6% / 40.0%** | 1 | **12** |
| A2 Impressions | 0.0% / 0.0% | 0.0% / 0.0% (unchanged) | 0 | 0 |
| A3 Follow-up | 2.6% / 14.3% | 2.6% / 14.3% (unchanged) | 1 | 1 |

- **A1 (Findings) recovers real signal**: matched findings went from 1 to
  12 once typed-but-lost content was actually graded. Sensitivity more
  than 10x'd (1.5% -> 17.6%). Hallucinated findings also rose (2 -> 18) -
  expected, since more real content is now being graded at all, so both
  correct and incorrect claims surface that were previously invisible.
  Normalized A1 score: 43.8% -> 45.6%.
- **A2 (Impressions) does not improve** - still 0 matched diagnoses even
  with the recovered content graded. This means the earlier 0% wasn't
  only a documentation-loss artifact - the model's actual diagnostic
  synthesis is genuinely weak, independent of the save problem.
  Normalized: 58.9% -> 59.8% (small movement, dominated by the same FNs
  either way).
- **A3 (Follow-up) is completely unchanged**, as expected and previously
  flagged - `select_action` clicks are logged immediately regardless of
  documentation saving, so this fix has no mechanism to affect it. This
  is a useful cross-check: it confirms the fix only moved what it was
  supposed to move.

## Bottom line

The original report's A1 sensitivity (1.5%) understated real findings
recall by roughly 10x because of the documentation-loss grading gap -
corrected figure is 17.6%. A2 and A3 hold at their original (poor)
levels - those failures are real, not measurement artifacts. Overall
picture doesn't change qualitatively: the model is still weak across all
three Accuracy axes, and Process remains the more volatile, often worse
axis episode-to-episode - but A1 in particular was meaningfully worse on
paper than in reality, purely due to a grading gap now fixed.

## Data sources

- `computer_use_eval/regraded_fixed_batch/all_receipts.csv` /
  `.jsonl` - the regraded output, isolated from the original
  `all_receipts.csv` (not merged, not overwritten).
- `computer_use_eval/regraded_fixed_batch/<episode_dir>/receipt.json` -
  per-episode, includes `used_typed_fallback: true/false` so it's
  traceable which episodes actually hit the new fallback path.
- `computer_use_eval/regrade_documentation_fix_batch.py` - the script
  that produced this, reusing the same grade_process/judge logic as the
  original run, with the documentation-fallback fix applied.
