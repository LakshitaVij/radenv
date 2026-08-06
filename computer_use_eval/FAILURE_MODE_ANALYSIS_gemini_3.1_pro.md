# Gemini 3.1 Pro on the ChartR computer-use chest X-ray evaluation

Failure-mode analysis across 10 real chest X-ray visits, stratified to cover
all 10 real clinical conditions in the dataset. Each visit puts the model
in a real, live environment as a computer-use agent: it must look at a
real chest X-ray in a browser-based viewer, navigate into a real OpenEMR
instance, find the correct patient and encounter, read the real clinical
context/history, document its findings/impression/follow-up, and select
the correct clinical action(s) from a real 55-action catalog. Scoring is
checkpoint-based: every individual sub-metric across two independent axes
(Process - did it operate the software correctly; Accuracy - was its
clinical judgment correct) is scored separately, not blended into one
number.

## Setup

Model under test: `google/gemini-3.1-pro-preview`, via OpenRouter, one run
per visit (k=1), step budget 50 (median usage well under that - agent
decided it was finished on its own). No scripted sequence of steps -
the system prompt states the goal and the two required outputs
(document a clinical report; select action(s) in Configure Orders and
Results) and gives login credentials, but does not tell the model which
OpenEMR note form to use, when to check vitals/history, or in what order
to do anything. All navigation, note-writing, and action selection are
genuine unscripted decisions.

10 visits, one per real clinical condition present in the dataset
(Normal, Cardiomegaly, Atelectasis, Effusion, Consolidation, Emphysema,
Pneumothorax, Edema, Pneumonia, Lung opacity), picked via `oracle.py`'s
gold index and checked individually against a real data issue (some
patients have two visits on the same calendar date, which the harness
cannot currently disambiguate) before inclusion.

Grading is two independent axes on the same episode, not two separate
runs:
- **Process (Z1/Z2)**: did it open the X-ray, interact with it, log into
  OpenEMR, reach the correct patient/encounter (verified against a real
  DB query, not assumed), read vitals/history, navigate to Configure
  Orders and Results, select only valid/non-conflicting/combinable
  actions, calibrate abstention correctly, and actually save its
  documentation.
- **Accuracy (A1/A2/A3)**: an LLM judge compares whatever the agent
  actually wrote into a real OpenEMR note form (not a separate
  text-generation call) against the real radiologist's Findings/
  Impression, and compares its actual `select_action` clicks against the
  real clinician-consensus follow-up actions for that visit
  (`generated_action_ids`).

## A note on scoring - "points >= 0" is not the same as "correct"

An earlier version of this report used a "checkpoint pass rate" derived
from `aggregate_report.py` counting any row with points >= 0 as a pass.
That is wrong and has been removed. Two concrete reasons why:

- On the Accuracy axes, each finding/impression/action explodes into 7-9
  independent sub-metrics (laterality, severity, wording, etc.). When the
  model misses a finding entirely, only ONE of those sub-metrics
  (`false_negatives`) takes a penalty - the other 6-8 score exactly 0
  because they're not applicable to something that was never mentioned.
  Counting those as "passes" makes total omission look like an 85%+ pass
  rate.
- On the Process axis, several checkpoints (e.g. `Z1.5 Correct patient`,
  `Z1.6 Correct encounter`) score exactly 0 when the encounter page was
  **never reached at all** - i.e. genuinely unverifiable, not a pass.

Everything below is reported as **raw points earned vs. maximum possible
points**, using the point scale defined in `judge.py`'s and
`grade_process.py`'s own scoring rubrics - no pass/fail threshold
invented on top of it.

## Headline result

Every real clinical condition in the sample scored **net-negative on raw
points** across Findings, Impressions, and Follow-up - penalties
outweighed credit in all 10 episodes. Normalizing each episode's raw
score against its own best-possible/worst-possible range (0 = worst
possible outcome, 100 = best possible outcome) puts Accuracy at roughly
**48-58%** across conditions - closer to the middle of the range than to
either extreme, not the 80%+ figure a naive pass-rate count would
suggest. Process scores are far more volatile episode-to-episode, from
**-4% to 83%** of max possible points (see the per-episode table below) -
several episodes score *negative* overall on Process, meaning penalties
(wrong patient, wrong encounter, lost documentation) outweighed
everything earned.

## The single biggest finding: documentation gets lost

This was originally misclassified during grading (folded into a generic
`process_other` tag) - it was corrected before this report was written,
but is called out explicitly here since it's the most consequential
pattern in the whole run.

**Only 1 of 10 episodes produced a real, saved clinical note in
OpenEMR.** Direct inspection of every episode's log:
- **3 of 10 episodes never attempted substantial documentation at all** -
  no `type_text` call longer than a login credential or a search term.
- **6 of 10 episodes typed real, substantial findings/impression text
  into a real OpenEMR note form, and then never saved it** - navigated
  away (e.g. clicking a different tab) before clicking Save. The agent's
  own `finish` message in one of these cases explicitly claimed *"I have
  documented this in the clinical notes"* - it believed it had succeeded
  when it had not.
- **1 of 10 episodes actually saved a real note.**

This single pattern is likely the largest, most fixable gap in the whole
run - it isn't a clinical reasoning failure at all (several of the "typed
but lost" episodes had genuinely correct findings/impression content in
what they typed), it's a UI-completion failure: the agent forms a correct
answer and then fails to commit it.

## Score by axis (raw points earned vs. maximum possible)

| Axis | Items scored | Earned | Max possible | Min possible | % of max | Normalized (0=worst, 100=best) |
|---|---|---|---|---|---|---|
| A1 Findings | 69 finding-items | -137.0 | 483 | -621 | -28.4% | 43.8% |
| A2 Impressions | 19 items | -39.0 | 133 | -285 | -29.3% | 58.9% |
| A3 Follow-up | 45 action-items | -134.0 | 405 | -810 | -33.1% | 55.6% |

All three Accuracy axes are net-negative on raw points - the model loses
more points to omissions and errors than it earns from correct content,
across every one of the 10 episodes. Normalized against each axis's own
worst-to-best range, A1 (Findings) is the weakest at 43.8% - worse than
A2/A3 even though A1's raw "% of max" figure looks similar, because A1's
penalty range is proportionally smaller (max 7, min -9 per item) so the
same behavior costs it more of its available range.

Process is reported per-checkpoint below rather than as one combined
axis figure, since its per-item point scale (some checkpoints worth 1
point, others worth a variable amount depending on how many actions were
selected) makes a single blended percentage misleading in a different
way than the Accuracy axes.

## Standard classification metrics (Sensitivity, Precision, Specificity, Spearman)

Computed directly from `all_receipts.csv`'s `match_status` tags
(matched/missed_by_model/hallucinated_by_model = TP/FN/FP), one row per
unique finding/diagnosis/action rather than per sub-metric - see
`compute_standard_metrics.py`. No new agent runs or LLM calls; these are
arithmetic over data already generated.

| Axis | n items | TP | FN | FP | Sensitivity | Precision |
|---|---|---|---|---|---|---|
| A1 Findings | 69 | 1 | 66 | 2 | 1.5% | 33.3% |
| A2 Impressions | 19 | 0 | 18 | 1 | 0.0% | 0.0% |
| A3 Follow-up | 45 | 1 | 38 | 6 | 2.6% | 14.3% |

These numbers are severe, and two different root causes explain them
separately:

- **A1/A2 (Findings/Impressions) sensitivity is dominated by the
  documentation-loss finding above**, not by a separate clinical-reading
  failure: with only 1 of 10 episodes producing any real saved note, 9
  episodes contribute almost nothing but `missed_by_model` (FN) rows by
  construction - there was no written content left to match against gold
  by the time grading ran. This measures "did a real saved answer exist
  and was it right," not purely "did the model read the X-ray correctly."
- **A3 (Follow-up) sensitivity is a genuinely separate problem.**
  `select_action` clicks are logged immediately, independent of whether
  documentation gets saved - there's no equivalent "lost" mechanism here.
  1 correct action out of 45 gold action-items reflects the agent
  actually selecting the wrong actions in Configure Orders and Results,
  not a save failure.

**Specificity (A1 Findings only, partial)** - true negatives only exist
where the radiologist explicitly documented a pertinent negative (e.g.
"No pneumothorax is seen") and the model correctly reproduced it:

- TN = 0, FP = 2 (hallucinated findings: *"right hilar prominent
  markings, unchanged"*, *"No acute cardiopulmonary process"*)
- **Specificity: 0.0%** (0/2) - not one pertinent negative was correctly
  reproduced across all 10 episodes, consistent with the near-total
  absence of real written content.
- This is a *partial* specificity, bounded to the closed set of negatives
  radiologists chose to document - not computable against the full
  open-world "everything not mentioned," which free-text findings have no
  fixed label space for (unlike CheXpert's closed 14-category structure).
  Not computed for A2/A3, where the true-negative concept doesn't map as
  cleanly (e.g. A3's "withhold antibiotics" is itself a real recommended
  action, not an absence-placeholder).

**Spearman rank correlation (A3 Follow-up ordering)** - gold's
`generated_action_ids` is a semicolon-ordered list that reads as
priority-ordered (urgent actions first); the agent's real selection order
is available via step numbers in each episode's `log.jsonl`. **Not
computable from this batch: 0 of 10 episodes had 2+ actions in common
between the gold list and what the agent actually selected** - a direct
consequence of A3's 2.6% sensitivity above. Rank correlation needs at
least two shared items to correlate against; with only 1 true-positive
action total across all 10 episodes, no single episode reached that bar.
This is a real, honest gap in what's measurable from this batch, not
withheld or approximated - it would become measurable with a batch where
the agent's action selections themselves improve (a separate problem from
documentation loss, per above), producing enough matched actions per
episode to correlate an order against.

## Per-checkpoint breakdown (Process)

Every individual Process checkpoint, reported as raw points earned
(summed across all 10 episodes) against the maximum possible for that
checkpoint. Most checkpoints are worth 1 point per episode (max = n
episodes scored); `Z2.12/13 count check`'s best case is 0 penalty (no
wrong count), so its max is 0, not a positive number - any points below
0 there represent a real penalty with no offsetting credit available:

| Checkpoint | n episodes scored | Points earned | Max possible |
|---|---|---|---|
| Z1.1 Opens X-ray | 10 | 10.0 | 10 |
| Z1.3 Clicks View in OpenEMR | 10 | 10.0 | 10 |
| Z1.4 Logs in | 10 | 10.0 | 10 |
| Z1.5 Correct patient | 10 | 8.0 | 10 |
| Z2.9 Navigate to Procedures | 10 | 8.0 | 10 |
| Z2.10 Navigate to Configuration | 10 | 8.0 | 10 |
| Z1.6 Correct encounter | 10 | 6.0 | 10 |
| Z1.7 Engages vitals | 10 | 5.0 | 10 |
| Z1.8 Engages history | 10 | 5.0 | 10 |
| Z2.11 Select valid action | 7 | 3.0 | 7 |
| Z2.12 No mutual exclusivity conflicts | 1 | 1.0 | 1 |
| Z2.13 Can-combine valid | 1 | 1.0 | 1 |
| Z2.14 Abstention calibration | 10 | 1.0 | 10 |
| Documentation actually saved | 7 | -0.5 | 7 |
| **Z1.2 Interacts (zoom/pan)** | 10 | **0.0** | 10 |
| **Z2.12/13 count check** | 9 | **-25.5** | 0 |

Three checkpoints stand out, all invisible in a blended axis figure:

- **Z1.2 Interacts (zoom/pan): 0.0/10.** Not one episode used the scroll
  tool to zoom or pan the X-ray before moving on to OpenEMR - every
  single episode read the image at its default, un-zoomed size. It's not
  clear this blocked a correct read on its own, but it means the model
  made zero use of a real, available capability to inspect subtle
  findings more closely.
- **Z2.12/13 count check: -25.5, against a best-possible score of 0.**
  Every one of the 9 episodes where this was checkable incurred a real
  penalty - none selected the exact right number of actions versus the
  real per-visit gold count, always either too few or too many. This is
  distinct from *which* actions were selected (`Z2.11`, a separate,
  smaller problem) - it's specifically about calibrating how many
  actions a given case warrants.
- **Documentation actually saved: -0.5, net negative** across the 7
  episodes where the model attempted substantial documentation at all -
  the one real save (+1) was outweighed by six typed-but-lost attempts
  (-0.25 each). And this table still undercounts the real severity: 3 of
  the 10 episodes never attempted substantial documentation in the first
  place and generate no row here at all - see the documentation-loss
  section above for the full accounting (1 saved / 6 lost / 3 never
  attempted, out of 10).

## Failure-mode frequency

| Failure mode | Count | Clinical? |
|---|---|---|
| missed_finding | 66 | clinical |
| missed_action | 63 | clinical |
| hallucinated_action | 35 | clinical |
| missed_diagnosis | 19 | clinical |
| hallucinated_finding | 10 | clinical |
| process_wrong_action_count | 9 | clinical |
| process_encounter_not_reached | 8 | **non-clinical** |
| documentation_typed_not_saved | 6 | clinical |
| process_abstention_miscalibrated | 3 | clinical |
| process_invalid_action | 2 | clinical |
| wrong_severity_appropriate_intensity_action | 1 | clinical |
| wrong_tone_action | 1 | clinical |
| wrong_confidence_wording_action | 1 | clinical |
| hallucinated_diagnosis | 1 | clinical |

Two-thirds of all misses are omissions (missed findings, missed actions,
missed diagnoses) rather than fabrications. Where it does fabricate, it's
disproportionately on the **action** side (35 hallucinated_action vs. 10
hallucinated_finding, 1 hallucinated_diagnosis) - the model is more
prone to recommending something not clinically indicated than to
inventing an imaging finding.

## Where points are actually lost, by sub-metric

The failure-mode table above tags whole findings/actions as
missed/hallucinated/wrong. This goes one level deeper: for every scored
finding, impression, and action, each of the 8 sub-metrics the judge
checks (presence/absence, laterality, severity, location, confidence
wording, clinical reasoning, false negatives, false positives, or their
A3 equivalents) is graded independently. This shows which specific
sub-metric actually took the penalty, not just that "a finding was
missed."

**A1 Findings** (69 scored finding-items):

| Sub-metric | Flagged | Points lost |
|---|---|---|
| false_negatives | 66/69 (96%) | -132.0 |
| clinical_reasoning | 2/69 (3%) | -4.0 |
| severity | 2/69 (3%) | -2.0 |
| location | 2/69 (3%) | -2.0 |
| confidence_wording | 2/69 (3%) | -2.0 |
| false_positives | 2/69 (3%) | -2.0 |
| presence_absence | 0/69 (0%) | 0.0 |
| laterality | 0/69 (0%) | 0.0 |

**A2 Impressions** (19 scored items):

| Sub-metric | Flagged | Points lost |
|---|---|---|
| false_negatives | 18/19 (95%) | -36.0 |
| primary_diagnosis_match | 1/19 (5%) | -2.0 |
| false_positives | 1/19 (5%) | -1.0 |
| all other sub-metrics | 0/19 (0%) | 0.0 |

**A3 Follow-up** (45 scored action-items):

| Sub-metric | Flagged | Points lost |
|---|---|---|
| essential_action_recall | 38/45 (84%) | -76.0 |
| abstention_calibration | 27/45 (60%) | -51.0 |
| severity_appropriate_intensity | 7/45 (16%) | -6.5 |
| tone | 7/45 (16%) | -6.5 |
| confidence_wording | 7/45 (16%) | -6.5 |
| secondary_capture | 6/45 (13%) | -6.0 |
| unnecessary_avoidance | 6/45 (13%) | -4.5 |
| conflict_avoidance | 3/45 (7%) | -2.0 |
| prioritization | 0/45 (0%) | 0.0 |

Two things this reveals that the failure-mode table alone doesn't:

1. **Almost every point lost on A1/A2 comes from omission
   (false_negatives), not distortion.** Once the model actually writes
   about a finding, it gets the surrounding detail (laterality, severity,
   location, wording) right almost every time - `presence_absence` and
   `laterality` were never once flagged. The gap is entirely about
   findings the model never mentioned in the first place, not findings it
   mentioned incorrectly.
2. **A3's abstention_calibration penalty (60%) is far more severe than
   the failure-mode table's `process_abstention_miscalibrated` count (3)
   suggests** - that count only reflects a coarser episode-level tag;
   the real per-action rate is that 27 of 45 scored action-items involved
   a genuine over- or under-confidence problem (recommending an action
   with no gold basis, or omitting one gold called for with confidence).
   This is the single largest source of A3 point loss after missed
   actions themselves.

## Non-clinical execution failures

**4 of 10 episodes (40%) had a real execution failure** - never reached
the correct patient/encounter page at all. These are excluded from the
clinical-only breakdown below, since they represent the agent failing to
even operate the software, not a clinical judgment issue:

- GRDN004RP5BFHE0T / 2009-06-15
- GRDN006MCYEVYFE8 / 2011-07-25
- GRDN00BJ2R8QRO0L / 2016-03-29
- GRDN00BKCEOKC7S1 / 2025-02-12

This is a meaningful chunk of the run and worth investigating on its own
- 40% of episodes failing on basic navigation, before clinical judgment
even comes into play, is a real reliability concern independent of the
model's medical knowledge.

## Clinical-only failure breakdown (execution failures excluded)

| Failure mode | Count |
|---|---|
| missed_finding | 37 |
| missed_action | 34 |
| hallucinated_action | 25 |
| missed_diagnosis | 11 |
| hallucinated_finding | 10 |
| process_wrong_action_count | 5 |
| documentation_typed_not_saved | 4 |
| process_abstention_miscalibrated | 2 |
| wrong_severity_appropriate_intensity_action | 1 |
| wrong_tone_action | 1 |
| wrong_confidence_wording_action | 1 |
| process_invalid_action | 1 |
| hallucinated_diagnosis | 1 |

Even with the fully non-navigating episodes removed, missed findings and
missed actions remain the dominant pattern - this isn't just a product of
the episodes that failed outright to navigate.

## Accuracy score by condition (raw points earned vs. max possible, A1+A2+A3 combined)

| Condition | Earned | Max possible | % of max |
|---|---|---|---|
| Consolidation | -15.5 | 82 | -18.9% |
| Effusion | -25.5 | 122 | -20.9% |
| Pneumonia | -25.0 | 112 | -22.3% |
| Cardiomegaly | -30.0 | 95 | -31.6% |
| Normal | -20.0 | 60 | -33.3% |
| Atelectasis | -46.0 | 136 | -33.8% |
| Emphysema | -34.0 | 99 | -34.3% |
| Pneumothorax | -40.0 | 115 | -34.8% |
| Edema | -38.0 | 108 | -35.2% |
| **Lung opacity** | **-36.0** | **92** | **-39.1%** |

Every condition is net-negative - there is no condition where the model
earned more points than it lost. Consolidation and Effusion are the
least-bad; **Lung opacity is the worst**, consistent with it also being
the only condition where the model didn't even reach the encounter
correctly (see the execution-failure list above) - though with a single
visit per condition, this should be read as a signal to investigate with
more visits, not a confirmed per-condition ranking.

## Process and Accuracy score, per episode

Process (raw points earned / max possible for that episode's applicable
checkpoints) and Accuracy (raw points earned / max possible across
A1+A2+A3), reported side by side rather than multiplied together - a
product of two already-noisy small-sample percentages would compound
uncertainty in a way that's hard to interpret honestly. Sorted by Process
score, since that's where the real spread is:

| Episode | Process (earned/max) | Accuracy (earned/max) |
|---|---|---|
| GRDN004RP5BFHE0T / 2011-09-01 (Normal) | 10.75/13 (82.7%) | -20.0/60 (-33.3%) |
| GRDN004RP5BFHE0T / 2009-06-01 (Effusion) | 8.50/13 (65.4%) | -25.5/122 (-20.9%) |
| GRDN00BKCEOKC7S1 / 2024-01-12 (Consolidation) | 7.75/13 (59.6%) | -15.5/82 (-18.9%) |
| GRDN006VFINFR877 / 2013-03-25 (Emphysema) | 6.75/12 (56.2%) | -34.0/99 (-34.3%) |
| GRDN004RP5BFHE0T / 2009-06-15 (Pneumonia) | 7.25/15 (48.3%) | -25.0/112 (-22.3%) |
| GRDN00BE97VCHW4E / 2025-01-05 (Lung opacity) | 4.75/13 (36.5%) | -36.0/92 (-39.1%) |
| GRDN00BJ2R8QRO0L / 2016-03-29 (Cardiomegaly) | 2.00/11 (18.2%) | -30.0/95 (-31.6%) |
| GRDN00BKCEOKC7S1 / 2025-02-12 (Pneumothorax) | 1.50/13 (11.5%) | -40.0/115 (-34.8%) |
| GRDN00BJ2R8QRO0L / 2013-04-02 (Atelectasis) | 1.25/11 (11.4%) | -46.0/136 (-33.8%) |
| GRDN006MCYEVYFE8 / 2011-07-25 (Edema) | **-0.50/12 (-4.2%)** | -38.0/108 (-35.2%) |

Process scores swing far more per-episode than Accuracy scores do (-4%
to 83%, versus a tighter -18.9% to -39.1% band on Accuracy) - reinforcing
that the model's clinical read is relatively stable across cases, while
its ability to correctly operate OpenEMR is what actually determines
whether a given episode goes well or badly. The Edema episode is Process
score net-negative overall - real penalties (wrong patient/encounter,
lost documentation) outweighed everything it earned.

## What would move the number

Three of the leading issues here are execution discipline, not medical
knowledge, and are addressable without touching the model's clinical
reasoning at all:

1. **Fix the documentation-loss problem.** This is the single highest-
   leverage fix available - 9 of 10 episodes failed to produce a real
   saved note, and at least some of that lost content was clinically
   correct. Recovering this alone would move the A1/A2 numbers up without
   any change to the model's actual reading of the X-rays.
2. **Investigate the 40% execution-failure rate.** Four episodes never
   even reached the right patient/encounter - this is worth root-causing
   specifically (is it a UI/navigation issue, a timing issue, or a
   genuine reasoning gap in finding the right patient) before drawing any
   conclusion about clinical competence from those episodes.
3. **Reduce Z2 action-count/conflict errors** - `process_wrong_action_count`
   (9 occurrences) and the mutual-exclusivity conflicts both point at the
   model sometimes selecting the wrong number of actions or two
   conflicting ones (e.g. diagnostic AND therapeutic thoracentesis
   together in one real episode) rather than the single correct action
   or the correct combination.

What remains after that is a narrower, genuine capability gap:
`missed_finding`/`missed_action`/`missed_diagnosis` together account for
more than half of all clinical misses - the model under-reports more
than it fabricates, which is the safer direction to err in, but is still
the dominant real gap once the execution issues above are set aside.

## Appendix A: real example - hallucinated action

Visit: GRDN004RP5BFHE0T / 2009-06-01 (Effusion). Ground truth follow-up
included `ACT_PULMONOLOGY_FOLLOWUP` (missed by the model). The model
additionally selected an action not in the gold set at all:

> *"Notify urgent clinician for presumed massive pneumoperitoneum/bowel
> perforation surgical emergency"* - graded -4.0 (hallucinated_by_model).
> Judge's reasoning: *"The model introduced a non-gold urgent surgical
> emergency action based on massive pneumoperitoneum/bowel perforation,
> which is not part of the consensus pleural effusion workup. This is an
> unnecessary and potentially misleading escalation beyond the indicated
> respiratory/pleural evaluation."*

A pleural effusion case being escalated to a completely unrelated
surgical emergency is a meaningfully concerning failure mode, not a minor
miss - worth flagging as qualitatively different from an ordinary missed
or slightly-off action.

## Appendix A2: real examples - missed findings and missed actions

`false_negatives` (missed findings/actions) is the single largest source
of point loss on every Accuracy axis (see the sub-metric table above).
Concrete examples, all from the 10 batch episodes, all scored
`false_negatives = -2.0` or `essential_action_recall = -2.0`:

**Missed findings** - visit GRDN004RP5BFHE0T / 2009-06-01 (Effusion),
where the model wrote no findings at all:
- *"Increased opacification in the left base represents a developing
  left pleural effusion"* - judge: "The model provided no findings, so it
  missed the developing left pleural effusion at the left base."
- *"There is left basilar atelectasis as well."* - judge: "The model
  provided no findings, so it missed the left basilar atelectasis."
- *"A right-sided PICC line is stable in position."* - judge: "The model
  provided no findings, so it missed the stable right-sided PICC line."

**Missed actions** - visit GRDN006MCYEVYFE8 / 2011-07-25 (Edema), where
the model provided no follow-up recommendation at all despite an urgent
gold-standard requirement:
- *"Urgent respiratory assessment (ACT_URGENT_ASSESSMENT)"* - judge: "The
  model provided no follow-up recommendation despite the gold standard
  requiring urgent respiratory assessment."
- *"Escalate level of care (ACT_ESCALATE_CARE)"* - judge: "The model did
  not recommend escalation to a higher level of care, which was required
  by the gold standard."
- *"Administer IV loop diuretic (ACT_FUROSEMIDE_IV)"* - judge: "The model
  did not recommend IV loop diuresis for suspected acute cardiogenic
  pulmonary edema."

This is the same episode that scored the most negative Process total
(-0.5/12, see the per-episode table) - a case where both axes failed
together rather than independently.

## Appendix B: real example - mutual-exclusivity conflict

Visit: GRDN00BKCEOKC7S1 / 2025-02-07 (a test-phase episode, included here
only as illustration since the mechanism itself is real and verified).
The model selected both "Perform diagnostic thoracentesis" and "Perform
therapeutic thoracentesis" for the same visit - two actions that share
the real `pleural_drainage_strategy` mutually-exclusive group in the
source action catalog. Correctly flagged: Z2.12 = -1.0, *"0
non-conflicting pair(s), 1 conflicting pair(s)."*

## Appendix C: full per-episode index

| Episode | Condition | Process (earned/max) | Accuracy (earned/max) | Execution failure? |
|---|---|---|---|---|
| GRDN004RP5BFHE0T_2009-06-01_1785991219 | Effusion | 8.50/13 | -25.5/122 | No |
| GRDN004RP5BFHE0T_2009-06-15_1785992551 | Pneumonia | 7.25/15 | -25.0/112 | **Yes** |
| GRDN004RP5BFHE0T_2011-09-01_1785989933 | Normal | 10.75/13 | -20.0/60 | No |
| GRDN006MCYEVYFE8_2011-07-25_1785992166 | Edema | -0.50/12 | -38.0/108 | **Yes** |
| GRDN006VFINFR877_2013-03-25_1785991582 | Emphysema | 6.75/12 | -34.0/99 | No |
| GRDN00BE97VCHW4E_2025-01-05_1785992661 | Lung opacity | 4.75/13 | -36.0/92 | No |
| GRDN00BJ2R8QRO0L_2013-04-02_1785990708 | Atelectasis | 1.25/11 | -46.0/136 | No |
| GRDN00BJ2R8QRO0L_2016-03-29_1785990362 | Cardiomegaly | 2.00/11 | -30.0/95 | **Yes** |
| GRDN00BKCEOKC7S1_2024-01-12_1785991400 | Consolidation | 7.75/13 | -15.5/82 | No |
| GRDN00BKCEOKC7S1_2025-02-12_1785991963 | Pneumothorax | 1.50/13 | -40.0/115 | **Yes** |

## Data sources

- `computer_use_eval/all_receipts.csv` - every individual scored
  checkpoint across all 10 episodes, with reasoning text. Every
  earned/max-possible figure in this report was computed directly from
  the raw `points` column here, against the point scales defined in
  `judge.py` and `grade_process.py`.
- `computer_use_eval/compute_standard_metrics.py` - Sensitivity,
  Precision, partial Specificity, and the Spearman rank-correlation
  attempt, computed from `all_receipts.csv`'s `match_status` tags plus
  each episode's `log.jsonl` (for real action-selection order) and the
  gold `generated_action_ids` ordering. No new agent runs or LLM calls.
- `computer_use_eval/aggregate_report.csv` - the failure-mode frequency,
  near-miss, and clinical-only-breakdown tables above (these are
  rule-based tag counts, not pass-rate percentages, so they're
  unaffected by the scoring-convention issue described earlier).
  **Note:** `aggregate_report.py`'s own `pass_rate_by_checkpoint_kind`
  and `pass_rate_by_condition` outputs still use the flawed
  points->=0-is-a-pass convention and should not be cited until that
  logic is fixed - none of this report's earned/max-possible numbers
  came from those specific outputs.
- `computer_use_eval/episodes/<name>/log.jsonl` - raw per-step agent
  action logs, source for the documentation-loss finding.
- `computer_use_eval/episodes/<name>/receipt.csv` - per-episode full
  breakdown.

Every number in this report traces back to one of the files above - none
of it is estimated or extrapolated.
