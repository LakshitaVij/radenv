"""
grade_process.py

Process Axis (Z1/Z2) grader for one agent_episode.py run. Reads a
completed episode's log.jsonl and computes the exact point rules from the
Process Axis design doc.

Design decisions confirmed with the user before writing this (see plan
doc "Marry Process Axis and Accuracy Axis into one graded episode"):
  - Z1 steps 7-8 (engages vitals/history): approximated as "reached the
    correct encounter page AND performed at least one further action
    there" - there's no way to detect a click landed on the specific
    Vitals vs. History panel from the log alone. Both steps use the same
    proxy since they live on the same page and can't be told apart.
  - Z2 steps 9-10 (navigate Procedures / Configuration): same limitation -
    both collapse into "did the log ever show a URL for Configure Orders
    and Results (types.php)", since the intermediate Procedures menu
    click isn't a separate page.
  - Z2 step 11 (valid action): scored PER selected action, summed - not
    one flat episode-level verdict. Consistent with steps 12-13, which
    are explicitly per-pair.
  - Z2 steps 12-13 ("too many/too few actions"): compared against THIS
    VISIT's actual gold action count (via oracle.py), not a fixed
    universal band - real visits have 1-4 gold actions, not always 1-3.
  - Z2 step 14 (abstention calibration): "should abstain" ground truth =
    gold action list is empty OR contains ACT_ABSTAIN/
    ACT_CLINICAL_CORRELATION. Same definition used in judge.py's A3
    abstention handling so Process and Accuracy don't quietly disagree.
  - Z1 range [1,8] and Z2 range [-3,7] are NOT clamped - computed as
    literal raw sums per the point rules, same policy as the Accuracy
    Axis (A3's stated range didn't match its own metrics' raw sum either;
    the user confirmed "don't force-fit" applies here too). Z2 in
    particular scales with how many actions were selected, so a 4-action
    episode can legitimately exceed +7.

Ground truth patient/encounter comes from a direct DB query (same
`docker compose exec mysql` pattern used throughout this session), not
hardcoded - so this grader stays correct if the seed data changes.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import openpyxl

COMPOSE_DIR = Path("/Users/lakshitavij/synthetic pipeline/openemr/docker/development-easy")
ACTION_LIBRARY_XLSX = Path("/Users/lakshitavij/ChartR-ClinicalRL/pulmonary_action_library_with_2026_guideline_rewards.xlsx")

# The 3 actions added this session that aren't in the xlsx - same
# can_combine/mutually_exclusive_group defaults used when they were
# registered into OpenEMR (see seed_action_hierarchy.py / the plan doc).
_NEW_ACTIONS_META = {
    "ACT_REPOSITION_LINE": {"name": "Reposition Or Retract Central Line/Catheter", "can_combine": True, "group": None},
    "ACT_MANAGE_PLEURAL_CATHETER": {"name": "Manage Indwelling Pleural Catheter", "can_combine": True, "group": None},
    "ACT_OPTIMIZE_BP": {"name": "Optimize Blood Pressure/Hemodynamics", "can_combine": True, "group": None},
}

ABSTAIN_ACTION_IDS = {"ACT_ABSTAIN", "ACT_CLINICAL_CORRELATION"}


def _humanize(action_id: str) -> str:
    return " ".join(w.capitalize() for w in action_id.replace("ACT_", "").split("_"))


def _load_action_catalog() -> dict[str, dict]:
    """action_id -> {name, can_combine, group}, for matching select_action's
    human-readable level_2 labels back to the real catalog."""
    catalog = {}
    if ACTION_LIBRARY_XLSX.exists():
        wb = openpyxl.load_workbook(ACTION_LIBRARY_XLSX, data_only=True)
        ws = wb["Action Library"]
        rows = list(ws.iter_rows(min_row=1, values_only=True))
        header = rows[0]
        for r in rows[1:]:
            d = dict(zip(header, r))
            if d.get("action_id"):
                catalog[d["action_id"]] = {
                    "name": d["action_name"],
                    "can_combine": bool(d.get("can_combine", True)),
                    "group": d.get("mutually_exclusive_group") or None,
                }
    for action_id, meta in _NEW_ACTIONS_META.items():
        catalog[action_id] = meta
    return catalog


_CATALOG = _load_action_catalog()
_NAME_TO_ID = {meta["name"].strip().lower(): action_id for action_id, meta in _CATALOG.items()}
_NAME_TO_ID.update({_humanize(action_id).strip().lower(): action_id for action_id in _CATALOG})


def _match_action_id(level_2_label: str) -> str | None:
    """Map select_action's on-screen label back to a real action_id - exact
    match first, falling back to a normalized/loose match since the agent
    may not reproduce the label byte-for-byte."""
    label = (level_2_label or "").strip().lower()
    if label in _NAME_TO_ID:
        return _NAME_TO_ID[label]
    for name, action_id in _NAME_TO_ID.items():
        if name in label or label in name:
            return action_id
    return None


def _run_sql(sql: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mysql", "mariadb", "-uroot", "-proot", "openemr", "-e", sql],
        cwd=COMPOSE_DIR, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _all_urls(page_state: dict) -> str:
    """OpenEMR's UI is frame-based - the top-level page stays on main.php
    while patient/encounter/Configure Orders content loads in iframes
    (confirmed via a real run where page.url() showed main.php for 12
    consecutive steps while the agent visibly navigated internally).
    Search every frame's URL, not just the top-level one."""
    return " ".join([page_state.get("url", "")] + page_state.get("frame_urls", []))


def ground_truth_pid_encounter(patient_id: str, visit_date: str) -> tuple[int | None, int | None]:
    """Real (pid, encounter_id) for this patient's visit, via direct DB
    query - not hardcoded, so this stays correct if seed data changes."""
    out = _run_sql(
        "SELECT pd.pid, fe.encounter FROM patient_data pd "
        "JOIN form_encounter fe ON fe.pid=pd.pid "
        f"WHERE pd.lname='{patient_id}' AND DATE(fe.date)='{visit_date}';"
    )
    lines = [l for l in out.strip().splitlines()[1:] if l.strip()]
    if not lines:
        return None, None
    pid, encounter = lines[0].split("\t")
    return int(pid), int(encounter)


# Real OpenEMR note-type forms an agent might legitimately choose to
# document in, now that the prompt no longer tells it which one to use -
# confirmed real, via each form's own table.sql (not guessed). Each entry:
# (real table name, the text field(s) worth reading as "the note").
NOTE_FORM_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "soap": ("form_soap", ["subjective", "objective", "assessment", "plan"]),
    "clinical_notes": ("form_clinical_notes", ["description"]),
    "clinic_note": ("form_clinic_note", ["history", "examination", "plan"]),
    "note": ("form_note", ["message"]),
    "clinical_instructions": ("form_clinical_instructions", ["instruction"]),
}


def read_all_note_forms(patient_id: str, visit_date: str) -> dict[str, dict[str, str]]:
    """Every real note-type form saved for this patient's encounter, across
    all the forms an agent might have legitimately chosen to use - not just
    one assumed form type. Returns {formdir: {field: text}} for whichever
    forms actually have a saved row; forms with nothing saved are omitted."""
    true_pid, true_encounter = ground_truth_pid_encounter(patient_id, visit_date)
    if true_pid is None:
        return {}
    found = {}
    for formdir, (table, fields) in NOTE_FORM_REGISTRY.items():
        field_list = ", ".join(f"s.{f}" for f in fields)
        try:
            out = _run_sql(
                f"SELECT {field_list} FROM forms f JOIN {table} s ON s.id = f.form_id "
                f"WHERE f.pid={true_pid} AND f.encounter={true_encounter} AND f.formdir='{formdir}' "
                "ORDER BY f.id DESC LIMIT 1;"
            )
        except subprocess.CalledProcessError:
            continue  # table doesn't exist in this OpenEMR install - skip, not fatal
        lines = [l for l in out.strip().splitlines()[1:] if l.strip() or l == ""]
        if not lines:
            continue
        values = lines[0].split("\t")
        field_map = dict(zip(fields, values))
        if any(v.strip() for v in field_map.values()):
            found[formdir] = field_map
    return found


_IMPRESSION_SPLIT_RE = re.compile(r"impression\s*:", re.IGNORECASE)
_FOLLOWUP_SPLIT_RE = re.compile(r"follow-?up\s*:", re.IGNORECASE)
_FINDINGS_LABEL_RE = re.compile(r"^findings\s*:\s*", re.IGNORECASE)


def split_three(text: str) -> tuple[str, str, str]:
    """Splits one free-text block into (findings, impression, follow_up)
    using literal 'Findings:'/'Impression:'/'Follow-up:' markers - the
    format the system prompt explicitly asks the agent to use. Whatever
    the agent actually labels gets graded; nothing else does - simpler and
    safer than any length-based heuristic (no risk of a login credential
    or patient ID slipping in, since those are never typed under one of
    these labels). Shared by grade_process.py's Process check and
    grade_episode.py's Accuracy extraction, so both use identical logic."""
    imp_m = _IMPRESSION_SPLIT_RE.search(text)
    fu_m = _FOLLOWUP_SPLIT_RE.search(text)

    if imp_m and fu_m and fu_m.start() > imp_m.start():
        findings = text[:imp_m.start()]
        impression = text[imp_m.end():fu_m.start()]
        follow_up = text[fu_m.end():]
    elif imp_m and not fu_m:
        findings = text[:imp_m.start()]
        impression = text[imp_m.end():]
        follow_up = ""
    elif fu_m and not imp_m:
        findings = text[:fu_m.start()]
        impression = ""
        follow_up = text[fu_m.end():]
    elif imp_m and fu_m:  # follow-up marker appears before impression marker - unusual order, still honor both
        findings = text[:min(imp_m.start(), fu_m.start())]
        impression = text[imp_m.end():] if imp_m.start() > fu_m.start() else text[imp_m.end():fu_m.start()]
        follow_up = text[fu_m.end():] if fu_m.start() > imp_m.start() else text[fu_m.end():imp_m.start()]
    else:
        findings, impression, follow_up = "", "", ""  # no marker at all - nothing labeled as documentation, grade nothing
    findings = _FINDINGS_LABEL_RE.sub("", findings.strip())
    return findings.strip(), impression.strip(), follow_up.strip()


def all_typed_text(entries: list[dict]) -> str:
    """Every type_text call's text, concatenated in call order - no length
    filtering, no exclusion list. Safe because split_three() only grades
    content that appears after an explicit 'Findings:'/'Impression:'/
    'Follow-up:' marker - a login credential or patient ID typed into an
    unrelated field is never labeled that way, so it's never graded,
    without needing to guess at it via length heuristics."""
    return "\n\n".join(
        e["args"]["text"] for e in entries
        if e.get("type") == "step" and e.get("tool") == "type_text"
    )


def score_documentation_saved(entries: list[dict], saved_notes: dict[str, dict[str, str]]) -> ScoreItem | None:
    """New check, per explicit user request: if the agent typed real
    documentation (labeled Findings:/Impression:/Follow-up:) but none of
    it ended up in any real saved note form, that's a real failure - it
    believed it documented something and didn't. -0.25 for that; +1 if
    labeled content was typed AND found saved; None (not scored) if no
    Findings:/Impression:/Follow-up: label was ever typed at all, since
    this check is specifically about typed-but-lost content, not about
    whether documentation was attempted."""
    findings, impression, follow_up = split_three(all_typed_text(entries))
    typed_sections = [s for s in (findings, impression, follow_up) if s]
    if not typed_sections:
        return None

    saved_text_blob = " ".join(
        v for field_map in saved_notes.values() for v in field_map.values()
    ).lower()

    def _found(text: str) -> bool:
        # A real save may have been edited/retyped slightly - require a
        # solid substring match (first 40 chars) rather than exact equality.
        needle = text.strip().lower()[:40]
        return bool(needle) and needle in saved_text_blob

    if any(_found(t) for t in typed_sections):
        return ScoreItem("Documentation actually saved", 1, "Typed labeled documentation (Findings/Impression/Follow-up) and it was found in a real saved note form.")
    return ScoreItem("Documentation actually saved", -0.25,
                      "Typed labeled documentation (Findings/Impression/Follow-up) but none of it was found in any saved note form - "
                      "likely typed into a form and never clicked Save, or navigated away before saving.")


def load_episode_log(log_path: Path) -> list[dict]:
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class ScoreItem:
    def __init__(self, step: str, points: float, note: str):
        self.step = step
        self.points = points
        self.note = note

    def to_row(self, axis: str) -> dict:
        return {"axis": axis, "item": self.step, "points": self.points, "note": self.note}


def score_z1(entries: list[dict]) -> list[ScoreItem]:
    steps = [e for e in entries if e.get("type") == "step"]
    tab_switches = [e for e in entries if e.get("type") == "tab_switch"]
    page_states = [e for e in entries if e.get("type") == "page_state"]

    items = []

    items.append(ScoreItem("Z1.1 Opens X-ray", 1, "Episode always starts on the X-ray viewer."))

    first_switch_step = tab_switches[0]["step"] if tab_switches else None
    interacted = any(
        e["tool"] == "scroll" and (first_switch_step is None or e["step"] < first_switch_step)
        for e in steps
    )
    items.append(ScoreItem("Z1.2 Interacts (zoom/pan)", 1 if interacted else 0,
                            "scroll tool call before leaving the X-ray viewer." if interacted else "No scroll/zoom before navigating away."))

    clicked_openemr = len(tab_switches) > 0
    items.append(ScoreItem("Z1.3 Clicks View in OpenEMR", 1 if clicked_openemr else 0,
                            "tab_switch event present." if clicked_openemr else "Never opened OpenEMR."))

    typed_texts = [e["args"].get("text", "").lower() for e in steps if e.get("tool") == "type_text"]
    logged_in = any("admin" in t for t in typed_texts) and any("pass" in t for t in typed_texts)
    items.append(ScoreItem("Z1.4 Logs in", 1 if logged_in else 0,
                            "Typed both username and password." if logged_in else "Login credentials not both entered."))

    return items, page_states


def score_z1_patient_encounter(page_states: list[dict], true_pid: int | None, true_encounter: int | None) -> list[ScoreItem]:
    items = []
    if true_pid is None:
        items.append(ScoreItem("Z1.5 Correct patient", 0, "No ground-truth encounter found for this visit - unverifiable."))
        items.append(ScoreItem("Z1.6 Correct encounter", 0, "No ground-truth encounter found for this visit - unverifiable."))
        return items

    pid_pattern = re.compile(rf"set_pid=0*{true_pid}\b")
    wrong_pid_pattern = re.compile(r"set_pid=(\d+)")
    reached_correct_pid = any(pid_pattern.search(_all_urls(ps)) for ps in page_states)
    reached_wrong_pid = any(
        (m := wrong_pid_pattern.search(_all_urls(ps))) and int(m.group(1)) != true_pid
        for ps in page_states
    )
    if reached_correct_pid:
        items.append(ScoreItem("Z1.5 Correct patient", 1, f"Reached set_pid={true_pid}."))
    elif reached_wrong_pid:
        items.append(ScoreItem("Z1.5 Correct patient", -0.25, "Reached a different patient than the correct one."))
    else:
        items.append(ScoreItem("Z1.5 Correct patient", 0, "Never reached any patient page (unverifiable from URLs)."))

    enc_pattern = re.compile(rf"set_encounter=0*{true_encounter}\b")
    wrong_enc_pattern = re.compile(r"set_encounter=(\d+)")
    reached_correct_enc = any(enc_pattern.search(_all_urls(ps)) for ps in page_states)
    reached_wrong_enc = any(
        (m := wrong_enc_pattern.search(_all_urls(ps))) and int(m.group(1)) != true_encounter
        for ps in page_states
    )
    if reached_correct_enc:
        items.append(ScoreItem("Z1.6 Correct encounter", 1, f"Reached set_encounter={true_encounter}."))
    elif reached_wrong_enc:
        items.append(ScoreItem("Z1.6 Correct encounter", -0.25, "Reached a different encounter than the correct one."))
    else:
        items.append(ScoreItem("Z1.6 Correct encounter", 0, "Never reached the encounter page (unverifiable from URLs)."))

    return items


def score_z1_vitals_history(page_states: list[dict], entries: list[dict], reached_encounter: bool) -> list[ScoreItem]:
    """Proxy per the confirmed design: reached the encounter page AND at
    least one further action happened there. Cannot distinguish clicking
    Vitals specifically from History specifically - both scored the same."""
    items = []
    if not reached_encounter:
        for label in ("Z1.7 Engages vitals", "Z1.8 Engages history"):
            items.append(ScoreItem(label, -0.25, "Encounter page never reached, so this panel was never clicked."))
        return items

    steps_after = [e for e in entries if e.get("type") == "step" and e["step"] > 0]
    engaged = len(steps_after) >= 2  # reached encounter + at least one more real action
    for label in ("Z1.7 Engages vitals", "Z1.8 Engages history"):
        items.append(ScoreItem(
            label, 1 if engaged else 0,
            "Reached encounter and took further action (proxy - can't isolate Vitals vs. History panel from the log)." if engaged
            else "Reached encounter but no further engagement detected.",
        ))
    return items


def score_z2_navigation(page_states: list[dict]) -> list[ScoreItem]:
    """Both 'Navigate to Procedures' and 'Navigate to Configuration' collapse
    into one check - reaching types.php - since the intermediate Procedures
    menu click isn't a separate page we can detect."""
    reached = any("types.php" in _all_urls(ps) for ps in page_states)
    items = []
    for label in ("Z2.9 Navigate to Procedures", "Z2.10 Navigate to Configuration"):
        items.append(ScoreItem(
            label, 1 if reached else 0,
            "Reached Configure Orders and Results." if reached else "Never reached Configure Orders and Results.",
        ))
    return items


def score_z2_actions(selected_actions: list[dict], gold_action_ids: list[str]) -> list[ScoreItem]:
    items = []
    matched_ids = []
    for sa in selected_actions:
        action_id = _match_action_id(sa.get("level_2", ""))
        matched_ids.append(action_id)
        if action_id is not None:
            items.append(ScoreItem("Z2.11 Select valid action", 1, f"'{sa.get('level_2')}' matches real catalog entry {action_id}."))
        else:
            items.append(ScoreItem("Z2.11 Select valid action", -1, f"'{sa.get('level_2')}' does not match any real catalog action."))

    valid_ids = [a for a in matched_ids if a is not None]

    # Z2.12: mutual exclusivity, per pair among valid selected actions.
    conflicts = 0
    valid_pairs = 0
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            g1, g2 = _CATALOG.get(valid_ids[i], {}).get("group"), _CATALOG.get(valid_ids[j], {}).get("group")
            if g1 and g2 and g1 == g2:
                conflicts += 1
            else:
                valid_pairs += 1
    if valid_pairs or conflicts:
        items.append(ScoreItem("Z2.12 No mutual exclusivity conflicts", valid_pairs * 1 + conflicts * -1,
                                f"{valid_pairs} non-conflicting pair(s), {conflicts} conflicting pair(s)."))

    # Z2.13: can_combine, per pair among valid selected actions.
    combine_ok = 0
    combine_bad = 0
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            c1, c2 = _CATALOG.get(valid_ids[i], {}).get("can_combine", True), _CATALOG.get(valid_ids[j], {}).get("can_combine", True)
            if c1 and c2:
                combine_ok += 1
            else:
                combine_bad += 1
    if combine_ok or combine_bad:
        items.append(ScoreItem("Z2.13 Can-combine valid", combine_ok * 1 + combine_bad * -1,
                                f"{combine_ok} valid-combination pair(s), {combine_bad} invalid-combination pair(s)."))

    # "Too many/too few" relative to THIS visit's real gold count.
    gold_n, model_n = len(gold_action_ids), len(valid_ids)
    if model_n > gold_n:
        extra = model_n - gold_n
        items.append(ScoreItem("Z2.15 Action count calibration", extra * -0.25, f"Selected {model_n} actions vs. {gold_n} expected - {extra} too many."))
    elif model_n < gold_n and gold_n > 0:
        missing = gold_n - model_n
        items.append(ScoreItem("Z2.15 Action count calibration", missing * -0.75, f"Selected {model_n} actions vs. {gold_n} expected - {missing} too few."))

    return items, valid_ids


def score_z2_abstention(valid_ids: list[str], gold_action_ids: list[str]) -> ScoreItem:
    gold_says_abstain = len(gold_action_ids) == 0 or any(a in ABSTAIN_ACTION_IDS for a in gold_action_ids)
    model_abstained = len(valid_ids) == 0 or any(a in ABSTAIN_ACTION_IDS for a in valid_ids)

    if gold_says_abstain and model_abstained:
        return ScoreItem("Z2.14 Abstention calibration", 1, "Gold expected abstention, agent abstained.")
    if not gold_says_abstain and model_abstained:
        return ScoreItem("Z2.14 Abstention calibration", -2, "Gold expected real action(s), agent abstained instead.")
    if gold_says_abstain and not model_abstained:
        return ScoreItem("Z2.14 Abstention calibration", 0.25, "Gold expected abstention, agent acted anyway.")
    return ScoreItem("Z2.14 Abstention calibration", 1, "Gold expected real action(s), agent acted.")


STEP_EFFICIENCY_FIXED_OVERHEAD = 15
# 1 X-ray interaction (zoom/pan) + 1 click "View in OpenEMR" + 2 type login
# credentials (username, password) + 1 submit login + 1 find/reach the
# correct patient + 1 find/reach the correct encounter + 1 engage vitals +
# 1 engage history + 1 navigate to a note form + 1 type documentation +
# 1 click Save + 1 navigate to Procedures + 1 navigate to Configuration/
# Configure Orders and Results + 1 finish = 15. Interaction with the X-ray
# is treated as a required step, not optional - per explicit user
# decision, since a real clinical read requires actually inspecting the
# image, not just glancing at the default view. This is a transparent,
# fixed floor - not a claim about the exact optimal click path (that
# depends on OpenEMR's UI in ways not verifiable from the log alone) -
# plus one select_action call per gold action for this visit (minimum 1,
# even for an abstention decision).

STEP_EFFICIENCY_INTERACTION_BUFFER = 2
# Extra allowance on top of the 1 required interaction step, since a real
# read of the X-ray plausibly needs more than one zoom/pan (e.g. checking
# multiple regions) - this buffer absorbs that without penalizing
# reasonable additional interaction as "wasted steps."


def _episode_fully_completed(process_items: list[ScoreItem]) -> bool:
    """'Completed fully end to end, without skipping anything' - stricter
    than a positive Z1/Z2 point total (which can stay positive even with a
    real gap, e.g. several correct action selections offsetting one bad
    one). Every checkpoint below must have cleanly passed."""
    by_step: dict[str, list[float]] = {}
    for item in process_items:
        by_step.setdefault(item.step, []).append(item.points)

    def _all_positive(step_name: str) -> bool:
        return step_name in by_step and all(p > 0 for p in by_step[step_name])

    required_clean = [
        "Z1.2 Interacts (zoom/pan)",
        "Z1.5 Correct patient", "Z1.6 Correct encounter",
        "Z1.7 Engages vitals", "Z1.8 Engages history",
        "Z2.9 Navigate to Procedures", "Z2.10 Navigate to Configuration",
    ]
    if not all(_all_positive(s) for s in required_clean):
        return False
    if any(p < 0 for p in by_step.get("Z2.11 Select valid action", [])):
        return False
    if "Z2.15 Action count calibration" in by_step:  # any row here means the count didn't match exactly
        return False
    doc_points = by_step.get("Documentation actually saved", [])
    if not doc_points or doc_points[0] != 1:
        return False
    return True


def score_step_efficiency(entries: list[dict], process_items: list[ScoreItem], gold_action_ids: list[str]) -> ScoreItem:
    """New check, per explicit user request: reward completing the full
    required workflow in the fewest steps, penalize unnecessary extra
    steps. Full +1 credit only if the episode both (a) completed the
    entire required workflow with no skipped/failed checkpoint (per
    _episode_fully_completed) and (b) took no more than the minimum step
    count. -0.25 per step beyond the minimum applies regardless of
    completion - unnecessary steps are penalized either way."""
    actual_steps = len([e for e in entries if e.get("type") == "step"])
    min_actions = max(1, len(gold_action_ids))
    minimum_steps = STEP_EFFICIENCY_FIXED_OVERHEAD + STEP_EFFICIENCY_INTERACTION_BUFFER + min_actions
    extra_steps = max(0, actual_steps - minimum_steps)
    penalty = round(-0.25 * extra_steps, 2)

    fully_completed = _episode_fully_completed(process_items)
    bonus = 1 if (fully_completed and extra_steps == 0) else 0
    points = round(bonus + penalty, 2)

    note = (
        f"{actual_steps} actual step(s) vs. {minimum_steps} minimum "
        f"({STEP_EFFICIENCY_FIXED_OVERHEAD} fixed incl. required X-ray interaction + "
        f"{STEP_EFFICIENCY_INTERACTION_BUFFER} interaction buffer + {min_actions} for "
        f"{len(gold_action_ids)} gold action(s)). "
        f"{'Fully completed' if fully_completed else 'NOT fully completed (skipped/failed a required checkpoint)'}, "
        f"{extra_steps} step(s) over minimum."
    )
    return ScoreItem("Z2.16 Step efficiency", points, note)


def grade_process(log_path: Path, patient_id: str, visit_date: str, gold_action_ids: list[str]) -> list[ScoreItem]:
    entries = load_episode_log(log_path)
    z1_core, page_states = score_z1(entries)

    true_pid, true_encounter = ground_truth_pid_encounter(patient_id, visit_date)
    z1_pe = score_z1_patient_encounter(page_states, true_pid, true_encounter)
    reached_encounter = any(i.points == 1 for i in z1_pe if i.step == "Z1.6 Correct encounter")
    z1_vh = score_z1_vitals_history(page_states, entries, reached_encounter)

    z2_nav = score_z2_navigation(page_states)
    selected_actions = [e for e in entries if e.get("type") == "select_action"]
    z2_actions, valid_ids = score_z2_actions(selected_actions, gold_action_ids)
    z2_abstain = score_z2_abstention(valid_ids, gold_action_ids)

    saved_notes = read_all_note_forms(patient_id, visit_date)
    doc_saved_item = score_documentation_saved(entries, saved_notes)
    doc_items = [doc_saved_item] if doc_saved_item else []

    all_items = z1_core + z1_pe + z1_vh + z2_nav + z2_actions + [z2_abstain] + doc_items
    efficiency_item = score_step_efficiency(entries, all_items, gold_action_ids)

    return all_items + [efficiency_item]


if __name__ == "__main__":
    import sys
    log_path = Path(sys.argv[1])
    patient_id = sys.argv[2]
    visit_date = sys.argv[3]
    gold_ids = sys.argv[4].split(";") if len(sys.argv) > 4 and sys.argv[4] else []

    items = grade_process(log_path, patient_id, visit_date, gold_ids)
    total = sum(i.points for i in items)
    for i in items:
        print(f"  {i.points:+.2f}  {i.step}: {i.note}")
    print(f"\nProcess total: {total:+.2f}")
