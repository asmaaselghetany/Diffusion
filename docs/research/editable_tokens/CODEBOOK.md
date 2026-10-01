# Editable-token Phase 0 — failure taxonomy codebook

**Purpose:** Prove or deny whether the *established* editable-token mechanism
explains **local** IFEval/code gaps under matched AR→block convert (C0 vs U0).
Not a discovery claim.

**Go threshold (frozen):** on C0, local share \((L1 \cup L2) / N_\text{fail} \ge 0.30\).
Do not lower after labeling.

## Labels (pick one primary)

| ID | Name | Definition | Typical IFEval cues |
|----|------|------------|---------------------|
| **L1** | Local span chaos | Mid-response breakage: garbled fragment, self-contradiction in a short span, truncated/mangled phrase that a local rewrite could fix | Nonsense mid-block; broken list item; answer derails then continues |
| **L2** | Local lexical / syntax | Small in-span edit would satisfy the constraint: typo, wrong case on a word, missing bullet marker, stray comma, wrong keyword instance | `punctuation:no_comma` almost followed; one forbidden word; near-miss markdown bullet |
| **G1** | Global constraint | Needs whole-response planning: counts, frequencies, keyword *sets*, language, case of *entire* answer | `length_constraints:*`, `keywords:frequency`, `change_case`, `language:*` |
| **G2** | Document-level format | Global structure / template of the whole document | `combination:repeat_prompt`, full JSON schema, multi-section template |

**Mixed:** set `mixed=yes` if both local and global fail modes are present; still pick the **primary** reason the prompt fails strict IFEval.

## Decision rules (annotator)

1. Read prompt constraints + model response.
2. Ask: “Would rewriting ≤1–2 local spans likely fix *all* failed instructions?” → lean **L***.
3. Ask: “Does success require tracking a global budget (N words, all-caps, keyword inventory)?” → lean **G***.
4. If both: primary = the failed instruction that is **sufficient** for strict fail; flag `mixed`.

## Exemplars

Fill 5 per class from C0/U0 `*_failures.jsonl` (full `log_samples` export
2026-09-18 — `valid_for_go_threshold: true`). Prefer pilot CSVs /
`CLAIM_K_paired_label_sheet.csv`. Do not use old truncated-log dumps.

| Class | key | Notes |
|-------|-----|-------|
| L1 | | |
| L1 | | |
| L1 | | |
| L1 | | |
| L1 | | |
| L2 | | |
| L2 | | |
| L2 | | |
| L2 | | |
| L2 | | |
| G1 | | |
| G1 | | |
| G1 | | |
| G1 | | |
| G1 | | |
| G2 | | |
| G2 | | |
| G2 | | |
| G2 | | |
| G2 | | |

## Artifacts

| File | Role |
|------|------|
| `scripts/analysis/export_ifeval_failures.py` | Score + export failures + pilot CSV |
| `scripts/submit_ifeval_samples.sh` | IFEval-only job with `--log_samples` |
| `C0_1762534_*` / `U0_1849335_*` | Per-cell scored / failures / pilot sheet / summary |
