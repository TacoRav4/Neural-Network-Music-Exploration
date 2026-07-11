# HANDOFF_CURRENT.md — Read this first, on any machine, any session

**This is the evergreen entry point.** Update it at each checkpoint (it should always describe "where things stand right now," not any one historical phase). `HANDOFF_PHASE3G.md` is a frozen, historical entry point written specifically for Phase 3G — still useful for deep context on the 6-level benchmark ladder and Phase 2C/2D/3B/3C's hardcoded-script caveat, but no longer the first thing to read. Full detail on every phase lives in `STATUS.md`; this file is a compressed pointer into it, written so a fresh session can orient in one read instead of exploring the repo.

**Checkpoint date: 2026-07-11. Last completed phase: 3J-B.**

## What this project is

A COGS 202/269 continuation: non-neural, fully-interpretable tonal (key/mode) inference over MIDI, built up in small, falsifiable, always-reversible steps. Everything lives under what is now the repo root (originally `07_Recurrent_Tonal_Inference/` inside a larger folder — that outer folder and its original COGS 202 root files are NOT part of this repo and were never touched by any of this work).

## Read order for a new session (minimize exploration, maximize signal)

1. **This file** (you're here).
2. `STATUS.md` §6 ("Next approved step") and §7 ("Not yet started") — the actual current decision point, in ~2 paragraphs.
3. Only if you need deep justification for a specific phase's numbers: that phase's own `05_Figures_Results/PHASE*_report.md`. Don't read all of them up front — they're large and the summary below plus STATUS.md's per-phase sections are almost always enough.

Do **not** start by grepping/exploring the codebase broadly. The state is already fully written down.

## Trajectory, ultra-compressed (STATUS.md has the full version of each line)

- **COGS 269/202** (done, frozen): static MLP, then MIDI + hand-coded EMA smoothing. Not current work.
- **Phase 1/1.5** (done): SRN vs. EMA on synthetic then real chord-id sequences. Found the real bottleneck was triadic-chord *representation*, not recurrence.
- **Phase 2** (done): built the non-neural pitch-class/scale-template baseline (`pitch_class_baseline.py`, `SCALE_TEMPLATES`, plain `np.argmax`) — this is the **frozen Stage 1 fast filter** everything since has been evaluating or trying to improve.
- **Phase 3A–3F** (done): staged-architecture design; built and QA'd the 6-level benchmark ladder (L1 Twinkle, L2 Bach, L3 Für Elise excerpt, L4 Chopin, L5 Clementi excerpt, L6 Twinkle 12).
- **Phase 3G-A/B** (done): ran the frozen baseline across all 6 pieces. Finding: it's a reliable **diatonic-collection** fast filter, not a full tonic/mode resolver — minor-key failures (Für Elise, Chopin both 0.0 strict) are a *deterministic* tie-break artifact (relative major/minor share template rows; `np.argmax` always favors the major-indexed key), not noise. Bach's errors are the same mechanism showing up as C/G/D neighborhood confusion. Clementi is a tonic–dominant–tonic excursion, not a monotonic modulation. Chopin's "silence" is a boundary-granularity + smoothing-memory convention effect, not a bug.
- **Phase 3H-A/B/C** (done): tried to fix the tie-break at the timestep level. A weighted key-profile (variant D) recovers minor keys but wrecks monophonic stability (Twinkle, Twinkle 12). Gating it in conservatively (3H-B) protects stability but recovers almost none of the benefit. A predeclared 54-condition Pareto sweep (3H-C) proved there's **no free-lunch region** in that whole gate family — not a tuning failure, a characterization of the family's limits.
- **Phase 3I** (done): synthesis memo. Conclusion: keep Stage 1 frozen, stop tuning timestep-level gates, move to a longer timescale (section/phrase-level resolution).
- **Phase 3J-A** (done): design-only plan for a section-level resolver, reviewed by 3 read-only subagents (architecture / methodology-leakage / music-theory), revised accordingly.
- **Phase 3J-B (done, most recent)**: implemented ONLY the revised Candidate B (mediant + asymmetric raised-leading-tone comparison, aggregated per stable collection segment). **Required verdict: code 3** — recovers minor mode (Für Elise 0.0→0.29, Chopin 0.0→0.50) but damages L6/Twinkle-12 stability unacceptably (pre_384s anchor 1.0→0.0). Post-run review (3 independent read-only subagents on the actual code/results) found a **real, algebra-confirmed mechanism**, not a bug: `minor_third_pc == major_tonic_pc` for every relative-key pair, so the "mediant" cue is structurally confounded with simple tonic-pitch prominence and misfires "minor" on any tonic-heavy major passage. The OTHER cue (raised leading tone) is not confounded and drove 69–84% of Für Elise/Chopin's *genuine* recovery while contributing **0%** of Bach/Clementi/Twinkle-12's spurious minor windows (all 100% mediant-driven artifact there). This decomposition is now the key actionable lead.

## Next decision point (not yet started — needs an explicit go-ahead, per this project's own guardrails)

**Phase 3J-C candidate**: a new, separately-designed (not a retroactive edit of 3J-B) non-neural resolver testing the raised-leading-tone cue *in isolation* (dropping the confounded mediant cue). This is the most promising, well-evidenced next step per Phase 3J-B's own findings, but has not been designed or authorized. Other live options, per Phase 3I/§6-§7 of STATUS.md: a corpus-wide chord-id EMA/SRN disagreement comparison (Phase 3G-C), or moving on to actual neural refinement (Stage 4, still explicitly deferred).

## Standing guardrails (still in force, don't relitigate these)

- Never modify old Phase 2C/2D/3B/3C scripts, or any frozen Phase 3G/3H/3I/3J-A/3J-B output — always write new, phase-labeled scripts/outputs.
- Anchors (expected-key windows) are sparse, **evaluation-only** — never an input to any segmentation/prediction/gating rule.
- No dense per-timestep accuracy claims anywhere (real MIDI has no such ground truth in this project).
- No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement has been run yet anywhere in Phase 3 — all deferred pending an explicit decision.
- Every new script/output/report is prefixed with its phase name; every completed phase gets a STATUS.md section plus a §6/§7 update.
- Constants for any new decision rule must be frozen in writing *before* any anchor comparison, and never adjusted after seeing results (Phase 3H-C and 3J-B's own precedent).

## Repo / environment state

- **Repo root is this folder** (what used to be `07_Recurrent_Tonal_Inference/` — its own internal folder structure, e.g. `04_Recurrent_Implementation/`, `05_Figures_Results/`, `03_MIDI_Data/`, is now the top level of the git repo).
- Remote: `git@github.com:TacoRav4/Neural-Network-Music-Exploration.git`, branch `recurrent-tonal-inference` (already pushed and up to date as of this checkpoint).
- No `requirements.txt` yet. Actual runtime dependencies across all scripts: `numpy`, `matplotlib`, `pretty_midi`, `torch` (torch only for the older Phase 1/1.5 SRN/MLP scripts — not needed for any Phase 3G/3H/3J pitch-class work). Install with `pip install numpy matplotlib pretty_midi torch`.
- Push access on a new machine needs its own auth (SSH key added as a new deploy key, or `gh auth login` / HTTPS + PAT) — the key used previously lives in one machine's SSH agent only.
