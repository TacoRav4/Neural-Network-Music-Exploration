# Phase 3H-C — Gate Sensitivity / Pareto Frontier Audit

**This is a diagnostic sensitivity analysis, NOT a new production model and NOT threshold tuning.** It sweeps a predeclared 6x3x3=54-condition grid over Phase 3H-B's three texture gates (weighted-margin threshold, density threshold, collection-stability window) using the exact same label-free gate logic (`compute_active_pc_count`, `compute_collection_stability_mask`, `compute_normalized_margin`, all imported unmodified from Phase 3H-B), to determine whether Phase 3H-B's specific configuration was simply too conservative, or whether no configuration in this space can simultaneously preserve monophonic stability and recover meaningful minor-mode predictions. **No condition below is selected, adopted, recommended, or declared a winner** -- the deliverable is the shape of the trade-off itself.

## Grid definition (fixed before running, not adjusted afterward)

- weighted-margin threshold: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
- density threshold (active_pc_count > this): [1, 2, 3]
- collection-stability window (windows): [2, 4, 6]
- total conditions: 54

## Baseline reference (not swept)

- L1 (Twinkle, C major) strict: A=1.0000, D=0.3585
- L3 (Für Elise, A minor) strict: A=0.0000, D=0.4151
- L4 (Chopin, E minor) strict: A=0.0000, D=0.4074
- L6 pre_384s strict: A=1.0000, D=0.6511
- L6 384_to_432s strict: A=0.8958, D=0.0417
- L6 post_432s strict: A=0.9412, D=0.5451

## Full grid (54 conditions)

`recovery_score` = mean strict-proportion GAIN over A on L3+L4 (higher is better). `damage_score` = mean strict-proportion LOSS below A on L1+L6's three anchors, floored at 0 per-anchor (lower is better). `pareto` = `*` if this condition is on the non-dominated (Pareto) frontier over (recovery, damage), blank otherwise.

| margin | density | stability | swap_rate | L1_strict | L1_switches | L6_pre384 | L6_384to432 | L6_post432 | L3_strict | L4_strict | recovery | damage | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1 | 2 | 0.858 | 0.358 | 45 | 0.696 | 0.052 | 0.586 | 0.340 | 0.338 | 0.339 | 0.536 |  |
| 0.00 | 1 | 4 | 0.804 | 0.358 | 45 | 0.699 | 0.062 | 0.594 | 0.208 | 0.250 | 0.229 | 0.531 |  |
| 0.00 | 1 | 6 | 0.763 | 0.358 | 45 | 0.702 | 0.083 | 0.590 | 0.160 | 0.181 | 0.170 | 0.526 |  |
| 0.00 | 2 | 2 | 0.685 | 0.368 | 46 | 0.910 | 0.146 | 0.869 | 0.340 | 0.338 | 0.339 | 0.386 | * |
| 0.00 | 2 | 4 | 0.633 | 0.368 | 46 | 0.910 | 0.156 | 0.876 | 0.208 | 0.250 | 0.229 | 0.382 |  |
| 0.00 | 2 | 6 | 0.594 | 0.368 | 46 | 0.910 | 0.177 | 0.873 | 0.160 | 0.181 | 0.170 | 0.377 |  |
| 0.00 | 3 | 2 | 0.513 | 0.472 | 41 | 0.955 | 0.385 | 0.873 | 0.340 | 0.282 | 0.311 | 0.288 | * |
| 0.00 | 3 | 4 | 0.465 | 0.472 | 41 | 0.955 | 0.385 | 0.880 | 0.208 | 0.208 | 0.208 | 0.286 |  |
| 0.00 | 3 | 6 | 0.428 | 0.472 | 41 | 0.955 | 0.406 | 0.876 | 0.160 | 0.157 | 0.159 | 0.282 |  |
| 0.02 | 1 | 2 | 0.644 | 0.509 | 41 | 0.856 | 0.219 | 0.659 | 0.321 | 0.292 | 0.306 | 0.399 |  |
| 0.02 | 1 | 4 | 0.602 | 0.509 | 41 | 0.856 | 0.219 | 0.667 | 0.198 | 0.213 | 0.206 | 0.397 |  |
| 0.02 | 1 | 6 | 0.568 | 0.509 | 41 | 0.856 | 0.240 | 0.663 | 0.151 | 0.157 | 0.154 | 0.392 |  |
| 0.02 | 2 | 2 | 0.565 | 0.509 | 41 | 0.919 | 0.250 | 0.878 | 0.321 | 0.292 | 0.306 | 0.320 |  |
| 0.02 | 2 | 4 | 0.522 | 0.509 | 41 | 0.919 | 0.250 | 0.886 | 0.198 | 0.213 | 0.206 | 0.318 |  |
| 0.02 | 2 | 6 | 0.489 | 0.509 | 41 | 0.919 | 0.271 | 0.882 | 0.151 | 0.157 | 0.154 | 0.314 |  |
| 0.02 | 3 | 2 | 0.448 | 0.547 | 39 | 0.964 | 0.458 | 0.882 | 0.321 | 0.236 | 0.278 | 0.246 | * |
| 0.02 | 3 | 4 | 0.408 | 0.547 | 39 | 0.964 | 0.458 | 0.890 | 0.198 | 0.171 | 0.185 | 0.244 |  |
| 0.02 | 3 | 6 | 0.377 | 0.547 | 39 | 0.964 | 0.479 | 0.886 | 0.151 | 0.134 | 0.143 | 0.240 |  |
| 0.05 | 1 | 2 | 0.429 | 0.594 | 37 | 0.937 | 0.438 | 0.678 | 0.217 | 0.227 | 0.222 | 0.297 |  |
| 0.05 | 1 | 4 | 0.395 | 0.594 | 37 | 0.937 | 0.438 | 0.684 | 0.123 | 0.153 | 0.138 | 0.296 |  |
| 0.05 | 1 | 6 | 0.371 | 0.594 | 37 | 0.937 | 0.458 | 0.686 | 0.085 | 0.106 | 0.096 | 0.290 |  |
| 0.05 | 2 | 2 | 0.359 | 0.594 | 37 | 0.976 | 0.448 | 0.898 | 0.217 | 0.227 | 0.222 | 0.230 | * |
| 0.05 | 2 | 4 | 0.325 | 0.594 | 37 | 0.976 | 0.448 | 0.904 | 0.123 | 0.153 | 0.138 | 0.229 |  |
| 0.05 | 2 | 6 | 0.300 | 0.594 | 37 | 0.976 | 0.469 | 0.906 | 0.085 | 0.106 | 0.096 | 0.223 |  |
| 0.05 | 3 | 2 | 0.342 | 0.632 | 35 | 0.976 | 0.490 | 0.898 | 0.217 | 0.171 | 0.194 | 0.210 | * |
| 0.05 | 3 | 4 | 0.311 | 0.632 | 35 | 0.976 | 0.490 | 0.904 | 0.123 | 0.111 | 0.117 | 0.209 |  |
| 0.05 | 3 | 6 | 0.288 | 0.632 | 35 | 0.976 | 0.510 | 0.906 | 0.085 | 0.083 | 0.084 | 0.203 |  |
| 0.10 | 1 | 2 | 0.308 | 0.717 | 25 | 0.964 | 0.854 | 0.686 | 0.142 | 0.157 | 0.149 | 0.154 |  |
| 0.10 | 1 | 4 | 0.290 | 0.717 | 25 | 0.964 | 0.854 | 0.692 | 0.057 | 0.116 | 0.086 | 0.152 |  |
| 0.10 | 1 | 6 | 0.274 | 0.717 | 25 | 0.964 | 0.854 | 0.694 | 0.047 | 0.074 | 0.061 | 0.152 |  |
| 0.10 | 2 | 2 | 0.246 | 0.717 | 25 | 0.982 | 0.865 | 0.904 | 0.142 | 0.157 | 0.149 | 0.092 | * |
| 0.10 | 2 | 4 | 0.228 | 0.717 | 25 | 0.982 | 0.865 | 0.910 | 0.057 | 0.116 | 0.086 | 0.091 |  |
| 0.10 | 2 | 6 | 0.212 | 0.717 | 25 | 0.982 | 0.865 | 0.912 | 0.047 | 0.074 | 0.061 | 0.090 |  |
| 0.10 | 3 | 2 | 0.235 | 0.736 | 25 | 0.982 | 0.885 | 0.904 | 0.142 | 0.111 | 0.126 | 0.082 | * |
| 0.10 | 3 | 4 | 0.218 | 0.736 | 25 | 0.982 | 0.885 | 0.910 | 0.057 | 0.074 | 0.065 | 0.081 | * |
| 0.10 | 3 | 6 | 0.204 | 0.736 | 25 | 0.982 | 0.885 | 0.912 | 0.047 | 0.051 | 0.049 | 0.081 |  |
| 0.15 | 1 | 2 | 0.155 | 0.858 | 14 | 0.988 | 0.906 | 0.906 | 0.038 | 0.093 | 0.065 | 0.047 | * |
| 0.15 | 1 | 4 | 0.151 | 0.858 | 14 | 0.988 | 0.906 | 0.912 | 0.000 | 0.093 | 0.046 | 0.046 |  |
| 0.15 | 1 | 6 | 0.141 | 0.858 | 14 | 0.988 | 0.906 | 0.914 | 0.000 | 0.065 | 0.032 | 0.045 |  |
| 0.15 | 2 | 2 | 0.155 | 0.858 | 14 | 0.988 | 0.906 | 0.906 | 0.038 | 0.093 | 0.065 | 0.047 | * |
| 0.15 | 2 | 4 | 0.151 | 0.858 | 14 | 0.988 | 0.906 | 0.912 | 0.000 | 0.093 | 0.046 | 0.046 |  |
| 0.15 | 2 | 6 | 0.141 | 0.858 | 14 | 0.988 | 0.906 | 0.914 | 0.000 | 0.065 | 0.032 | 0.045 |  |
| 0.15 | 3 | 2 | 0.149 | 0.868 | 12 | 0.988 | 0.906 | 0.906 | 0.038 | 0.065 | 0.051 | 0.045 | * |
| 0.15 | 3 | 4 | 0.145 | 0.868 | 12 | 0.988 | 0.906 | 0.912 | 0.000 | 0.065 | 0.032 | 0.043 | * |
| 0.15 | 3 | 6 | 0.136 | 0.868 | 12 | 0.988 | 0.906 | 0.914 | 0.000 | 0.042 | 0.021 | 0.043 | * |
| 0.20 | 1 | 2 | 0.021 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.009 | 0.019 | 0.014 | 0.004 | * |
| 0.20 | 1 | 4 | 0.021 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |
| 0.20 | 1 | 6 | 0.020 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |
| 0.20 | 2 | 2 | 0.021 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.009 | 0.019 | 0.014 | 0.004 | * |
| 0.20 | 2 | 4 | 0.021 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |
| 0.20 | 2 | 6 | 0.020 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |
| 0.20 | 3 | 2 | 0.020 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.009 | 0.019 | 0.014 | 0.004 | * |
| 0.20 | 3 | 4 | 0.019 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |
| 0.20 | 3 | 6 | 0.018 | 0.991 | 2 | 1.000 | 0.938 | 0.933 | 0.000 | 0.019 | 0.009 | 0.004 |  |

## Pareto frontier (non-dominated conditions)

16 of 54 conditions are non-dominated:

- margin=0.20, density>1, stability=2w: recovery=0.014, damage=0.004, L1_strict=0.991, L3_strict=0.009, L4_strict=0.019, swap_rate=0.021
- margin=0.20, density>2, stability=2w: recovery=0.014, damage=0.004, L1_strict=0.991, L3_strict=0.009, L4_strict=0.019, swap_rate=0.021
- margin=0.20, density>3, stability=2w: recovery=0.014, damage=0.004, L1_strict=0.991, L3_strict=0.009, L4_strict=0.019, swap_rate=0.020
- margin=0.15, density>3, stability=6w: recovery=0.021, damage=0.043, L1_strict=0.868, L3_strict=0.000, L4_strict=0.042, swap_rate=0.136
- margin=0.15, density>3, stability=4w: recovery=0.032, damage=0.043, L1_strict=0.868, L3_strict=0.000, L4_strict=0.065, swap_rate=0.145
- margin=0.15, density>3, stability=2w: recovery=0.051, damage=0.045, L1_strict=0.868, L3_strict=0.038, L4_strict=0.065, swap_rate=0.149
- margin=0.15, density>1, stability=2w: recovery=0.065, damage=0.047, L1_strict=0.858, L3_strict=0.038, L4_strict=0.093, swap_rate=0.155
- margin=0.15, density>2, stability=2w: recovery=0.065, damage=0.047, L1_strict=0.858, L3_strict=0.038, L4_strict=0.093, swap_rate=0.155
- margin=0.10, density>3, stability=4w: recovery=0.065, damage=0.081, L1_strict=0.736, L3_strict=0.057, L4_strict=0.074, swap_rate=0.218
- margin=0.10, density>3, stability=2w: recovery=0.126, damage=0.082, L1_strict=0.736, L3_strict=0.142, L4_strict=0.111, swap_rate=0.235
- margin=0.10, density>2, stability=2w: recovery=0.149, damage=0.092, L1_strict=0.717, L3_strict=0.142, L4_strict=0.157, swap_rate=0.246
- margin=0.05, density>3, stability=2w: recovery=0.194, damage=0.210, L1_strict=0.632, L3_strict=0.217, L4_strict=0.171, swap_rate=0.342
- margin=0.05, density>2, stability=2w: recovery=0.222, damage=0.230, L1_strict=0.594, L3_strict=0.217, L4_strict=0.227, swap_rate=0.359
- margin=0.02, density>3, stability=2w: recovery=0.278, damage=0.246, L1_strict=0.547, L3_strict=0.321, L4_strict=0.236, swap_rate=0.448
- margin=0.00, density>3, stability=2w: recovery=0.311, damage=0.288, L1_strict=0.472, L3_strict=0.340, L4_strict=0.282, swap_rate=0.513
- margin=0.00, density>2, stability=2w: recovery=0.339, damage=0.386, L1_strict=0.368, L3_strict=0.340, L4_strict=0.338, swap_rate=0.685

## Does any region achieve both goals?

Using two purely descriptive, predeclared narrative bars -- stability damage <= 0.02 and minor-mode recovery >= 0.1 -- applied identically to every condition (chosen for readability before scanning the grid's own results, not used to select or tune any configuration):

**No condition in this 54-point grid meets both bars simultaneously.** Every condition with meaningfully low damage (<= 0.02) also has low minor-mode recovery, and every condition with meaningful recovery (>= 0.1) also carries non-trivial stability damage -- the Pareto frontier itself (see plot) trades one directly against the other across its full range, with no flat/free region at low damage. **Within this predeclared grid and this gate design (density + collection-stability + top1-vs-top2 margin, applied at individual timesteps), timestep-level gated weighted-profile resolution appears likely insufficient to recover meaningful minor-mode predictions without some monophonic-stability cost.** This does not rule out other, structurally different resolvers (e.g. phrase-level rather than window-level gating, or a different evidence representation entirely) -- it specifically characterizes this gate family's achievable trade-off space.

## Plot

`05_Figures_Results/PHASE3H_C_gate_sensitivity_pareto.png` -- damage (x, lower better) vs. recovery (y, higher better), colored by margin threshold, Pareto frontier marked with stars and connected.

## Scope note

This is Phase 3H-C only: a diagnostic sensitivity sweep over Phase 3H-B's existing gate design. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run or implemented. No new gate logic was introduced -- `compute_active_pc_count`, `compute_collection_stability_mask`, and `compute_normalized_margin` are imported unmodified from Phase 3H-B; only their threshold/window arguments are swept across the predeclared grid. Variant D's weighted-profile predictions are Phase 3H-A's exact, unchanged predictions (`variant_D_weighted_profile`, imported verbatim). Anchors are used exclusively inside `compute_anchor_metrics`, strictly for evaluation after each condition's key_id sequence is already fully computed -- no anchor or expected-key value is ever passed into `compute_variant_E_for_condition` or any gate function. No condition is selected as a winner; this script does not modify or overwrite any Phase 3G-A/3G-B/3H-A/3H-B output.
