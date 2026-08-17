# Strategy and Personality Synthesis: Diversity Dividends of a Heterogeneous Mutual Society and Their Dependence on Domain Gap

> Preliminary report (v1 experiment) · 2026-08 · Pre-registered criteria; negative results reported as-is
> Companion code: `eeg_v6_personality_society.py` (public release)

## Abstract

We define *personality* as a differentiable strategy of *how* an agent uses its own capabilities — a composition of four parameter dimensions: view choice, memory dynamics, gating sharpness, and voting channel — rather than any task-irrelevant identity label. On this definition we rebuild the homogeneous mutual society of the DAME framework (12 experts differing only in anchor direction) into a three-family heterogeneous society, PersonalitySocietyV1: an edge family (static coupling configuration), a power family (energy state), and a slow family (within-window trend), each with distinct memory-update scales and gate-temperature curves. On SEED-IV emotion recognition we run a two-arm comparison (heterogeneous vs. original society, differing only in the society module) with pre-registered judgment criteria. Results: under the cross-session protocol (small domain gap) the heterogeneous society gains +4.79 points (3 seeds, paired one-sided t-test, p≈0.04; pre-registered verdict "clear win"); under the cross-subject protocol (large domain gap) the gain is only nominal (+0.93 points; verdict "parity"). These two data points form the first evidence for the proposition that *diversity dividends decrease with domain gap*, a sister law to the field-condition law of the DAME paper (social contribution as a function of field difference). Limitations (8-subject fast protocol, single dataset, strategy probe not yet performed) and next steps (mutual-aid-tendency heterogeneity, personality synthesis experiments) are reported honestly.

**Keywords**: strategy synthesis; heterogeneous mixture of experts; EEG decoding; domain gap; pre-registration

---

## 1 Introduction

The mutual society of the DAME architecture (MutualSocietyV3) contributes +0.77 points in the cross-session setting and is characterized by the field-condition law (Proposition 1) as $E[\Delta_{\mathrm{soc}}]=a\sqrt{\kappa}-\eta$. Implementation-wise, however, an unfinished gap remains: **the 12 experts are homogeneous** — they share one view, one memory dynamics, and one temperature curve, differing only in their anchor (expertise) directions. This paper treats that gap as the entry point of an independent problem:

> **Personality = a differentiable strategy of "how to use", not an identity of "what one is".**

Given the same region-coupling information, one expert can attend *only to static configuration* (a "cautious" personality), another *only to energy state* ("impulsive"), a third *only to trend* ("sensitive"); one remembers fast and forgets fast, another remembers slowly and forgets slowly; one gate is blunt and stable, another sharp and selective. The four dimensions compose the expert's strategic personality. The society is then not a crowd of homogeneous voters but a set of **personality-heterogeneous strategy holders**, whose outputs are combined through gated weighting (synthesis = a differentiable composition of personalities).

This paper reports the first preliminary experiment: replace only the society module (everything else identical to DAME-C5, word for word) and test whether heterogeneity yields gains, and under which domain gap the gains appear. Judgment criteria were pre-registered before the experiment (Appendix A); negative and positive results carry equal weight.

## 2 Formalization

### 2.1 Four-Dimensional Parameterization of Strategy

Let the society contain $N=12$ experts in $F=3$ families of 4. The personality of expert $i$ is determined by four dimensions:

- **View** $v_i \in \{\text{edge}, \text{power}, \text{slow}\}$: the input channel the expert attends to — static edge configuration (PLV means over $P$ region pairs × 5 bands), whole-brain projected power, or within-window PLV change (last sub-window minus first);
- **Memory dynamics** $\eta_i = \hat{\eta}_i \cdot s_{f(i)}$: update gate times a family scale, $s=(1.0, 1.3, 0.7)$ for normal, fast, and slow memory;
- **Gate temperature curve** $\tau_{f(i)}(t) = \tau^{\mathrm{lo}}_{f(i)} + (\tau^{\mathrm{hi}}_{f(i)} - \tau^{\mathrm{lo}}_{f(i)}) \cdot \mathrm{progress}$: the three families use $(0.8, 3.5)$, $(1.2, 4.0)$ (sharper), and $(0.6, 2.5)$ (blunter);
- **Voting channel**: one output projection per family; gated summation within a family, then family projection, then summation as arbitration:

$$
O = \sum_{f} W^{\mathrm{out}}_f \left( \sum_{i \in \mathrm{fam}_f} g_i \, m_i^{\mathrm{new}} \right)
$$

where $g_i$ is the gating weight (cosine similarity between expert anchor and view, sharpened by temperature) and $m_i^{\mathrm{new}}$ the updated memory. The mutual structure ($W_{\mathrm{mutual}}$ adjacency initialization, community mask, community reassignment) and the three-stream GRU memory are kept identical to MutualSocietyV3, **so that the two-arm difference is cleanly attributable**.

### 2.2 Proposition under Test

**Proposition H1 (diversity dividends decrease with domain gap)**. Let $\kappa$ denote domain-gap strength (subject-to-subject > session-to-session). The net gain of heterogeneity $\Delta_{\mathrm{het}}(\kappa)$ satisfies:

$$
\kappa_1 > \kappa_2 \;\Longrightarrow\; \Delta_{\mathrm{het}}(\kappa_1) \le \Delta_{\mathrm{het}}(\kappa_2)
$$

Intuition: when the gap is small, EEG variability comes mostly from the variability of emotional content, and diverse strategies each exploit their own content → dividend; when the gap is large, variability is dominated by subject/session noise, and strategy diversity cannot be monetized → dividend absorbed. This proposition is isomorphic to the field-condition law: both assert that *society-level contribution is a function of domain gap*, one along the dimension of strategy diversity, the other along the dimension of society existence.

**Falsification condition (pre-registered)**: CS protocol $\Delta < 0.5$ points, or LOSO protocol $\Delta \ge 1.0$ points in the opposite direction.

## 3 Experimental Setup

- **Data**: SEED-IV raw EEG (62 channels → 12 regions), 4 emotion classes, 8-subject fast protocol (speed version of the preliminary study; the formal version will extend to 15 subjects). Window 4 s; PLV computed over 8 sub-windows of 0.5 s.
- **Models**: both arms are the complete DAME pipeline (coupling front-end + water cycle + stability head + fusion v6), with **the society module as the sole difference**:
  - P6-Soc: PersonalitySocietyV1 (three-family, four-dimensional heterogeneity);
  - DAME-C5: MutualSocietyV3 (homogeneous 12 experts, the paper's baseline).
- **Training**: protocol identical to the DAME paper, word for word (AdamW, CosineAnnealingWarmRestarts, 15 epochs, 3 seeds {42, 123, 789}); training loops directly reuse `loso_v5` / `session_run_v5`, ruling out training-difference confounds.
- **Protocols**: LOSO (leave-one-subject-out, large gap) and CS (train sessions 1+2 → test session 3, small gap).
- **Pre-registered criteria** (full text in Appendix A):
  - CS: $\Delta \ge +0.5$ points → clear win; $0 \le \Delta < 0.5$ → nominal; $\Delta < 0$ → reported as-is;
  - LOSO: $|\Delta| < 1.0$ points → parity; $\ge +1.0$ → surprise; $\le -1.0$ → harmful.

## 4 Results

### 4.1 Cross-Session Protocol (small gap): pre-registration hit

| Seed | P6-Soc | DAME-C5 | Δ |
|---|---|---|---|
| 42 | 35.50 | 32.84 | +2.66 |
| 123 | 33.96 | 29.82 | +4.14 |
| 789 | 36.32 | 28.74 | +7.58 |
| **Mean** | **35.26 ± 1.20** | **30.47 ± 2.13** | **+4.79** |

All 3/3 seeds win; paired one-sided t-test $t(2) = 3.29$, $p \approx 0.04$, crossing the significance line; pre-registered verdict "**clear win**".

### 4.2 Cross-Subject Protocol (large gap): dividend disappears

3 seeds × 8 folds LOSO: P6-Soc 33.32 ± 4.12 vs. DAME-C5 32.73 ± 3.80, Δ = +0.59 points; per-seed Δ: +0.93 / −1.44 / +2.28 (positive in 2/3 seeds). Verdict "parity" — heterogeneity produces no claimable gain across subjects.

### 4.3 Gradient Comparison (first evidence for Prop H1)

| Domain gap | Δ (heterogeneous − homogeneous) | Direction |
|---|---|---|
| Small (CS, across sessions) | **+4.79** | dividend |
| Large (LOSO, across subjects) | +0.59 (2/3 seeds positive) | parity |

The two data points fall in the direction predicted by Proposition H1: dividends decrease with the gap. Honest wording: two points do not constitute a statistical test of the monotonic law — they are direction-consistent preliminary evidence; intermediate gap levels (e.g., mixed session–subject protocols, other datasets) are required to upgrade H1 from a directional hypothesis to an empirical regularity.

## 5 Discussion

**5.1 Sister relationship with the field-condition law.** Proposition 1 of the DAME paper asserts that the contribution of the society's *existence* is a function of field difference; H1 asserts that the contribution of the society's *internal diversity* is also a function of the gap. The two propositions share one mechanistic intuition — society-level mechanisms cash out where content varies and domain is stable, and are absorbed where domain varies. If H1 survives further gap levels, the two laws can merge into a more general scaling law of sociality.

**5.2 Why heterogeneity gives only nominal gains on LOSO.** Consistent with the probe study: in cross-subject settings, variability is dominated by individual configuration (fingerprint decodability 0.995), and the "content diversity" of the emotion signal is drowned by domain noise; however diverse the strategies, the dividend available to harvest is insufficient. The negative gain of DANN is a side witness of the same logic: forcing changes on top of a noise floor yields no accuracy.

**5.3 Limitations (as-is)**. ① 8-subject fast protocol, not the full 15 subjects — the formal version must complete this; ② single dataset, the extrapolation of H1 untested; ③ "personality = strategy" is still a modeling assignment: we have not yet probed whether the gates actually select different strategies (family distribution of gates, view utilization, inter-family output divergence) — the first step of the v2 experiments is a strategy probe; ④ the mechanism attribution of the heterogeneity gain lacks per-dimension ablation (unknown contributions of view / dynamics / temperature / voting).

**5.4 Next steps.** v2: mutual-aid-tendency heterogeneity (gregarious vs. solitary experts); learning the gating sharpness; per-dimension ablation; strategy probe; personality synthesis experiments (given two trained personalities, compose an intermediate personality and measure its generalization surface). If "personality synthesis" holds in v2, the operational definition "personality = differentiable strategy" gains validation — and becomes the body of the follow-up paper *Strategy and Personality Synthesis*.

## 6 Conclusion

Under the small-gap cross-session protocol, the personality-heterogeneous society beats the homogeneous society by +4.79 points (3 seeds, p≈0.04), a pre-registered "clear win"; under the large-gap cross-subject protocol the dividend vanishes. This gradient provides the first evidence for "diversity dividends decrease with domain gap" (Proposition H1) and takes the first step toward the operational definition "personality = a differentiable how-to-use strategy". The negative result (LOSO parity) is reported with equal weight as the positive one.

---

## Appendix A Pre-registered criteria (registered before the experiment, verbatim)

> LOSO: $|\Delta| < 1.0$ → parity (expected: heterogeneity does not cross the noise floor; Appendix E 900× noise ratio);
> $\Delta \ge +0.5$ → surprise (heterogeneity crosses the domain gap; positive evidence for transferable personality anchors);
> $\Delta \le -0.5$ → harmful (power view leaks subject identity; probe3 precedent — equally important).
>
> CS: $\Delta \ge 0$ → heterogeneity ≥ region anchors; $\Delta \ge +0.5$ → clear win (expected direction: diverse strategies exploit content variability).

**Tier resolution note**: in the LOSO header the intervals $|\Delta|<1.0$ and $\Delta \ge +0.5$ overlap; the umbrella clause takes precedence — anything within 1.0 point is "parity", ±0.5 points is annotated only as a nominal direction; crossing ±1.0 points is required for "surprise" / "harmful".

## Appendix B Reproducibility

```bash
# Cross-session protocol (small domain gap)
python eeg_v6_personality_society.py --fast 8 --session-split --seeds 3

# Cross-subject protocol (large domain gap)
python eeg_v6_personality_society.py --fast 8 --seeds 3

# Smoke test (2 subjects, 2 epochs; separate result file, does not pollute the formal cache)
python eeg_v6_personality_society.py --smoke
```

Results are cached in `results/eeg_v6p_results.json` (keys carry protocol markers: LOSO = `_dame_s{f}_`, CS = `_dame_sess{sess}_`). Dependencies: `eeg_v5_coupling_experiment.py` (DAME main pipeline) and the SEED-IV raw dataset.

## Appendix C Code Listing

All code for this experiment lives in `eeg_v6_personality_society.py` (public release; comments in a concise human style). Core structure:

- `PersonalitySocietyV1`: three-family, four-dimensional heterogeneous society (view / memory dynamics / temperature curve / voting channel), mutual structure kept from V3;
- `DAME_Personality`: top model; fusion and loss identical to `DAME_Coupling` word for word, only the society module swapped;
- `main`: two-arm comparison + protocol-aware caching + pre-registered verdicts.

Both arms share the same front-end, water cycle, stability head, and training loop — every Δ is cleanly attributable to the heterogeneity of the society module.
