# DARWIN

**Drift-Aware, Subject-Adaptive EEG Decoding Without Subject-Level Calibration**

DARWIN is a motor-imagery EEG decoding framework that adapts to new users and to signal drift *during* a session, without collecting labelled calibration data from the new user. It combines a self-supervised encoder, a donor-matching prior, and a recursive filter that tracks slow drift in a single shared embedding space.

> **Status:** Research project in active development. Numbers and results will be added as experiments complete.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Datasets](#datasets)
- [Evaluation Protocol](#evaluation-protocol)
- [Baselines](#baselines)
- [Ablations](#ablations)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Scope and Limitations](#scope-and-limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

EEG-based brain-computer interfaces degrade when the user changes (cross-subject shift), when the recording day changes (cross-session shift), and when the signal slowly drifts within a session. Most solutions need labelled calibration trials from every new user.

DARWIN takes a different approach:

- It uses **no subject-level, label-based calibration**. The held-out subject's labels are never used.
- Montage differences are handled by a small projection learned from **unlabelled** data (LSP).
- Drift is tracked online by a filter that operates in one standardized 64-dimensional space shared by every module.

DARWIN does not claim to be the first method to address cross-session and cross-user shift. Online Riemannian mean updates, Kalman-style drift tracking and cross-session adaptation already exist. The contribution is the unified design and its evaluation against those families.

## Key Features

- **Montage-flexible input** via LSP, a ~1K-parameter learned spatial projection to 16 channels. It can revert to rigid channel intersection with a single flag.
- **Self-supervised encoder** (EEGNet-based) trained with a subject-balanced InfoNCE objective, frozen after pre-training.
- **SSRS (donor-based prior):** blends embeddings from similar donor subjects to initialize the state of the filter. High uncertainty in the match produces a larger initial gain.
- **DTKF (drift-tracking Kalman filter):** a per-dimension recursive update that subtracts the task-evoked component before updating, so it follows drift rather than the stimulus.
- **FiLM conditioning** (~2K parameters) that feeds the filtered state into the classifier.
- **Uncertainty-aware decoding** by sampling from the filter posterior, evaluated with risk-coverage curves, ECE and NLL.
- **Change-point handling** that skips artifact windows and re-matches donors after a genuine step change.

## Architecture

```
Raw EEG (any montage)
      │
CAR + MAD preprocessing
      │
LSP  (~1K params, 16 ch)
      │
EEGNet encoder  (frozen after Phase 0)
      │
┌─────▼───────────────────────────────────────────┐
│   64-d standardized state space (pre-projection) │
│   every module reads or writes this space        │
└─────▲──────────────▲───────────────────▲─────────┘
      │              │                   │
 Phase 0 SSL     SSRS prior          DTKF update
 (InfoNCE)    (donor blend: m0, P0)  (class-marginal innovation)
                                          ▲
                                          │ class posterior
      state ──► FiLM (~2K) ──► Frozen classifier ──► Prediction + uncertainty
```

Training runs in phases:

| Phase | What is trained | Notes |
|-------|-----------------|-------|
| 0 | Encoder (self-supervised, InfoNCE, subject-balanced) | Frozen afterwards. Uses unlabelled data only |
| 1 | FiLM and classifier head on donor data | Frozen at test time |
| Test | Nothing is trained | State is updated online by the filter |

## How It Works

### Prior (SSRS)

With donor weights `s_i` and donor embeddings `e_i`:

$$m_0 = \sum_i s_i e_i, \qquad P_0 = \mathrm{diag}(v_w)\,(1 + \lambda \tilde H)$$

`v_w` is the weighted per-dimension variance of the blended donors and `H̃ ∈ [0, 1]` is the normalized entropy of the similarity distribution. If no good match exists, the prior falls back to the population mean and variance.

### State model

For window embedding `z_t`, class posterior `π_k(t)` from the frozen classifier and source class offsets `μ_k`:

$$x_t = x_{t-1} + w_t,\; w_t \sim \mathcal{N}(0, Q) \qquad z_t = x_t + \sum_k \pi_k(t)\,\mu_k + v_t,\; v_t \sim \mathcal{N}(0, R)$$

### Class-marginal update

The expected task-evoked part of the embedding is removed before the update, so the filter does not chase the stimulus:

$$\nu_t = z_t - x_{t-1} - \sum_k \pi_k(t)\,\mu_k, \qquad K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}, \qquad x_t = x_{t-1} + K_t\,\nu_t$$

All operations are per dimension. The class posterior is computed from the state at `t-1`, so a prediction never reinforces itself within a step. The primary policy updates on every window; updating on baseline windows only is available as an alternative.

### Noise estimation (label-free)

- **R:** per-dimension residual variance from stationary within-class segments of donor subjects.
- **Q:** per-dimension variance of the window-to-window displacement of the session-level embedding mean (within-session drift only). The between-day offset is handled by the SSRS prior.
- **Optional:** R can be scaled by a clipped running average of the normalized innovation squared.

### Change-point handling

For each window the normalized innovation squared is compared with a chi-square quantile. Windows above the gate are skipped (blinks, electrode slips). After `k` consecutive skips a change point is declared, `P` is reset to `P_0`, and SSRS is re-run on the most recent 10 s buffer.

### Uncertainty-aware decoding

`S` samples (about 10–20) are drawn from `N(m_t, P_t)`, passed through FiLM and the classifier, and their softmax outputs averaged. Predictive entropy and between-sample disagreement serve as uncertainty scores. They are judged empirically (risk-coverage, ECE, NLL) and are not assumed to be calibrated.

## Datasets

| Dataset | Subjects | Channels | Sessions | Role |
|---------|----------|----------|----------|------|
| OpenBMI | 54 | 62 | 2 | Primary: significance testing, cross-session and drift analysis |
| BCI Competition IV-2a | 9 | 22 | 2 | Cross-subject and cross-session, per-subject reporting |
| BCI Competition IV-2b | 9 | 3 | 5 | LSP stress test on a 3-channel montage, per-subject reporting |
| PhysioNet MI | 109 | 64 | 1 | Largest donor pool, cross-subject and streaming only |
| SEED | 15 | 62 | 3 | Optional generality check (emotion recognition) |

Pooled-donor experiments harmonize to left-hand vs right-hand imagery. Four-class results (2a, PhysioNet) are reported per dataset. Please verify session splits for the dataset release you use.

## Evaluation Protocol

1. **Cross-subject LOSO (headline):** transductive leave-one-subject-out. Unlabelled data from the held-out subject may be used in Phase 0, but its labels never are. The held-out subject is always excluded from the donor pool and from Phase 1.
2. **Cross-session LOSO:** train on donors' first session, test on the held-out subject's later session. Phase 0 never sees the test session.
3. **Streaming within-session:** chronological windows, with accuracy over time-in-session and first-half vs second-half accuracy.
4. **Synthetic drift injection:** slow gain modulation, low-frequency offset and per-channel attenuation applied to the raw EEG (ramp, step and random-walk profiles), measuring how much of the no-drift accuracy each method recovers.
5. **Health metrics:** SSRS fallback rate, mean normalized innovation squared (should be near 1) and change-point counts.
6. **Strict variant:** subject-grouped 5-fold with test subjects excluded from Phase 0 and the donor pool.

Statistics: paired Wilcoxon signed-rank test across subjects, at least 3 seeds, and identical splits and preprocessing for every method. Baselines are tuned on source folds only.

## Baselines

| Family | Methods |
|--------|---------|
| Zero adaptation | EEGNet trained on pooled donors |
| Static alignment | Euclidean alignment (EA) + EEGNet; CORAL; Riemannian re-centring with MDM / tangent-space LR |
| Online alignment | Online Riemannian mean update |
| Online normalization | AdaBN; exponential moving standardization (EMS); tuned fixed-gain EMA |
| Test-time adaptation | Tent |
| Adversarial | DANN-style subject-adversarial model |
| Calibration reference | 5-shot fine-tuning |
| Foundation models | LaBraM, CBraMod (frozen linear probe, reference only) |

## Ablations

Each ablation targets a single question:

| # | Comparison | Question |
|---|------------|----------|
| 1 | Full filter vs tuned EMA vs AdaBN vs EMS | Is the filter formulation justified? |
| 2 | Class-marginal vs baseline-window-only vs uncorrected updates | Does it track drift or the task? |
| 3 | Chronological vs shuffled vs time-reversed streams | Does it use temporal structure? |
| 4 | Skewed class sequences (e.g., 80/20, single-class runs) | Does the class-marginal update lock in errors? |
| 5 | Oracle centroids (unlabelled full-session, labelled few-shot) | How far is the online estimate from the upper bound? |
| 6 | Fixed R vs innovation-scaled R | Is scaling worth the complexity? |
| 7 | Change-point handling on vs off | Do step changes break the filter? |
| 8 | Donor bank size sweep | Does more donors help or cause negative transfer? |
| 9 | Resting vs task warm-up windows for SSRS | What data does donor matching need? |
| 10 | Standardized vs raw vs tangent-space state | Does the geometry matter? |
| 11 | FiLM matched vs mismatched conditioning | Does consistency training help? |
| 12 | Phase 0 subject-probe vs task-probe; transductive vs strict | Does the shared space keep task structure? |

SSRS on/off and LSP on/off are reported in every table.

## Getting Started

> The commands below are placeholders. Replace them with the actual entry points once the code is organized.

```bash
# Clone the repository
git clone https://github.com/<your-username>/DARWIN-.git
cd DARWIN-

# Install dependencies
pip install -r requirements.txt
```

Typical workflow:

```bash
# 1. Prepare datasets and preprocessing (CAR + MAD)
# 2. Phase 0: self-supervised encoder training
# 3. Phase 1: train FiLM and classifier on donor data
# 4. Run evaluation protocols (LOSO, cross-session, streaming, drift injection)
```

## Project Structure

> Update this section to match the repository layout.

```
DARWIN-/
├── data/            # Dataset loaders and preprocessing
├── models/          # LSP, EEGNet encoder, FiLM, classifier
├── filter/          # SSRS prior, DTKF update, change-point logic
├── baselines/       # EA, AdaBN, EMS, Tent, DANN, online Riemannian mean
├── experiments/     # Evaluation protocols and ablations
├── configs/         # Experiment configurations
└── README.md
```

## Scope and Limitations

- DARWIN claims **no subject-level, label-based calibration**. The phrase "calibration-free" is intentionally not used unqualified, because montage handling relies on a projection learned from unlabelled data.
- The headline protocol is **transductive**: unlabelled held-out data may be used in Phase 0.
- BCI IV-2a and 2b have only 9 subjects each and cannot carry statistical claims on their own. OpenBMI is the primary dataset for significance.
- If the within-session drift study shows negligible drift, the contribution reduces to donor selection plus cross-session offset handling.
- If the filter does not outperform a tuned EMA, it will be reported as a gated EMA.

## Roadmap

- Full Gaussian-sum (mixture) filter with a posterior over donor weights
- Fully adaptive, label-free estimation of Q and R
- FiLM-confidence modulation (filter uncertainty scales FiLM parameters)
- Fine-tuned foundation-model baselines (LaBraM, CBraMod)
- LSP retrained on unlabelled target data; electrode-coordinate parametrization for unseen montages
- Strict LOSO with per-fold Phase 0 retraining
- Additional multi-day datasets as generality checks

## Citation

If you use this work, please cite it as:

```bibtex
@misc{darwin2026,
  title  = {DARWIN: Drift-Aware, Subject-Adaptive EEG Decoding Without Subject-Level Calibration},
  author = {Richa},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

Specify a license (e.g., MIT) and add a `LICENSE` file.
