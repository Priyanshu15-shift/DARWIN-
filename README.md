# DARWIN

### Dual Adaptation Framework for Robust Cross-Session and Cross-Subject EEG Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-ongoing-blue)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

> A calibration-free deep learning framework for robust EEG decoding that jointly models inter-subject variability and intra-session drift through similarity-based subject conditioning and continuous latent-state tracking — without gradient updates at test time.

---

## Table of Contents

- [Overview](#overview)
- [Why DARWIN](#why-darwin)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Electroencephalography (EEG)-based Brain-Computer Interface (BCI) systems routinely lose accuracy when a decoder trained on one subject or recording session is deployed on a new user or a new session. This degradation stems from two distinct, and often conflated, sources of distribution shift:

| Source | Nature | Cause |
|---|---|---|
| **Cross-subject variability** | Discrete, fixed at session start, unstructured | Anatomical and physiological differences between individuals (skull thickness, cortical folding, electrode placement) |
| **Cross-session variability** | Continuous, evolving, structured | Fatigue, electrode impedance changes, cognitive state, attention drift |

Most existing approaches try to erase these differences through adversarial training, domain adaptation, or meta-learning. These methods can work well in narrow settings, but they typically require per-user calibration, are unstable to optimize at small subject counts, or add inference-time cost that is incompatible with real-time BCI use.

**DARWIN takes the opposite approach: instead of forcing invariance, it explicitly models both sources of variability**, using a mechanism suited to each one — similarity-based *matching* for the discrete, subject-level component, and Kalman *filtering* for the continuous, session-level component — inside a single shared latent space. The result is a framework that adapts to new users and drifting sessions at inference time with no labels, no backpropagation, and no calibration procedure.

## Why DARWIN

Prior work generally treats subject and session shift as a single nuisance factor to be removed. DARWIN's core hypothesis is that these two factors are *separable* and *qualitatively different*, and therefore deserve different tools:

- **Subject identity** is essentially fixed once a session begins — a matching/selection problem best solved by comparing a new user against a pool of known source subjects.
- **Session drift** unfolds continuously over time — a tracking/filtering problem best solved with a lightweight state estimator that updates every sample.

By keeping both signals in the *same* 64-dimensional embedding space, DARWIN also sidesteps a common design flaw in Kalman-filter-based neural decoding: an undefined or hand-engineered measurement model. Because the state (subject embedding) and the measurement (encoder output) live in the identical space, the innovation term is simply their difference — no learned observation matrix required.

## Key Contributions

- **Unified Latent Representation** — A single shared 64-dimensional embedding space is used consistently across self-supervised pretraining, subject adaptation, drift estimation, and classification, eliminating representation mismatch between modules.
- **Learnable Spatial Projection (LSP)** — Heterogeneous EEG channel layouts across datasets are harmonized through a trainable, data-adaptive spatial mixer rather than rigid channel intersection, preserving information from non-overlapping electrodes.
- **Subject Similarity-Based Adaptation (SSRS)** — New users are initialized through similarity-weighted donor embeddings drawn from a source subject pool, with an entropy-gated fallback to population-level conditioning when donor matching is unreliable.
- **Continuous Drift Tracking (DTKF)** — A lightweight diagonal Kalman filter continuously updates the subject representation throughout inference using only unlabeled data.
- **Feature-wise Conditional Modulation (FiLM)** — Learned EEG representations are dynamically modulated according to the current subject/session state, without altering the backbone encoder.
- **Calibration-Free Deployment** — No backpropagation, parameter updates, or supervised calibration are required at inference time.
- **Early Empirical Validation** — A dedicated diagnostic stage tests the core subject/session separability hypothesis using baseline features, *before* any DARWIN-specific module is built, so the central assumption is checked rather than assumed.

## Architecture

DARWIN is a single, unified pipeline in which every stage reads from and writes to the same 64-dimensional embedding space:

```
Raw EEG (C channels × T samples)
        │
        ▼
CAR + MAD  (common average reference, outlier clipping)
        │
        ▼
Learnable Spatial Projection  →  C' shared channels
        │
        ▼
EEG Feature Encoder  (9-band spectral backbone)
        │
        ▼
Projection Head  →  64-dim unified embedding z
        │
        ▼
Self-Supervised Representation Learning  (InfoNCE, subject-balanced negatives)
        │
        ▼
  ┌───────────────────────────────┐
  │   Subject & Session Adaptation │
  │                                 │
  │   SSRS (donor matching)         │
  │   DTKF (drift tracking) ────────┼──── updates subject embedding every sample
  └───────────────────────────────┘
        │
        ▼
FiLM-Conditioned Classifier
        │
        ▼
Task Prediction
```

**Key invariant:** the projection-head output is the *only* embedding space in the system. Self-supervised pretraining, SSRS donor centroids, DTKF measurements, and FiLM conditioning all operate on this same 64-dimensional representation by construction — there is no separate measurement model and no bridging module between components.

## Core Components

### Learnable Spatial Projection (LSP)

A lightweight, dataset-aware channel mixer that aligns EEG montages of differing size (e.g. 64, 32, 22, or 16 channels) into a common spatial representation. Each dataset gets its own learnable projection matrix, initialized as a soft, spatially-local channel selector and refined during pretraining. Unlike rigid channel intersection, LSP preserves information from electrodes that don't overlap across datasets, at a cost of roughly 1K parameters total.

### Self-Supervised Representation Learning

Task-aware EEG embeddings are learned via contrastive (InfoNCE) pretraining over pooled, unlabeled data from all datasets. Negative sampling is **subject-balanced** — every batch includes same-subject positives alongside cross-subject negatives — specifically to prevent the encoder from taking the shortcut of encoding subject identity instead of task-relevant neural content. Two linear probes (subject-ID and trial/class) gate progression to the next training phase, catching an identity-shortcut failure mode early and cheaply.

### Source Subject Relevance Scorer (SSRS)

Given as little as ~10 seconds of unlabeled EEG from a previously unseen user, SSRS computes a user centroid in the shared embedding space, scores its similarity against every source subject's centroid, and forms an initial personalized embedding as a similarity-weighted combination of donor embeddings. An entropy check on the similarity distribution governs a graceful fallback: when confidence is low, DARWIN conditions on a population-average embedding rather than trusting an unreliable match.

### Drift-Tracking Kalman Filter (DTKF)

Rather than assuming a subject's representation is static for the duration of a session, DTKF treats it as a latent state that evolves slowly over time. A diagonal (per-dimension) Kalman filter updates this state on every incoming sample using the innovation between the current encoder output and the previous state estimate — no observation matrix is needed, since state and measurement share the same space. Outlier rejection and value clipping guard against artifact-driven runaway drift, keeping the mechanism robust and cheap enough for real-time use.

### FiLM-Based Subject Conditioning

Feature-wise Linear Modulation applies a learned per-band scale and shift, derived from the current subject/session embedding, to the encoder's features before classification. This lets predictions stay personalized to the current user and moment in time without modifying or retraining the backbone encoder.

## Methodology

DARWIN is trained in two phases and deployed with a fixed, frozen backbone.

**Phase I — Self-Supervised Representation Learning**
- Learn dataset-independent EEG representations from pooled, unlabeled data
- Contrastive (InfoNCE) objective with subject-balanced negative sampling
- Encoder is frozen once two linear-probe diagnostics pass (subject-ID and task-content)

**Phase II — Supervised Fine-Tuning**
- Initialize subject representations from the frozen embedding space
- Train FiLM conditioning layers and the lightweight classifier
- Encoder weights remain fixed throughout

**Test-Time Adaptation (Deployment)**
1. Estimate an initial subject embedding via SSRS from a short window of unlabeled data.
2. Continuously refine that embedding sample-by-sample using DTKF.
3. Condition extracted EEG features through FiLM.
4. Produce a task prediction with the frozen classifier.

No labels, calibration procedure, or optimization step is required at any point during deployment.

## Repository Structure

```
DARWIN/
├── README.md
├── LICENSE
├── CHANGELOG.md        
├── requirements.txt
│
├── reports/              ← Research progress lives here
│   ├── weekly/
│   │   ├── week01.md
│
├── docs/
│   ├── architecture.md         ← module-by-module spec (LSP, SSRS, FiLM, DTKF, classifier)
│   ├── preprocessing.md        ← the raw-EEG → windowed-embedding data pipeline
│   ├── training.md
│   ├── evaluation.md
│   ├── datasets.md
│   ├── usage.md
│   └── references.md
│
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── trainers/
│   ├── evaluation/
│   └── utils/
│
├── notebooks/
│
└── experiments/
```

## Datasets

DARWIN is designed to support heterogeneous EEG datasets that differ in channel layout, sampling protocol, and task paradigm. The preprocessing pipeline accommodates these differences through the Learnable Spatial Projection rather than a fixed channel intersection, so no dataset is forced to sacrifice electrodes just because another dataset lacks them.

Full dataset preparation details are documented in [`docs/datasets.md`](docs/datasets.md).

## Evaluation

DARWIN is evaluated across several generalization regimes designed to isolate and stress-test each source of distribution shift:

- **Leave-One-Subject-Out (LOSO)** — train on all but one subject, evaluate on the held-out subject; SSRS initializes from the source pool and DTKF tracks drift across the held-out session.
- **Leave-One-Session-Out (LOSSO)** — train on session 1 across all subjects, evaluate on session 2 of the same subjects; isolates pure session-drift recovery.
- **Cross-Dataset Evaluation** — train on a subset of datasets, evaluate on an unseen dataset with a different electrode montage and task paradigm.
- **Calibration Curve** — accuracy as a function of unlabeled adaptation data (0s, 5s, 10s, 30s, 60s), showing how quickly SSRS + DTKF approach the fully-supervised upper bound.
- **Component Ablations** — each module (LSP, SSRS, DTKF, FiLM, subject-balanced sampling) is removed in isolation and compared against the full system.

Results are benchmarked against classical EEG decoders (EEGNet, FBCNet, EEG Conformer), a graph-attention baseline (GAT), a domain-generalization baseline (CORAL-DG), and adversarial/disentanglement state-of-the-art methods (SCALE-Net, TS²-DER).

Diagnostic tooling — subject-ID/task-content linear probes, embedding-space visualization, SSRS fallback-rate logging, and DTKF innovation-residual analysis — is used throughout to debug and validate each component rather than relying on end-task accuracy alone.

## Design Principles

DARWIN is built around four guiding principles:

1. **Explicit modeling instead of enforced invariance** — subject and session variation are treated as signal to condition on, not noise to erase.
2. **Lightweight adaptation suitable for real-time deployment** — the full model targets well under 50K parameters with no test-time backpropagation.
3. **Calibration-free inference** — no supervised calibration step is required for a new user or session.
4. **Unified latent representations across all adaptation modules** — a single embedding space removes the need for bridging or measurement-model engineering between components.

## Roadmap

Planned future directions beyond the current scope:

- Graph-based subject similarity propagation for SSRS
- Full-covariance (non-diagonal) Kalman filtering for DTKF
- Cross-paradigm EEG adaptation
- Multi-modal physiological signal integration
- Online continual learning

## Documentation

Comprehensive technical documentation lives in the `docs/` directory:

| Document | Description |
|---|---|
| `architecture.md` | Complete architectural design |
| `methodology.md` | Training methodology |
| `preprocessing.md` | EEG preprocessing pipeline |
| `datasets.md` | Dataset preparation |
| `training.md` | Training procedures |
| `evaluation.md` | Experimental protocols |
| `usage.md` | Running experiments |
| `references.md` | Bibliography |

## Citation

If you use DARWIN in your research, please cite the accompanying publication (to be released):

```bibtex
@article{darwin2026,
  title   = {DARWIN: Dual Adaptation Framework for Robust Cross-Session and Cross-Subject EEG Decoding},
  author  = {Authors},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Acknowledgements

DARWIN is inspired by recent advances in self-supervised representation learning, domain generalization, adaptive filtering, and calibration-free Brain-Computer Interface research.
#   D A R W I N -  
 #   D A R W I N -  
 