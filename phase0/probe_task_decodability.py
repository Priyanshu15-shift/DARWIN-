"""
probe_task_decodability.py

Gate G3 (task probe): task/class decodability of the frozen Phase 0 embeddings.
(DARWIN spec rev A, section 9. This was called "Gate 1" in the earlier code
and docstrings; the old Phase 0 probe gate is now G3.)

WHAT IT MEASURES
    The frozen Phase 0 LSP + encoder produce 64-d PRE-PROJECTION embeddings
    (the encoder output -- the state space everything downstream uses). A
    single linear layer (nn.Linear, no hidden layers) is trained to predict the
    task label (motor-imagery class) from those embeddings.

    Run on pre-projection embeddings: the projection head (64 -> 128 -> 64) is
    NEVER applied here. It exists only to shape the InfoNCE loss in Phase 0.
    LSP and encoder are frozen and in eval mode (BatchNorm uses source running
    statistics), no augmentation is applied, and the probe sees the clean
    signal.

G3 CRITERION (spec rev A, section 9)
    Split: SUBJECT-DISJOINT (whole subjects held out for validation).
    Pass : final validation score >= chance + 5 percentage points,
           where chance = 1 / (number of classes in THAT dataset).
    Fail : near chance. That is the failure signal -- it means Phase 0 did not
           keep task structure, and the risk table says to fix augmentation
           and delay Phase 1.
    The score is the LAST epoch's, not the best over epochs (selecting the max
    on the reporting set would inflate it).
    The spec does not define how to aggregate across datasets when a pooled
    checkpoint is probed: this script reports each dataset separately and
    prints an overall verdict that requires every probed dataset to pass.

    The subject-ID probe (probe_subject_leakage.py) is NOT part of this gate.
    Per spec rev A it is a diagnostic only (plotted against this probe for
    ablation 12): a collapsed subject probe is a warning, a very high one
    alone is not a failure.

CHANGES vs the previous version of this script (measurement fixes only; the
encoder, LSP and the checkpoint are untouched)
    1. Embeddings are STANDARDIZED (train-fold mean/std) inside the probe. The
       raw 64-d encoder output has arbitrary scale, which made the probe's
       training loss bounce and, on PhysioNet, sit frozen at the majority class.
    2. The verdict metric is BALANCED accuracy (mean per-class recall) by
       default. Its chance level is exactly 1/n_classes even when classes are
       imbalanced (PhysioNet: one class is ~48% of epochs, so plain accuracy of
       0.48 "passed" a 0.25 threshold while predicting the majority class for
       everything). Plain accuracy is still printed; pick it with
       --verdict_metric accuracy to get the old behaviour.
    3. Class-weighted loss, so the probe cannot win by predicting the majority.
    4. MANY subject-disjoint folds instead of one split: leave-one-subject-out
       for small datasets (<= 12 subjects, i.e. 2a/2b), repeated random
       subject-disjoint splits otherwise. Reports mean +/- std over folds. A
       single 2-subject validation set was too noisy to trust.
    5. Prints per-class counts and a MAJORITY-CLASS baseline next to every
       result, plus pooled per-class recall.
    6. Embeddings are extracted once per dataset (not once per split).
    7. --json_out for comparing checkpoints (e.g. trained vs untrained).

    train_probe() / LinearProbe / extract_embeddings() keep their old call
    signatures, so probe_subject_leakage.py and probe_dataset_leakage.py still
    import and run unchanged -- they now also get standardized inputs (their
    numbers will move slightly; re-run both checkpoints with the same code
    before comparing).

SESSIONS
    By default the probe reads only the sessions Phase 0 was allowed to train on
    (the checkpoint's `train_sessions`), so it never touches a held-out test
    session. Use --probe_sessions all to include every session.

Usage:
    python probe_task_decodability.py --checkpoint phase0_checkpoint_<...>.pt
    python probe_task_decodability.py --checkpoint <ckpt> --dataset bci_iv_2a
    python probe_task_decodability.py --checkpoint <ckpt> --json_out task_probe.json

Accepts the pooled checkpoint format (`dataset_keys`,
`channel_names_by_dataset`) and the older single-dataset format
(`dataset_key`, `channel_names`).
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.dirname(__file__))
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset
from cache_config import CACHE_DIR  # shared, versioned cache path --
                                      # stdlib-only import, no moabb
                                      # dependency pulled in just to
                                      # resolve where the cache lives.

# G3: pass if final val score >= chance + this margin (5 percentage points).
G3_MARGIN = 0.05

# Datasets with at most this many subjects use leave-one-subject-out folds.
LOSO_MAX_SUBJECTS = 12


class ProbeDataset(PairDataset):
    """
    PairDataset, extended to also load and return the 'y' (task/class
    label) array already stored in every cached .npz file. Reuses ALL of
    PairDataset's indexing, subject and session filtering, and lazy
    per-subject caching -- only _load_subject and __getitem__ change.

    label_map: dict mapping raw label values (as they appear in 'y',
    e.g. numpy strings like "left_hand") to integer class ids. MUST be
    built once from ALL subjects (see build_label_map below) and passed
    identically to every ProbeDataset split -- building it per-split
    would risk train and val disagreeing on which int means which class
    if a class happens to be absent from one split.
    """

    def __init__(self, cache_dir, datasets, label_map, subjects=None,
                 sessions=None, dtype=torch.float32):
        self.label_map = label_map
        super().__init__(cache_dir, datasets, subjects=subjects,
                         sessions=sessions, dtype=dtype)

    def _load_subject(self, dataset_name, subject_id):
        key = (dataset_name, subject_id)
        if key not in self._cache:
            path = os.path.join(self.cache_dir, dataset_name, f"subject_{subject_id}.npz")
            with np.load(path, allow_pickle=False) as d:
                self._cache[key] = (d["X"], d["y"])
        return self._cache[key]

    def __getitem__(self, idx):
        dataset_name, subject_id, epoch_idx = self.index[idx]
        X, y = self._load_subject(dataset_name, subject_id)
        epoch = X[epoch_idx]
        raw_label = y[epoch_idx]
        label = self.label_map[raw_label]  # raw string/value -> int class id
        return torch.tensor(epoch, dtype=self.dtype), torch.tensor(label, dtype=torch.long)


def build_label_map(cache_dir, dataset_key, subjects_list):
    """
    Scans 'y' across ALL given subjects (not just train or val -- see
    ProbeDataset docstring for why) to build a consistent, sorted
    raw-label -> int mapping. Only reads the 'y' array per subject, not
    'X', so this is cheap even for many subjects.
    """
    labels = set()
    for subj in subjects_list:
        path = os.path.join(cache_dir, dataset_key, f"subject_{subj}.npz")
        with np.load(path, allow_pickle=False) as d:
            labels.update(np.unique(d["y"]).tolist())

    sorted_labels = sorted(labels, key=str)
    label_map = {lbl: i for i, lbl in enumerate(sorted_labels)}
    print(f"[probe] Discovered {len(label_map)} classes in '{dataset_key}': "
          f"{ {str(k): v for k, v in label_map.items()} }")
    return label_map


def list_available_subjects(cache_dir, dataset_key):
    """Mirrors the subject-listing logic in PairDataset._build_index,
    without touching any array data."""
    ds_dir = os.path.join(cache_dir, dataset_key)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Expected cache dir at {ds_dir}, not found.")
    return sorted(
        f[len("subject_"):-4]
        for f in os.listdir(ds_dir)
        if f.startswith("subject_") and f.endswith(".npz")
    )


# ---------------------------------------------------------------------------
# Pure-numpy helpers: folds and metrics (no torch -- unit-testable)
# ---------------------------------------------------------------------------
def make_folds(subjects, cv="auto", n_splits=5, val_fraction=0.2, seed=0,
               loso_max=LOSO_MAX_SUBJECTS):
    """
    Subject-DISJOINT folds. Returns (cv_used, [(train_subjects, val_subjects), ...]).

    cv="loso"   : one fold per subject (that subject is the validation set).
    cv="random" : n_splits random subject-disjoint splits, val_fraction of the
                  subjects held out each time (seeds seed, seed+1, ...).
    cv="auto"   : "loso" if there are <= loso_max subjects, else "random".
    Splitting by subject (not by epoch) matters: epochs from the same
    subject/session are correlated, so an epoch-level split would leak.
    """
    subjects = sorted(subjects)
    if len(subjects) < 2:
        raise ValueError(f"Need at least 2 subjects for a subject-disjoint "
                         f"split, got {len(subjects)}.")
    if cv == "auto":
        cv = "loso" if len(subjects) <= loso_max else "random"

    folds = []
    if cv == "loso":
        for s in subjects:
            folds.append(([x for x in subjects if x != s], [s]))
    elif cv == "random":
        n_val = min(max(1, int(round(len(subjects) * val_fraction))), len(subjects) - 1)
        for k in range(n_splits):
            rng = np.random.RandomState(seed + k)
            perm = rng.permutation(len(subjects))
            val = sorted(subjects[i] for i in perm[:n_val])
            val_set = set(val)
            train = [s for s in subjects if s not in val_set]
            folds.append((train, val))
    else:
        raise ValueError(f"Unknown cv mode '{cv}'")
    return cv, folds


def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall over the classes PRESENT in y_true. Chance is
    1/n_classes regardless of class imbalance; a majority-class predictor
    scores 1/n_present."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def majority_baseline(train_y, val_y):
    """Predict the TRAIN majority class for every val epoch. Returns
    (accuracy, balanced_accuracy)."""
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)
    maj = int(np.bincount(train_y).argmax())
    pred = np.full_like(val_y, maj)
    return float((pred == val_y).mean()), balanced_accuracy(val_y, pred)


# ---------------------------------------------------------------------------
# Linear probe head
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    """Single nn.Linear layer -- intentionally no hidden layers. The point of
    a linear probe is to measure what's linearly separable in the frozen
    embedding space, not what a probe network could learn on its own.

    Inputs are standardized INSIDE the module with (mu, sigma) fitted on the
    training embeddings (see set_standardization). Defaults are the identity,
    so an un-fitted probe behaves exactly like a bare nn.Linear. Because the
    standardization lives in the module, callers can still do probe(embeds)
    on raw embeddings."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        self.register_buffer("mu", torch.zeros(embed_dim))
        self.register_buffer("sigma", torch.ones(embed_dim))

    @torch.no_grad()
    def set_standardization(self, mean, std):
        std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-6)
        self.mu.copy_(mean.to(self.mu.device))
        self.sigma.copy_(std.to(self.sigma.device))

    def forward(self, x):
        return self.fc((x - self.mu) / self.sigma)


# ---------------------------------------------------------------------------
# Embedding extraction (frozen backbone, pre-projection)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(encoder, lsp_bank, dataset_key, dataloader, device,
                       embed_dim=64):
    """Run frozen LSPBank + encoder over a dataloader, return (embeddings, labels).

    Returns the PRE-PROJECTION 64-d encoder output `encoder(lsp(x))`. The
    projection head is deliberately not used: it only shapes the InfoNCE loss
    during Phase 0, and the state space for SSRS, DTKF and FiLM is the
    pre-projection representation. LSP and encoder are forced into eval mode
    (BatchNorm on running source statistics, dropout off) and no augmentation
    is applied.
    """
    encoder.eval()
    lsp_bank.eval()

    all_embeds, all_labels = [], []
    for raw_eeg, task_label in dataloader:
        raw_eeg = raw_eeg.to(device)              # (B, C_dataset, T)
        h = lsp_bank(raw_eeg, dataset_key)         # (B, 16, T)
        embed = encoder(h)                         # (B, 64) -- PRE-projection
        assert embed.shape[-1] == embed_dim, (
            f"Expected {embed_dim}-d pre-projection embeddings, got "
            f"{tuple(embed.shape)} -- is the projection head being applied?")

        all_embeds.append(embed.cpu())
        all_labels.append(task_label.cpu())

    return torch.cat(all_embeds), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Probe training loop
# ---------------------------------------------------------------------------
def train_probe(train_embeds, train_labels, val_embeds, val_labels,
                embed_dim, num_classes, device, epochs=100, lr=1e-3,
                standardize=True, class_weight=False, batch_size=64,
                verbose=True, seed=None):
    """
    Train the linear probe. Returns (probe, best_val_acc, final_val_acc) --
    unchanged return shape, so the subject and dataset probes keep working.

    standardize  : fit (mean, std) on the TRAIN embeddings and bake them into
                   the probe (default on).
    class_weight : inverse-frequency weighted cross-entropy (use for
                   imbalanced task labels).
    The val accuracy tracked here is plain accuracy; the task probe computes
    balanced accuracy itself from the final predictions.
    """
    if seed is not None:
        torch.manual_seed(seed)

    probe = LinearProbe(embed_dim, num_classes).to(device)
    train_x = train_embeds.to(device).float()
    train_y = train_labels.to(device).long()
    if standardize:
        probe.set_standardization(train_x.mean(dim=0), train_x.std(dim=0))

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    if class_weight:
        counts = torch.bincount(train_y, minlength=num_classes).float()
        w = counts.sum() / (num_classes * counts.clamp(min=1.0))
        w[counts == 0] = 0.0
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    n = train_x.shape[0]
    best_val_acc = 0.0
    final_val_acc = 0.0
    loss = torch.tensor(float("nan"))
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(probe(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()

        val_acc = evaluate_probe(probe, val_embeds, val_labels, device)
        best_val_acc = max(best_val_acc, val_acc)
        final_val_acc = val_acc

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    # NOTE: best_val_acc selects the max over epochs on the SAME val set
    # used for reporting -- a mild form of test-set selection that inflates
    # the headline number. final_val_acc (the last epoch, no selection) is
    # the figure the G3 verdict uses; both are returned for reference.
    return probe, best_val_acc, final_val_acc


@torch.no_grad()
def predict_probe(probe, embeds, device):
    probe.eval()
    logits = probe(embeds.to(device).float())
    return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def evaluate_probe(probe, embeds, labels, device):
    preds = predict_probe(probe, embeds, device)
    return float((preds == labels.cpu().numpy()).mean())


# ---------------------------------------------------------------------------
# Per-dataset G3 probe
# ---------------------------------------------------------------------------
def probe_one_dataset(dataset_key, lsp_bank, encoder, args, device, sessions):
    """Run the G3 task probe on one dataset. Returns a result dict."""
    print(f"\n{'=' * 60}\nG3 task probe -- {dataset_key}\n{'=' * 60}")

    # Label map built from ALL subjects, not per split.
    all_subjects = list_available_subjects(args.cache_dir, dataset_key)
    label_map = build_label_map(args.cache_dir, dataset_key, all_subjects)
    num_classes = len(label_map)
    class_names = {v: str(k) for k, v in label_map.items()}

    # ONE dataset over every subject (after session filtering); embeddings are
    # extracted once and re-split by subject for every fold.
    ds = ProbeDataset(cache_dir=args.cache_dir, datasets=[dataset_key],
                      label_map=label_map, sessions=sessions)
    subj_arr = np.array([s for (_ds, s, _i) in ds.index])
    loader = DataLoader(ds, batch_size=args.probe_batch_size, shuffle=False,
                        num_workers=0)   # shuffle=False: order == ds.index order
    print("Extracting pre-projection embeddings (all subjects, once)...")
    embeds, labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, loader, device, embed_dim=args.embed_dim)
    ds._cache.clear()
    y = labels.numpy()
    assert len(y) == len(subj_arr), "embedding / index length mismatch"
    print(f"embeddings: {tuple(embeds.shape)} from {len(np.unique(subj_arr))} subjects")

    # Class balance and the majority-class reference.
    counts = np.bincount(y, minlength=num_classes)
    print("Class counts (all probed epochs):")
    for c in range(num_classes):
        print(f"  {class_names[c]:<12} {counts[c]:>6}  ({counts[c] / counts.sum():.1%})")
    print(f"Majority-class share: {counts.max() / counts.sum():.4f}")

    cv_used, folds = make_folds(
        np.unique(subj_arr).tolist(), cv=args.cv, n_splits=args.n_splits,
        val_fraction=args.val_fraction, seed=args.split_seed,
        loso_max=args.loso_max)
    print(f"\nCV: {cv_used}, {len(folds)} subject-disjoint folds "
          f"(standardize={not args.no_standardize}, "
          f"class_weight={not args.no_class_weight})")

    fold_rows = []
    pooled_true, pooled_pred = [], []
    for k, (train_subj, val_subj) in enumerate(folds):
        tr = np.isin(subj_arr, train_subj)
        va = np.isin(subj_arr, val_subj)
        tr_x, tr_y = embeds[tr], labels[tr]
        va_x, va_y = embeds[va], labels[va]

        probe, best_acc, final_acc = train_probe(
            tr_x, tr_y, va_x, va_y, embed_dim=args.embed_dim,
            num_classes=num_classes, device=device, epochs=args.epochs,
            lr=args.lr, standardize=not args.no_standardize,
            class_weight=not args.no_class_weight, verbose=False,
            seed=args.split_seed + k)
        pred = predict_probe(probe, va_x, device)
        va_np = va_y.numpy()
        bal = balanced_accuracy(va_np, pred)
        maj_acc, maj_bal = majority_baseline(tr_y.numpy(), va_np)
        fold_rows.append({"val_subjects": list(val_subj), "n_val": int(va.sum()),
                          "acc": final_acc, "bal_acc": bal,
                          "majority_acc": maj_acc, "majority_bal_acc": maj_bal})
        pooled_true.append(va_np)
        pooled_pred.append(pred)
        print(f"  fold {k + 1:>2}/{len(folds)}  val={','.join(map(str, val_subj)):<14} "
              f"n={int(va.sum()):>5}  acc={final_acc:.3f}  bal_acc={bal:.3f}  "
              f"majority_acc={maj_acc:.3f}")

    def mean_std(key):
        v = np.array([r[key] for r in fold_rows], dtype=float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        return float(v.mean()), sd

    acc_m, acc_s = mean_std("acc")
    bal_m, bal_s = mean_std("bal_acc")
    maj_m, _ = mean_std("majority_acc")

    metric_key = "bal_acc" if args.verdict_metric == "balanced" else "acc"
    score_m, score_s = (bal_m, bal_s) if metric_key == "bal_acc" else (acc_m, acc_s)
    sem = score_s / np.sqrt(len(fold_rows)) if len(fold_rows) > 1 else float("nan")

    chance = 1.0 / num_classes
    threshold = chance + G3_MARGIN
    passed = score_m >= threshold
    borderline = (not np.isnan(sem)) and abs(score_m - threshold) < sem

    # Pooled per-class recall (over all folds' held-out predictions).
    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    recall = {class_names[c]: float((pp[pt == c] == c).mean())
              for c in range(num_classes) if (pt == c).any()}

    print(f"\n--- {dataset_key}: G3 task probe ({len(folds)} folds, {cv_used}) ---")
    print(f"Classes:                 {num_classes}")
    print(f"Chance level:            {chance:.4f}")
    print(f"G3 threshold:            {threshold:.4f} (chance + {G3_MARGIN * 100:.0f} pp)")
    print(f"Accuracy:                {acc_m:.4f} +/- {acc_s:.4f}")
    print(f"Balanced accuracy:       {bal_m:.4f} +/- {bal_s:.4f}")
    print(f"Majority-class acc:      {maj_m:.4f}  (predict the train majority class)")
    print(f"Accuracy over majority:  {acc_m - maj_m:+.4f}")
    print(f"Verdict metric:          {args.verdict_metric} = {score_m:.4f} "
          f"(SEM {sem:.4f})" if not np.isnan(sem) else
          f"Verdict metric:          {args.verdict_metric} = {score_m:.4f}")
    print("Pooled per-class recall: " +
          "  ".join(f"{k}={v:.3f}" for k, v in recall.items()))
    print(f"G3 ({dataset_key}): {'PASS' if passed else 'FAIL'}"
          f"{'  [borderline: within 1 SEM of the threshold]' if borderline else ''}")

    return {
        "dataset": dataset_key, "num_classes": num_classes, "chance": chance,
        "threshold": threshold, "cv": cv_used, "n_folds": len(folds),
        "verdict_metric": args.verdict_metric, "score": score_m, "score_std": score_s,
        "score_sem": None if np.isnan(sem) else float(sem),
        "acc": acc_m, "acc_std": acc_s, "bal_acc": bal_m, "bal_acc_std": bal_s,
        "majority_acc": maj_m, "class_counts": {class_names[c]: int(counts[c])
                                                 for c in range(num_classes)},
        "pooled_recall": recall, "folds": fold_rows,
        "n_subjects": int(len(np.unique(subj_arr))), "n_epochs": int(len(y)),
        "passed": bool(passed), "borderline": bool(borderline),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="G3 task probe on frozen, pre-projection Phase 0 embeddings")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to Phase 0 checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Probe only this dataset (default: every dataset "
                              "in the checkpoint, each reported separately)")
    parser.add_argument("--embed_dim", type=int, default=64,
                         help="Matches Phase0Encoder's fixed 64-d output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR,
                         help="Defaults to the current pipeline version's "
                              "cache dir (see cache_config.py).")
    parser.add_argument("--cv", choices=["auto", "loso", "random"], default="auto",
                         help="auto (default): leave-one-subject-out when the "
                              "dataset has <= --loso_max subjects, else repeated "
                              "random subject-disjoint splits.")
    parser.add_argument("--loso_max", type=int, default=LOSO_MAX_SUBJECTS)
    parser.add_argument("--n_splits", type=int, default=5,
                         help="Number of random subject-disjoint splits (cv=random)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of subjects held out per random split")
    parser.add_argument("--split_seed", type=int, default=0,
                         help="Base seed (random splits use seed, seed+1, ...)")
    parser.add_argument("--verdict_metric", choices=["balanced", "accuracy"],
                         default="balanced",
                         help="Metric compared with chance + 5 pp. balanced "
                              "(default) = mean per-class recall; accuracy = "
                              "the old plain accuracy.")
    parser.add_argument("--no_standardize", action="store_true",
                         help="Disable embedding standardization (old behaviour)")
    parser.add_argument("--no_class_weight", action="store_true",
                         help="Disable class-weighted loss (old behaviour)")
    parser.add_argument("--probe_batch_size", type=int, default=128,
                         help="Batch size for embedding EXTRACTION")
    parser.add_argument("--probe_sessions", choices=["train", "all"], default="train",
                         help="train (default): only the sessions Phase 0 was "
                              "allowed to see (checkpoint's train_sessions). "
                              "all: every session.")
    parser.add_argument("--json_out", type=str, default=None,
                         help="Write the results to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using cache_dir: {args.cache_dir}")

    # -----------------------------------------------------------------
    # 1. Load frozen Phase 0 LSPBank + encoder (projection head NOT loaded)
    # -----------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # weights_only=False because the checkpoint holds plain Python objects
    # (channel-name lists, session dicts). Only load checkpoints you trust.
    if "dataset_keys" in ckpt:                                   # pooled format
        dataset_keys = list(ckpt["dataset_keys"])
        channel_names_by_dataset = ckpt["channel_names_by_dataset"]
    else:                                                        # legacy single-dataset
        dataset_keys = [ckpt["dataset_key"]]
        channel_names_by_dataset = {ckpt["dataset_key"]: ckpt["channel_names"]}

    lsp_bank = LSPBank({d: channel_names_by_dataset[d] for d in dataset_keys},
                       n_shared=16).to(device)
    lsp_bank.load_state_dict(ckpt["lsp_state_dict"])
    encoder = Phase0Encoder(n_channels=16, sample_rate=250).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for module in (lsp_bank, encoder):
        module.eval()                       # frozen + eval mode (BN on source stats)
        for p in module.parameters():
            p.requires_grad = False

    train_sessions = ckpt.get("train_sessions")
    sessions = train_sessions if args.probe_sessions == "train" else None
    print(f"Loaded checkpoint with datasets {dataset_keys}, "
          f"final Phase 0 loss: {ckpt.get('final_loss', 'n/a')}")
    print(f"Probe sessions: {sessions if sessions else 'all'} "
          f"(checkpoint train_sessions: {train_sessions})")
    print("Probing PRE-PROJECTION 64-d encoder embeddings (projection head not used).")

    to_probe = [args.dataset] if args.dataset else dataset_keys
    for d in to_probe:
        if d not in dataset_keys:
            raise ValueError(f"Dataset '{d}' is not in the checkpoint ({dataset_keys}).")

    results = [probe_one_dataset(d, lsp_bank, encoder, args, device, sessions)
               for d in to_probe]

    # -----------------------------------------------------------------
    # G3 summary
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}\nG3 SUMMARY (task probe, pre-projection, subject-disjoint folds)\n{'=' * 60}")
    for r in results:
        note = "  [borderline]" if r["borderline"] else ""
        print(f"  {r['dataset']:<10} {r['verdict_metric']} {r['score']:.4f} +/- {r['score_std']:.4f} "
              f"vs threshold {r['threshold']:.4f} "
              f"(chance {r['chance']:.2f}, {r['num_classes']} classes, "
              f"{r['n_folds']} {r['cv']} folds; acc {r['acc']:.3f}, "
              f"majority {r['majority_acc']:.3f}) -> "
              f"{'PASS' if r['passed'] else 'FAIL'}{note}")
    failed = [r["dataset"] for r in results if not r["passed"]]
    if not failed:
        print("\nG3: PASS (every probed dataset is at least chance + 5 pp).")
    else:
        print(f"\nG3: FAIL on {failed}. Per the risk table: Phase 0 learned a subject "
              f"shortcut -- fix augmentation, delay Phase 1.")
    print("The subject-ID probe is a diagnostic only (not a gate): run "
          "probe_subject_leakage.py and plot it against this result for ablation 12.")

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint, "space": "pre_projection_64d",
            "split": f"subject_disjoint_{args.cv}", "probe_sessions": args.probe_sessions,
            "train_sessions": train_sessions, "verdict_metric": args.verdict_metric,
            "results": results,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()