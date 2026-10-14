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
    Pass : final validation accuracy >= chance + 5 percentage points,
           where chance = 1 / (number of classes in THAT dataset).
    Fail : near chance. That is the failure signal -- it means Phase 0 did not
           keep task structure, and the risk table says to fix augmentation
           and delay Phase 1.
    The accuracy reported is the LAST epoch's, not the best over epochs
    (selecting the max on the reporting set would inflate it).
    The spec does not define how to aggregate across datasets when a pooled
    checkpoint is probed: this script reports each dataset separately and
    prints an overall verdict that requires every probed dataset to pass.

    The subject-ID probe (probe_subject_leakage.py) is NOT part of this gate.
    Per spec rev A it is a diagnostic only (plotted against this probe for
    ablation 12): a collapsed subject probe is a warning, a very high one
    alone is not a failure.

SESSIONS
    By default the probe reads only the sessions Phase 0 was allowed to train on
    (the checkpoint's `train_sessions`), so it never touches a held-out test
    session. Use --probe_sessions all to include every session.

Usage:
    python probe_task_decodability.py --checkpoint phase0_checkpoint_<...>.pt
    python probe_task_decodability.py --checkpoint <ckpt> --dataset bci_iv_2a

Accepts the pooled checkpoint format (`dataset_keys`,
`channel_names_by_dataset`) and the older single-dataset format
(`dataset_key`, `channel_names`).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.dirname(__file__))
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset
from cache_config import CACHE_DIR  # shared, versioned cache path --
                                      # stdlib-only import, no moabb
                                      # dependency pulled in just to
                                      # resolve where the cache lives.

# G3: pass if final val accuracy >= chance + this margin (5 percentage points).
G3_MARGIN = 0.05


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
    without touching any array data -- just used to decide the
    train/val subject split before building the real datasets."""
    ds_dir = os.path.join(cache_dir, dataset_key)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Expected cache dir at {ds_dir}, not found.")
    return sorted(
        f[len("subject_"):-4]
        for f in os.listdir(ds_dir)
        if f.startswith("subject_") and f.endswith(".npz")
    )


def split_subjects_train_val(cache_dir, dataset_key, val_fraction=0.2, seed=0):
    """
    Subject-DISJOINT train/val split (the G3 split). Splitting by subject
    rather than by epoch matters here: epochs from the same subject/session
    are correlated, so an epoch-level split would leak information across
    train/val and inflate the probe's accuracy.
    """
    subjects = list_available_subjects(cache_dir, dataset_key)
    rng = np.random.RandomState(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_subjects = sorted(shuffled[:n_val])
    train_subjects = sorted(shuffled[n_val:])

    print(f"[probe] {dataset_key}: {len(train_subjects)} train subjects "
          f"{train_subjects}, {len(val_subjects)} val subjects {val_subjects}")
    return train_subjects, val_subjects


# ---------------------------------------------------------------------------
# Linear probe head
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    """Single nn.Linear layer -- intentionally no hidden layers. The point of
    a linear probe is to measure what's linearly separable in the frozen
    embedding space, not what a probe network could learn on its own."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


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
                 embed_dim, num_classes, device, epochs=100, lr=1e-3):
    probe = LinearProbe(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_embeds, train_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_acc = 0.0
    final_val_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_probe(probe, val_embeds, val_labels, device)
        best_val_acc = max(best_val_acc, val_acc)
        final_val_acc = val_acc

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    # NOTE: best_val_acc selects the max over epochs on the SAME val set
    # used for reporting -- a mild form of test-set selection that inflates
    # the headline number. final_val_acc (the last epoch, no selection) is
    # the figure the G3 verdict uses; both are returned for reference.
    return probe, best_val_acc, final_val_acc


@torch.no_grad()
def evaluate_probe(probe, embeds, labels, device):
    probe.eval()
    embeds, labels = embeds.to(device), labels.to(device)
    logits = probe(embeds)
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Per-dataset G3 probe
# ---------------------------------------------------------------------------
def probe_one_dataset(dataset_key, lsp_bank, encoder, args, device, sessions):
    """Run the G3 task probe on one dataset. Returns a result dict."""
    print(f"\n{'=' * 60}\nG3 task probe -- {dataset_key}\n{'=' * 60}")

    train_subjects, val_subjects = split_subjects_train_val(
        args.cache_dir, dataset_key, val_fraction=args.val_fraction, seed=args.split_seed)

    # Label map built from ALL subjects (train + val together), not per split.
    all_subjects = sorted(train_subjects + val_subjects)
    label_map = build_label_map(args.cache_dir, dataset_key, all_subjects)
    num_classes = len(label_map)

    train_dataset = ProbeDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key], label_map=label_map,
        subjects={dataset_key: train_subjects}, sessions=sessions)
    val_dataset = ProbeDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key], label_map=label_map,
        subjects={dataset_key: val_subjects}, sessions=sessions)

    train_loader = DataLoader(train_dataset, batch_size=args.probe_batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.probe_batch_size,
                             shuffle=False, num_workers=0)

    print("Extracting pre-projection embeddings (train split)...")
    train_embeds, train_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, train_loader, device, embed_dim=args.embed_dim)
    print("Extracting pre-projection embeddings (val split)...")
    val_embeds, val_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, val_loader, device, embed_dim=args.embed_dim)
    print(f"train_embeds: {tuple(train_embeds.shape)}, val_embeds: {tuple(val_embeds.shape)}")

    print("\nTraining task/class decodability probe...")
    _probe, best_val_acc, final_val_acc = train_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        embed_dim=args.embed_dim, num_classes=num_classes,
        device=device, epochs=args.epochs, lr=args.lr,
    )

    chance = 1.0 / num_classes
    threshold = chance + G3_MARGIN
    passed = final_val_acc >= threshold
    print(f"\n--- {dataset_key}: G3 task probe ---")
    print(f"Classes:            {num_classes}")
    print(f"Chance level:       {chance:.4f}")
    print(f"G3 threshold:       {threshold:.4f} (chance + {G3_MARGIN * 100:.0f} pp)")
    print(f"Best val accuracy:  {best_val_acc:.4f} (max over epochs -- optimistic, reference only)")
    print(f"Final val accuracy: {final_val_acc:.4f} (last epoch -- used for the verdict)")
    print(f"Above chance by:    {final_val_acc - chance:+.4f}")
    print(f"G3 ({dataset_key}): {'PASS' if passed else 'FAIL'}")
    return {
        "dataset": dataset_key, "num_classes": num_classes, "chance": chance,
        "threshold": threshold, "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc, "n_val_subjects": len(val_subjects),
        "n_val_epochs": int(val_embeds.shape[0]), "passed": passed,
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
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of subjects held out for val (subject-disjoint split)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_sessions", choices=["train", "all"], default="train",
                         help="train (default): only the sessions Phase 0 was "
                              "allowed to see (checkpoint's train_sessions). "
                              "all: every session.")
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
    print(f"\n{'=' * 60}\nG3 SUMMARY (task probe, pre-projection, subject-disjoint split)\n{'=' * 60}")
    for r in results:
        note = "  [few val subjects -- noisy]" if r["n_val_subjects"] < 3 else ""
        print(f"  {r['dataset']:<10} acc {r['final_val_acc']:.4f} vs threshold "
              f"{r['threshold']:.4f} (chance {r['chance']:.2f}, {r['num_classes']} classes, "
              f"{r['n_val_subjects']} val subjects) -> {'PASS' if r['passed'] else 'FAIL'}{note}")
    failed = [r["dataset"] for r in results if not r["passed"]]
    if not failed:
        print("\nG3: PASS (every probed dataset is at least chance + 5 pp).")
    else:
        print(f"\nG3: FAIL on {failed}. Per the risk table: Phase 0 learned a subject "
              f"shortcut -- fix augmentation, delay Phase 1.")
    print("The subject-ID probe is a diagnostic only (not a gate): run "
          "probe_subject_leakage.py and plot it against this result for ablation 12.")


if __name__ == "__main__":
    main()