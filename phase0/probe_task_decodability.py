"""
probe_task_decodability.py

Gate 1 diagnostic (part 1 of 2): Task/class decodability probe.

Freezes the Phase 0 encoder + LSP, extracts embeddings for a held-out split,
and trains a single linear layer (nn.Linear, no hidden layers) to predict the
trial/task label (e.g. motor imagery class for BCI-IV 2a: left hand, right
hand, feet, tongue -> 4 classes).

We want this HIGH. Low accuracy here means the Phase 0 embedding space isn't
retaining task-relevant signal, and there's no point moving on to Module A/B/C
until this is fixed (more pretraining, augmentation issues, etc.)

Usage:
    python phase0\\probe_task_decodability.py --checkpoint phase0_checkpoint_bci_iv_2a.pt

Run this alongside probe_subject_leakage.py -- Gate 1 requires BOTH:
  - high accuracy here
  - low accuracy on the subject-ID probe
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.dirname(__file__))
# Confirmed against train_phase0.py -- these are the real class names.
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset


class ProbeDataset(PairDataset):
    """
    PairDataset, extended to also load and return the 'y' (task/class
    label) array that's already sitting in every cached .npz file --
    sampler.py's own docstring confirms 'y' is there, just unused by
    Phase 0 pretraining. Reuses ALL of PairDataset's indexing,
    subject-filtering (via the `subjects=` constructor arg), and lazy
    per-subject caching -- only _load_subject and __getitem__ change.

    label_map: dict mapping raw label values (as they appear in 'y',
    e.g. numpy strings like "left_hand") to integer class ids. MUST be
    built once from ALL subjects (see build_label_map below) and passed
    identically to every ProbeDataset split -- building it per-split
    would risk train and val disagreeing on which int means which class
    if a class happens to be absent from one split.
    """

    def __init__(self, cache_dir, datasets, label_map, subjects=None, dtype=torch.float32):
        self.label_map = label_map
        super().__init__(cache_dir, datasets, subjects=subjects, dtype=dtype)

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
    Subject-DISJOINT train/val split. Splitting by subject rather than
    by epoch matters here: epochs from the same subject/session are
    correlated, so an epoch-level split would leak information across
    train/val and inflate the probe's accuracy -- exactly the trap
    flagged earlier for this probe.
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
# Embedding extraction (frozen backbone)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(encoder, lsp_bank, dataset_key, dataloader, device):
    """Run frozen LSPBank+encoder over a dataloader, return (embeddings, labels).

    Deliberately probes `encoder(h)` -- the 64-d encoder output -- NOT
    `proj_head(e)`. The projection head only exists to shape the InfoNCE
    loss during Phase 0 pretraining and is conventionally discarded
    afterward; the representation that matters for Gate 1 (and for
    everything downstream in Modules A/B/C) is the encoder output.
    train_phase0.py never applies test-time augmentation, so eval here
    also skips augment_batch() -- probe on the clean, unaugmented signal.

    NOTE: still requires wiring to your actual label source. See
    dataloader construction in main() below -- train_phase0.py's
    dataset.index only carries (dataset_name, subject_id, epoch_idx),
    with no task/class label. That label must live somewhere else in
    your cache layer (per sampler.py) -- point this function's
    dataloader at whatever yields (raw_eeg, task_label) pairs once
    that's confirmed.
    """
    encoder.eval()
    lsp_bank.eval()

    all_embeds, all_labels = [], []
    for raw_eeg, task_label in dataloader:
        raw_eeg = raw_eeg.to(device)              # (B, C_dataset, T)
        h = lsp_bank(raw_eeg, dataset_key)         # (B, 16, T)
        embed = encoder(h)                         # (B, 64) -- pre-projection-head

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

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    return probe, best_val_acc


@torch.no_grad()
def evaluate_probe(probe, embeds, labels, device):
    probe.eval()
    embeds, labels = embeds.to(device), labels.to(device)
    logits = probe(embeds)
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to Phase 0 checkpoint (.pt), e.g. "
                              "phase0_checkpoint_bci_iv_2a.pt")
    parser.add_argument("--num_classes", type=int, default=4,
                         help="4 for BCI-IV 2a motor imagery (LH/RH/feet/tongue), "
                              "2 for BCI-IV 2b (LH/RH only)")
    parser.add_argument("--embed_dim", type=int, default=64,
                         help="Matches Phase0Encoder's fixed 64-d output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of subjects held out for val (subject-disjoint split)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # 1. Load frozen Phase 0 LSPBank + encoder from checkpoint
    # -----------------------------------------------------------------
    # Confirmed against train_phase0.py's checkpoint dict: keys are
    # 'lsp_state_dict', 'encoder_state_dict', 'projection_head_state_dict'
    # (projection head intentionally not loaded here -- see note above).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # NOTE: weights_only=False because this checkpoint's 'channel_names'
    # value is a plain Python list, and older torch.save/load pairings
    # need weights_only=False to unpickle that alongside the state dicts.
    # Only load checkpoints you trust (i.e. ones you or a teammate produced).
    dataset_key = ckpt["dataset_key"]          # e.g. "bci_iv_2a"
    channel_names = ckpt["channel_names"]      # real channel names used at train time

    lsp_bank = LSPBank({dataset_key: channel_names}, n_shared=16).to(device)
    lsp_bank.load_state_dict(ckpt["lsp_state_dict"])
    for p in lsp_bank.parameters():
        p.requires_grad = False

    encoder = Phase0Encoder(n_channels=16, sample_rate=250).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Loaded checkpoint trained on '{dataset_key}', "
          f"final Phase 0 loss: {ckpt.get('final_loss', 'n/a')}")

    # -----------------------------------------------------------------
    # 2. Subject-disjoint train/val split, then build labeled dataloaders
    # -----------------------------------------------------------------
    train_subjects, val_subjects = split_subjects_train_val(
        args.cache_dir, dataset_key, val_fraction=args.val_fraction, seed=args.split_seed)

    # Label map built from ALL subjects (train + val together), not per
    # split -- see ProbeDataset/build_label_map docstrings for why.
    all_subjects = sorted(train_subjects + val_subjects)
    label_map = build_label_map(args.cache_dir, dataset_key, all_subjects)
    num_classes = len(label_map)
    if num_classes != args.num_classes:
        print(f"[probe] NOTE: discovered {num_classes} classes in the data, "
              f"overriding --num_classes={args.num_classes}")

    train_dataset = ProbeDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key], label_map=label_map,
        subjects={dataset_key: train_subjects})
    val_dataset = ProbeDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key], label_map=label_map,
        subjects={dataset_key: val_subjects})

    train_loader = DataLoader(train_dataset, batch_size=args.probe_batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.probe_batch_size,
                             shuffle=False, num_workers=0)

    # -----------------------------------------------------------------
    # 3. Extract frozen embeddings
    # -----------------------------------------------------------------
    print("Extracting embeddings (train split)...")
    train_embeds, train_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, train_loader, device)
    print("Extracting embeddings (val split)...")
    val_embeds, val_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, val_loader, device)

    print(f"train_embeds: {train_embeds.shape}, val_embeds: {val_embeds.shape}")

    # -----------------------------------------------------------------
    # 4. Train the linear probe
    # -----------------------------------------------------------------
    print("\nTraining task/class decodability probe...")
    probe, best_val_acc = train_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        embed_dim=args.embed_dim, num_classes=num_classes,
        device=device, epochs=args.epochs, lr=args.lr,
    )

    chance = 1.0 / num_classes
    print(f"\n=== Task/class decodability probe result ===")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Chance level:      {chance:.4f}")
    print(f"Above chance by:   {best_val_acc - chance:+.4f}")
    print("\nGate 1 needs this HIGH alongside a LOW subject-ID leakage probe result.")
    print("Run probe_subject_leakage.py next if you haven't already.")


if __name__ == "__main__":
    main()