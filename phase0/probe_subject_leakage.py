"""
probe_subject_leakage.py

Gate 1 diagnostic (part 2 of 2): Subject-ID leakage probe.

Freezes the same Phase 0 encoder + LSP as probe_task_decodability.py, but
trains the linear probe to predict WHICH SUBJECT a clip came from instead
of its task/class label.

We want this LOW -- ideally close to chance (1 / n_subjects). High subject
decodability means the embedding space is using subject identity as a
shortcut, which is exactly what InfoNCE's subject-balanced negative
sampling (losses.py excluding same-subject-different-clip pairs) was
designed to prevent.

CRITICAL DIFFERENCE from probe_task_decodability.py's split strategy:
That probe used a SUBJECT-DISJOINT split (whole subjects held out) because
it was testing cross-subject generalization of task signal. This probe
does the OPPOSITE: an EPOCH-level split, stratified so every subject
appears in both train and val. That's not a stylistic choice -- the
classes here ARE the subjects, so a subject-disjoint split would be
nonsensical (you can't train a probe to recognize a subject it never saw
labeled examples of, then ask it to recognize that same subject in val).

Usage:
    python phase0\\probe_subject_leakage.py --checkpoint phase0_checkpoint_bci_iv_2a.pt

Run alongside probe_task_decodability.py -- Gate 1 requires BOTH:
  - high accuracy on that probe (task signal retained)
  - low accuracy on THIS probe (no subject-identity shortcut)
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.append(os.path.dirname(__file__))
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset

# Reuse shared probe machinery rather than duplicating it -- keeps both
# probes' LinearProbe/train loop/eval logic identical by construction,
# so any future fix to one automatically applies to both.
from probe_task_decodability import LinearProbe, train_probe, extract_embeddings


class SubjectLabelDataset(PairDataset):
    """
    PairDataset, extended to return (epoch, subject_int_label) instead of
    just epoch. Unlike ProbeDataset in probe_task_decodability.py, this
    does NOT need to touch the 'y' array in the cache at all -- subject
    identity is already sitting right there in self.index (built by
    PairDataset itself), so _load_subject is inherited UNCHANGED from
    PairDataset (loads only 'X'). Only __getitem__ needs to change.

    subject_label_map: dict {(dataset_name, subject_id): int}, built
    once from ALL subjects via build_subject_label_map below, and shared
    identically across the whole dataset here (there's only ONE dataset
    instance for this probe, not a train/val pair -- see split_positions
    _stratified for how train/val positions are carved out of it).
    """

    def __init__(self, cache_dir, datasets, subject_label_map, subjects=None, dtype=torch.float32):
        self.subject_label_map = subject_label_map
        super().__init__(cache_dir, datasets, subjects=subjects, dtype=dtype)

    def __getitem__(self, idx):
        dataset_name, subject_id, epoch_idx = self.index[idx]
        arr = self._load_subject(dataset_name, subject_id)  # just X, inherited as-is
        epoch = arr[epoch_idx]
        label = self.subject_label_map[(dataset_name, subject_id)]
        return torch.tensor(epoch, dtype=self.dtype), torch.tensor(label, dtype=torch.long)


def build_subject_label_map(dataset):
    """
    Sorted (dataset_name, subject_id) -> int mapping, built from every
    subject actually present in `dataset`. Sorted for reproducibility
    across runs, not for any semantic reason (subject-id ordering here
    is arbitrary, same as build_subject_id_tensor in train_phase0.py).
    """
    keys = sorted(dataset.subject_groups().keys())
    label_map = {k: i for i, k in enumerate(keys)}
    print(f"[probe] {len(label_map)} subjects -> int labels: "
          f"{ {str(k): v for k, v in label_map.items()} }")
    return label_map


def split_positions_stratified(dataset, val_fraction=0.2, seed=0):
    """
    EPOCH-level split, stratified per subject: every subject's own clips
    get shuffled and split individually, so every subject ends up
    represented in BOTH train and val. This is the deliberate opposite
    of probe_task_decodability.py's subject-disjoint split -- see module
    docstring for why that split would be wrong here.
    """
    groups = dataset.subject_groups()  # {(ds, subj): [positions]}
    rng = np.random.RandomState(seed)

    train_positions, val_positions = [], []
    for key, positions in groups.items():
        positions = list(positions)
        rng.shuffle(positions)
        n_val = max(1, int(round(len(positions) * val_fraction)))
        val_positions.extend(positions[:n_val])
        train_positions.extend(positions[n_val:])

    print(f"[probe] stratified split: {len(train_positions)} train epochs, "
          f"{len(val_positions)} val epochs, {len(groups)} subjects "
          f"represented in both")
    return train_positions, val_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to Phase 0 checkpoint (.pt), e.g. "
                              "phase0_checkpoint_bci_iv_2a.pt")
    parser.add_argument("--embed_dim", type=int, default=64,
                         help="Matches Phase0Encoder's fixed 64-d output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of EACH subject's epochs held out "
                              "for val (stratified, not subject-disjoint)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # 1. Load frozen Phase 0 LSPBank + encoder from checkpoint
    # -----------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset_key = ckpt["dataset_key"]
    channel_names = ckpt["channel_names"]

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
    # 2. Build subject-label map, then stratified epoch-level split
    # -----------------------------------------------------------------
    # ONE dataset instance covering ALL subjects (no subjects= filtering
    # like the task probe -- every subject needs to appear in this
    # probe's train set to have any chance of being recognized in val).
    full_dataset = PairDataset(cache_dir=args.cache_dir, datasets=[dataset_key])
    subject_label_map = build_subject_label_map(full_dataset)
    num_subjects = len(subject_label_map)

    labeled_dataset = SubjectLabelDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key],
        subject_label_map=subject_label_map)

    train_positions, val_positions = split_positions_stratified(
        full_dataset, val_fraction=args.val_fraction, seed=args.split_seed)

    train_loader = DataLoader(Subset(labeled_dataset, train_positions),
                               batch_size=args.probe_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(labeled_dataset, val_positions),
                             batch_size=args.probe_batch_size, shuffle=False, num_workers=0)

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
    print("\nTraining subject-ID leakage probe...")
    probe, best_val_acc = train_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        embed_dim=args.embed_dim, num_classes=num_subjects,
        device=device, epochs=args.epochs, lr=args.lr,
    )

    chance = 1.0 / num_subjects
    print(f"\n=== Subject-ID leakage probe result ===")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Chance level:      {chance:.4f}")
    print(f"Above chance by:   {best_val_acc - chance:+.4f}")
    print("\nGate 1 needs this LOW alongside a HIGH task/class decodability result.")
    print("(Compare against probe_task_decodability.py's output.)")


if __name__ == "__main__":
    main()