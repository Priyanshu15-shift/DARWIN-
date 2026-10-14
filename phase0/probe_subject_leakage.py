"""
probe_subject_leakage.py

Subject-ID probe: a DIAGNOSTIC, not a gate (DARWIN spec rev A, section 9).
It pairs with the G3 task probe (probe_task_decodability.py) and is plotted
against it for ablation 12 (subject probe vs task probe Pareto).

WHAT IT MEASURES
    How linearly separable subject identity is in the frozen Phase 0
    embeddings. Run on pre-projection embeddings: the same 64-d encoder
    output the task probe reads and the state space that SSRS, DTKF and FiLM
    use. The projection head (64 -> 128 -> 64) is NEVER applied; it exists
    only to shape the InfoNCE loss. LSP and encoder are frozen and in eval
    mode (BatchNorm on source running statistics), with no augmentation.

HOW TO READ IT (diagnostic, no pass/fail)
    SSRS matches a new user to donor subjects by embedding similarity, so it
    needs subject structure to exist in this space.
      - Collapsed (near chance): WARNING. SSRS would have nothing to match
        against. This script flags it.
      - Very high on its own: NOT a failure. Informational only. Read it
        together with the task probe: the concern is a subject shortcut
        (high subject-ID AND near-chance task probe), and that call belongs to
        G3, not to this script.
    This probe never gates anything. Only the G3 task probe does.

SPLIT: EPOCH-LEVEL, STRATIFIED (deliberately different from the task probe)
    The classes here ARE the subjects, so a subject-disjoint split would be
    nonsensical. Every subject's epochs are split individually so each subject
    appears in both train and val. Caveat: epochs from the same session are
    correlated, so this split is optimistic about how identifiable a subject
    is. Treat the number as an upper-bound diagnostic.

SESSIONS
    By default only the sessions Phase 0 was allowed to train on (the
    checkpoint's `train_sessions`) are used. --probe_sessions all uses every
    session.

POOLED CHECKPOINTS
    Each dataset is probed separately (its own LSP matrix, its own subjects,
    its own chance = 1 / n_subjects).

Usage:
    python probe_subject_leakage.py --checkpoint phase0_checkpoint_<...>.pt
    python probe_subject_leakage.py --checkpoint <ckpt> --json_out subj_probe.json
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.append(os.path.dirname(__file__))
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset
from cache_config import CACHE_DIR  # shared, versioned cache path --
                                      # stdlib-only import, no moabb
                                      # dependency pulled in just to
                                      # resolve where the cache lives.

# Reuse shared probe machinery rather than duplicating it -- keeps both
# probes' LinearProbe/train loop/eval logic identical by construction.
from probe_task_decodability import LinearProbe, train_probe, extract_embeddings

# Diagnostic flag only: a subject probe within this margin of chance is
# reported as "collapsed" (a warning). Not a pass/fail threshold.
COLLAPSE_MARGIN = 0.05


class SubjectLabelDataset(PairDataset):
    """
    PairDataset, extended to return (epoch, subject_int_label). Does not touch
    the 'y' array: subject identity is already in self.index, so
    _load_subject is inherited unchanged (loads only 'X').

    subject_label_map: dict {(dataset_name, subject_id): int}, built once from
    this dataset via build_subject_label_map below.
    """

    def __init__(self, cache_dir, datasets, subject_label_map=None, subjects=None,
                 sessions=None, dtype=torch.float32):
        self.subject_label_map = subject_label_map
        super().__init__(cache_dir, datasets, subjects=subjects,
                         sessions=sessions, dtype=dtype)

    def __getitem__(self, idx):
        dataset_name, subject_id, epoch_idx = self.index[idx]
        arr = self._load_subject(dataset_name, subject_id)  # just X, inherited as-is
        epoch = arr[epoch_idx]
        label = self.subject_label_map[(dataset_name, subject_id)]
        return torch.tensor(epoch, dtype=self.dtype), torch.tensor(label, dtype=torch.long)


def build_subject_label_map(dataset):
    """
    Sorted (dataset_name, subject_id) -> int mapping, built from every subject
    present in `dataset` (after any session filtering). Sorted for
    reproducibility only; the ordering has no meaning.
    """
    keys = sorted(dataset.subject_groups().keys())
    label_map = {k: i for i, k in enumerate(keys)}
    print(f"[probe] {len(label_map)} subjects -> int labels")
    return label_map


def split_positions_stratified(dataset, val_fraction=0.2, seed=0):
    """
    EPOCH-level split, stratified per subject: every subject's own clips are
    shuffled and split individually, so every subject appears in BOTH train and
    val. See the module docstring for why this differs from the task probe.
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


def probe_one_dataset(dataset_key, lsp_bank, encoder, args, device, sessions):
    """Run the subject-ID diagnostic on one dataset. Returns a result dict."""
    print(f"\n{'=' * 60}\nSubject-ID probe (diagnostic) -- {dataset_key}\n{'=' * 60}")

    # ONE dataset instance covering ALL subjects (every subject must appear in
    # the probe's train set to be recognisable in val).
    probe_dataset = SubjectLabelDataset(
        cache_dir=args.cache_dir, datasets=[dataset_key], sessions=sessions)
    probe_dataset.subject_label_map = build_subject_label_map(probe_dataset)
    num_subjects = len(probe_dataset.subject_label_map)

    train_positions, val_positions = split_positions_stratified(
        probe_dataset, val_fraction=args.val_fraction, seed=args.split_seed)

    train_loader = DataLoader(Subset(probe_dataset, train_positions),
                               batch_size=args.probe_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(probe_dataset, val_positions),
                             batch_size=args.probe_batch_size, shuffle=False, num_workers=0)

    print("Extracting pre-projection embeddings (train split)...")
    train_embeds, train_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, train_loader, device, embed_dim=args.embed_dim)
    print("Extracting pre-projection embeddings (val split)...")
    val_embeds, val_labels = extract_embeddings(
        encoder, lsp_bank, dataset_key, val_loader, device, embed_dim=args.embed_dim)
    print(f"train_embeds: {tuple(train_embeds.shape)}, val_embeds: {tuple(val_embeds.shape)}")

    print("\nTraining subject-ID probe...")
    _probe, best_val_acc, final_val_acc = train_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        embed_dim=args.embed_dim, num_classes=num_subjects,
        device=device, epochs=args.epochs, lr=args.lr,
    )

    chance = 1.0 / num_subjects
    collapsed = final_val_acc < chance + COLLAPSE_MARGIN
    print(f"\n--- {dataset_key}: subject-ID probe (diagnostic) ---")
    print(f"Subjects:           {num_subjects}")
    print(f"Chance level:       {chance:.4f}")
    print(f"Best val accuracy:  {best_val_acc:.4f} (max over epochs -- optimistic, reference only)")
    print(f"Final val accuracy: {final_val_acc:.4f} (last epoch -- report this one)")
    print(f"Above chance by:    {final_val_acc - chance:+.4f}")
    if collapsed:
        print("WARNING: subject structure looks collapsed (near chance). SSRS "
              "needs subject structure to match against.")
    return {
        "dataset": dataset_key, "num_subjects": num_subjects, "chance": chance,
        "final_val_acc": final_val_acc, "best_val_acc": best_val_acc,
        "n_val_epochs": int(val_embeds.shape[0]), "collapsed_warning": bool(collapsed),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subject-ID diagnostic probe on frozen, pre-projection Phase 0 embeddings")
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
                         help="Fraction of EACH subject's epochs held out "
                              "for val (stratified, not subject-disjoint)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_sessions", choices=["train", "all"], default="train",
                         help="train (default): only the sessions Phase 0 was "
                              "allowed to see (checkpoint's train_sessions). "
                              "all: every session.")
    parser.add_argument("--json_out", type=str, default=None,
                         help="Write the results to this JSON file (for the "
                              "ablation 12 subject-vs-task plot).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using cache_dir: {args.cache_dir}")

    # -----------------------------------------------------------------
    # 1. Load frozen Phase 0 LSPBank + encoder (projection head NOT loaded)
    # -----------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    print("This probe is a DIAGNOSTIC, not a gate -- G3 is the task probe.")

    to_probe = [args.dataset] if args.dataset else dataset_keys
    for d in to_probe:
        if d not in dataset_keys:
            raise ValueError(f"Dataset '{d}' is not in the checkpoint ({dataset_keys}).")

    results = [probe_one_dataset(d, lsp_bank, encoder, args, device, sessions)
               for d in to_probe]

    # -----------------------------------------------------------------
    # Diagnostic summary (no verdict)
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}\nSUBJECT-ID DIAGNOSTIC SUMMARY (pre-projection, epoch-level stratified split)\n{'=' * 60}")
    for r in results:
        flag = "  <-- WARNING: collapsed" if r["collapsed_warning"] else ""
        print(f"  {r['dataset']:<10} acc {r['final_val_acc']:.4f} "
              f"(chance {r['chance']:.4f}, {r['num_subjects']} subjects){flag}")
    print("\nDiagnostic only: no pass/fail. A collapsed result is a warning for SSRS; "
          "a very high one is informational on its own. Read it against the G3 task "
          "probe (probe_task_decodability.py) -- plot both for ablation 12.")

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint, "space": "pre_projection_64d",
            "split": "epoch_level_stratified", "probe_sessions": args.probe_sessions,
            "train_sessions": train_sessions, "results": results,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()