"""
probe_dataset_leakage.py

Dataset-ID probe: a DIAGNOSTIC, not a gate (DARWIN spec rev A, section 9).
(The original script was missing from the codebase zip; this is a rewrite from
the spec, not a restore -- see "DESIGN CHOICES" below.)

WHAT IT MEASURES
    How well a linear probe can tell WHICH DATASET (BCI IV-2a, 2b, PhysioNet,
    OpenBMI) an epoch came from, using the frozen Phase 0 embeddings. Each
    dataset has its own LSP matrix and montage, so the encoder can pick up a
    dataset fingerprint (montage, recording hardware, epoch statistics) instead
    of task structure. A high value means the shared 64-d space still carries
    dataset identity, which can distort donor matching across datasets (SSRS)
    and the pooled population statistics.

    Run on pre-projection embeddings: the same 64-d encoder output the task and
    subject probes read, and the state space used by SSRS, DTKF and FiLM. The
    projection head is NEVER applied. LSP and encoder are frozen and in eval
    mode (BatchNorm on source running statistics), no augmentation.

HOW TO READ IT (diagnostic, no pass/fail)
    No threshold, no verdict. Read it next to the G3 task probe
    (probe_task_decodability.py) and the subject-ID diagnostic
    (probe_subject_leakage.py). Per-dataset recall and the confusion matrix show
    WHICH datasets are separable. If it is near 1.0 and the task probe is weak,
    suspect a dataset shortcut and consider balancing datasets in the Phase 0
    sampler.

DESIGN CHOICES (rewritten from the spec -- change them if your original differed)
    - Split: SUBJECT-DISJOINT within every dataset. Holding out whole subjects
      stops the probe from using subject identity (a subject belongs to exactly
      one dataset), so what remains is genuine dataset-level structure.
    - Balanced classes: PhysioNet has far more subjects than 2a/2b, so train and
      val are each subsampled to the same number of epochs per dataset. Chance
      is then 1 / (number of datasets).
    - Sessions: by default only the sessions Phase 0 was allowed to train on
      (the checkpoint's `train_sessions`). --probe_sessions all uses every
      session.
    - Needs a pooled checkpoint with at least two datasets.

Usage:
    python probe_dataset_leakage.py --checkpoint phase0_checkpoint_<...>.pt
    python probe_dataset_leakage.py --checkpoint <ckpt> --json_out dataset_probe.json
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
from cache_config import CACHE_DIR

# Shared probe machinery -- identical linear probe / train loop / extraction as
# the task and subject probes, by construction.
from probe_task_decodability import train_probe, extract_embeddings


class DatasetLabelDataset(PairDataset):
    """One dataset's epochs, each returned as (epoch, constant dataset int)."""

    def __init__(self, cache_dir, dataset_key, label, sessions=None, dtype=torch.float32):
        self.label = label
        super().__init__(cache_dir, [dataset_key], sessions=sessions, dtype=dtype)

    def __getitem__(self, idx):
        dataset_name, subject_id, epoch_idx = self.index[idx]
        arr = self._load_subject(dataset_name, subject_id)
        return (torch.tensor(arr[epoch_idx], dtype=self.dtype),
                torch.tensor(self.label, dtype=torch.long))


def split_subjects(dataset_obj, dataset_key, val_fraction, rng):
    """Subject-disjoint train/val positions for one dataset."""
    groups = dataset_obj.subject_groups()          # {(ds, subj): [positions]}
    keys = sorted(groups.keys())
    if len(keys) < 2:
        raise ValueError(f"'{dataset_key}' has {len(keys)} subject(s) after "
                         f"session filtering; need at least 2 for a "
                         f"subject-disjoint split.")
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    n_val = min(max(1, int(round(len(shuffled) * val_fraction))), len(shuffled) - 1)
    val_keys, train_keys = shuffled[:n_val], shuffled[n_val:]
    val_pos = [p for k in val_keys for p in groups[k]]
    train_pos = [p for k in train_keys for p in groups[k]]
    print(f"[probe] {dataset_key}: {len(train_keys)} train subjects, "
          f"{len(val_keys)} val subjects, {len(train_pos)}/{len(val_pos)} "
          f"train/val epochs available")
    return train_pos, val_pos


def subsample(positions, n, rng):
    positions = list(positions)
    if len(positions) <= n:
        return positions
    idx = rng.choice(len(positions), size=n, replace=False)
    return [positions[i] for i in sorted(idx)]


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-ID diagnostic probe on frozen, pre-projection Phase 0 embeddings")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Pooled Phase 0 checkpoint (.pt) with >= 2 datasets")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Fraction of each dataset's SUBJECTS held out for val")
    parser.add_argument("--max_epochs_per_dataset", type=int, default=4000,
                         help="Cap on epochs per dataset per split (after "
                              "balancing); keeps extraction fast.")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_sessions", choices=["train", "all"], default="train")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using cache_dir: {args.cache_dir}")

    # 1. Load frozen LSPBank + encoder (projection head NOT loaded)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "dataset_keys" not in ckpt:
        raise SystemExit("This probe needs a pooled checkpoint (key 'dataset_keys') "
                         "with at least two datasets.")
    dataset_keys = list(ckpt["dataset_keys"])
    if len(dataset_keys) < 2:
        raise SystemExit(f"Checkpoint has only {dataset_keys}; need >= 2 datasets.")
    channel_names_by_dataset = ckpt["channel_names_by_dataset"]

    lsp_bank = LSPBank({d: channel_names_by_dataset[d] for d in dataset_keys},
                       n_shared=16).to(device)
    lsp_bank.load_state_dict(ckpt["lsp_state_dict"])
    encoder = Phase0Encoder(n_channels=16, sample_rate=250).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for module in (lsp_bank, encoder):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    train_sessions = ckpt.get("train_sessions")
    sessions = train_sessions if args.probe_sessions == "train" else None
    print(f"Loaded checkpoint with datasets {dataset_keys}, "
          f"final Phase 0 loss: {ckpt.get('final_loss', 'n/a')}")
    print(f"Probe sessions: {sessions if sessions else 'all'}")
    print("Probing PRE-PROJECTION 64-d encoder embeddings (projection head not used).")
    print("This probe is a DIAGNOSTIC, not a gate.")

    # 2. Subject-disjoint split per dataset, then balance classes
    rng = np.random.RandomState(args.split_seed)
    objs, train_pos, val_pos = {}, {}, {}
    for label, ds in enumerate(dataset_keys):
        objs[ds] = DatasetLabelDataset(args.cache_dir, ds, label, sessions=sessions)
        train_pos[ds], val_pos[ds] = split_subjects(objs[ds], ds, args.val_fraction, rng)

    n_train = min(args.max_epochs_per_dataset, min(len(v) for v in train_pos.values()))
    n_val = min(args.max_epochs_per_dataset, min(len(v) for v in val_pos.values()))
    print(f"[probe] balancing: {n_train} train and {n_val} val epochs per dataset "
          f"({len(dataset_keys)} datasets)")

    # 3. Extract pre-projection embeddings per dataset (each uses its own LSP)
    def embed_split(positions_by_ds):
        embeds, labels = [], []
        for ds in dataset_keys:
            pos = subsample(positions_by_ds[ds],
                            n_train if positions_by_ds is train_pos else n_val, rng)
            loader = DataLoader(Subset(objs[ds], pos), batch_size=args.probe_batch_size,
                                shuffle=False, num_workers=0)
            e, y = extract_embeddings(encoder, lsp_bank, ds, loader, device,
                                      embed_dim=args.embed_dim)
            embeds.append(e)
            labels.append(y)
            objs[ds]._cache.clear()
        return torch.cat(embeds), torch.cat(labels)

    print("Extracting pre-projection embeddings (train split)...")
    train_embeds, train_labels = embed_split(train_pos)
    print("Extracting pre-projection embeddings (val split)...")
    val_embeds, val_labels = embed_split(val_pos)
    print(f"train_embeds: {tuple(train_embeds.shape)}, val_embeds: {tuple(val_embeds.shape)}")

    # 4. Train the probe
    print("\nTraining dataset-ID probe...")
    n_ds = len(dataset_keys)
    probe, best_val_acc, final_val_acc = train_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        embed_dim=args.embed_dim, num_classes=n_ds,
        device=device, epochs=args.epochs, lr=args.lr,
    )

    with torch.no_grad():
        probe.eval()
        preds = probe(val_embeds.to(device)).argmax(dim=1).cpu().numpy()
    truth = val_labels.numpy()
    conf = np.zeros((n_ds, n_ds), dtype=int)
    for t, p in zip(truth, preds):
        conf[t, p] += 1
    recall = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)

    chance = 1.0 / n_ds
    print(f"\n{'=' * 60}\nDATASET-ID DIAGNOSTIC (pre-projection, subject-disjoint, balanced)\n{'=' * 60}")
    print(f"Datasets:           {dataset_keys}")
    print(f"Chance level:       {chance:.4f}")
    print(f"Best val accuracy:  {best_val_acc:.4f} (max over epochs -- optimistic, reference only)")
    print(f"Final val accuracy: {final_val_acc:.4f} (last epoch -- report this one)")
    print(f"Above chance by:    {final_val_acc - chance:+.4f}")
    print("Per-dataset recall:")
    for ds, r in zip(dataset_keys, recall):
        print(f"  {ds:<10} {r:.4f}")
    print("Confusion matrix (rows = true dataset, cols = predicted):")
    print("            " + " ".join(f"{d[:9]:>9}" for d in dataset_keys))
    for ds, row in zip(dataset_keys, conf):
        print(f"  {ds:<10}" + " ".join(f"{v:>9d}" for v in row))
    print("\nDiagnostic only: no pass/fail. A high value means the shared space still "
          "carries dataset identity. Read it against the G3 task probe and the subject-ID "
          "diagnostic.")

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint, "space": "pre_projection_64d",
            "split": "subject_disjoint_balanced", "probe_sessions": args.probe_sessions,
            "train_sessions": train_sessions, "datasets": dataset_keys,
            "chance": chance, "final_val_acc": final_val_acc, "best_val_acc": best_val_acc,
            "per_dataset_recall": {d: float(r) for d, r in zip(dataset_keys, recall)},
            "confusion": conf.tolist(),
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()