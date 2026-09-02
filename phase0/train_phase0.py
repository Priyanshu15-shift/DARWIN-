"""
train_phase0.py

Phase 0 training loop for DARWIN, per spec v3 Section 7:

    Optimizer:        AdamW
    Learning rate:    1e-3 (frozen after Phase 0, no scheduling required)
    Batch size:       >=64 clips total, subject-balanced composition
    Negative sampling: every batch >=2 clips/subject (handled by sampler.py)
    Epochs:           train until InfoNCE loss plateaus, no fixed count

    Backprop scope (Section 7): LSP matrices + EEGNet encoder +
    projection head. Nothing else. -- this is naturally satisfied here
    since those are the only three nn.Module pieces with parameters in
    this pipeline; nothing extra needs to be frozen manually.

WIRING (matches the block diagram in spec Section 4):
    raw epoch (C_dataset, T)
      -> augment() TWICE, independently -> view1, view2   [augmentations.py]
      -> LSP: W_dataset @ x -> (16, T)                     [lsp.py]
      -> EEGNet 9-band encoder -> (64,)                    [encoder.py]
      -> projection head -> z (64,)                        [projection_head.py]
      -> InfoNCE(z1, z2, subject_ids)                       [losses.py]

IMPORTANT ordering detail: augmentation happens BEFORE LSP, on the raw
per-dataset-channel epoch -- not after. This matters because
channel_dropout (one of the 7 augmentations) is specifically meant to
"pressure LSP" per spec Section 6's own annotation ("Electrode failure
/ missing channels (pressures LSP)") -- i.e. LSP needs to see epochs
with randomly dropped-out physical channels DURING training so it
learns to be robust to that, which only works if augmentation runs on
the raw per-dataset channels LSP actually consumes, not on LSP's
already-projected 16-channel output.

MULTI-DATASET POOLING (updated -- previously a documented gap, now built):
An earlier version of this script only supported training on ONE
dataset at a time. Spec Section 3 says "Input: Raw EEG epochs from all
datasets," and an isolated single-dataset retrain (bci_iv_2a only, with
the losses.py fix applied) confirmed the fix alone wasn't sufficient --
subject-ID leakage dropped only modestly (97.4% -> 90.8%) and task
decodability actually got slightly worse (40.5% -> 34.3%), most likely
because 9 subjects from one dataset is too little diversity for the
encoder to find a better solution than partial subject-shortcut
learning. That result motivated building real multi-dataset pooling.

sampler.py's SubjectBalancedBatchSampler already supported drawing P
subjects from the union across multiple datasets -- no changes needed
there. What was actually missing: routing each dataset's clips through
ITS OWN LSP matrix before a shared encoder within one pooled batch.
See group_positions_by_dataset's docstring for the mechanics -- raw
epochs from different datasets can't be stacked into one tensor (they
have different channel counts AND different epoch lengths), so each
batch is grouped by dataset, processed separately through
augmentation -> that dataset's LSP -> the shared encoder, then
reassembled into one (B, 64) embedding tensor before the loss.
"""

import argparse
import os
import sys
import time

import torch
from torch.optim import AdamW

sys.path.append(os.path.dirname(__file__))
from sampler import make_phase0_dataloader
from augmentations import augment_batch, AugmentationConfig
from lsp import LSPBank
from encoder import Phase0Encoder
from projection_head import ProjectionHead
from losses import InfoNCELoss


def build_subject_id_tensor(dataset, batch_positions, device):
    """
    sampler.py's batch positions index into dataset.index, which holds
    (dataset_name, subject_id, epoch_idx) tuples. subject_id is a
    string (e.g. "1", "23") -- InfoNCE needs integer ids for the
    equality-comparison mask in losses.py. This builds a per-batch
    integer id (arbitrary numbering is fine, only equality/inequality
    within the batch matters, not the actual values).

    Already correctly multi-dataset-safe as written: keys on
    (dataset_name, subject_id), not just subject_id, so e.g. "subject 1"
    in bci_iv_2a and "subject 1" in physionet get distinct integer ids.
    No change needed here for pooling.
    """
    raw_subject_keys = [dataset.index[p][:2] for p in batch_positions]  # (ds, subj) per slot
    unique_keys = sorted(set(raw_subject_keys))
    key_to_int = {k: i for i, k in enumerate(unique_keys)}
    ids = [key_to_int[k] for k in raw_subject_keys]
    return torch.tensor(ids, dtype=torch.long, device=device)


def group_positions_by_dataset(dataset, batch_positions):
    """
    Splits one P x K batch's positions by which dataset each belongs to.

    NECESSARY for pooling: different datasets have different raw channel
    counts AND different epoch lengths (e.g. bci_iv_2a is (22, 1001),
    bci_iv_2b is (3, 1126), physionet is (64, 752)) -- there is no way to
    torch.stack a batch that mixes datasets into one raw tensor, the
    shapes don't even match. So each dataset's positions in the batch
    get pulled out, processed separately (through augmentation, THAT
    dataset's own LSP matrix, and the encoder -- exactly like the
    existing single-dataset code path, just called once per dataset
    present in this batch), and only reassembled into one (B, 64)
    embedding tensor AFTER the encoder, where every dataset's output is
    already the same shape regardless of its raw input shape.

    Returns: {dataset_name: (orig_batch_indices, positions)} -- orig_batch_indices
    are needed to put each dataset's resulting embeddings back in the
    batch's original order afterward.
    """
    groups = {}
    for i, p in enumerate(batch_positions):
        ds_name = dataset.index[p][0]
        if ds_name not in groups:
            groups[ds_name] = ([], [])
        groups[ds_name][0].append(i)
        groups[ds_name][1].append(p)
    return groups


def train_phase0(
    dataset_keys,
    channel_names_by_dataset,
    cache_dir="./cache",
    batch_size=64,
    k_per_subject=2,
    lr=1e-3,
    max_epochs=100,
    plateau_patience=10,
    plateau_min_delta=1e-4,
    checkpoint_path="phase0_checkpoint_pooled.pt",
    device=None,
    seed=0,
):
    """
    Trains LSP + encoder + projection head across MULTIPLE datasets
    pooled into one InfoNCE objective, per spec v3 Sections 4/5/7 --
    supersedes the earlier single-dataset-only version (see git history
    / earlier checkpoint comments for that version).

    dataset_keys: list, e.g. ["bci_iv_2a", "bci_iv_2b", "physionet"].
        sampler.py's SubjectBalancedBatchSampler already draws P
        subjects from the union across all given datasets -- no changes
        needed there. What DOES change here: each batch's positions get
        grouped by dataset (group_positions_by_dataset) and routed
        through that dataset's own LSP matrix before a shared encoder,
        since raw epochs from different datasets can't be stacked into
        one tensor (different channel counts AND different epoch
        lengths -- see group_positions_by_dataset's docstring).
    channel_names_by_dataset: dict {dataset_key: [channel names]}, one
        entry per dataset_keys, needed by LSPBank's spatial init for
        EACH dataset.
    plateau_patience / plateau_min_delta: see original docstring --
        unchanged semantics, just now measuring the pooled loss instead
        of a single dataset's loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")
    print(f"[train] pooling datasets: {dataset_keys}")

    torch.manual_seed(seed)

    # --- Data ---
    # make_phase0_dataloader already supports a multi-dataset `datasets`
    # list -- PairDataset indexes all of them together, and
    # SubjectBalancedBatchSampler draws P subjects from the FULL union
    # of eligible (dataset, subject) keys across all of them. No changes
    # needed in sampler.py itself for pooling.
    loader = make_phase0_dataloader(
        cache_dir=cache_dir,
        datasets=dataset_keys,
        batch_size=batch_size,
        k_per_subject=k_per_subject,
        num_workers=0,  # keep 0 on Windows to avoid multiprocessing spawn issues;
                        # raise once this is confirmed working end-to-end
        seed=seed,
    )
    dataset = loader.dataset

    # --- Model components ---
    # LSPBank already accepts a dict with MULTIPLE dataset entries at
    # construction (it's designed as a "bank" of per-dataset matrices) --
    # passing the full dict here is the only change needed on this line
    # versus the single-dataset version.
    lsp_bank = LSPBank(channel_names_by_dataset, n_shared=16).to(device)
    encoder = Phase0Encoder(n_channels=16, sample_rate=250).to(device)
    proj_head = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=64).to(device)
    loss_fn = InfoNCELoss(temperature=0.1)

    aug_config = AugmentationConfig()  # spec Section 6 defaults

    # Backprop scope per spec Section 7: LSP + encoder + projection head.
    params = (
        list(lsp_bank.parameters())
        + list(encoder.parameters())
        + list(proj_head.parameters())
    )
    optimizer = AdamW(params, lr=lr)

    print(f"[train] LSP params: {lsp_bank.total_param_count()}")
    print(f"[train] Encoder params: {encoder.param_count()}")
    print(f"[train] Projection head params: {proj_head.param_count()}")
    print(f"[train] Batches per epoch: {len(loader.batch_sampler)}")

    best_loss = float("inf")
    epochs_without_improvement = 0
    loss_history = []

    for epoch in range(1, max_epochs + 1):
        lsp_bank.train()
        encoder.train()
        proj_head.train()

        epoch_losses = []
        t0 = time.time()

        for batch_positions in loader.batch_sampler:
            B = len(batch_positions)
            subject_ids = build_subject_id_tensor(dataset, batch_positions, device)

            # Group this batch's positions by dataset -- see
            # group_positions_by_dataset's docstring for why this is
            # necessary rather than optional for pooling.
            groups = group_positions_by_dataset(dataset, batch_positions)

            # Per-dataset forward pass: augmentation -> THAT dataset's
            # LSP matrix -> shared encoder. Each call here is IDENTICAL
            # in shape/behavior to what the single-dataset version
            # already did successfully -- we're just calling it once per
            # dataset present in the batch instead of once for the whole
            # batch, then reassembling afterward.
            e1 = torch.zeros(B, 64, device=device)
            e2 = torch.zeros(B, 64, device=device)

            for ds_name, (orig_indices, positions) in groups.items():
                raw_sub = torch.stack(
                    [dataset[p] for p in positions], dim=0
                ).to(device)  # (sub_B, C_dataset, T_dataset) -- native shape for THIS dataset

                # Two INDEPENDENT augmented views, per spec Section 6's
                # "critical rule": augment() called twice, independent
                # random seeds, never share augmentation parameters.
                view1 = augment_batch(raw_sub.cpu(), aug_config).to(device)
                view2 = augment_batch(raw_sub.cpu(), aug_config).to(device)

                # Augmentation happens BEFORE LSP (see module docstring).
                h1 = lsp_bank(view1, ds_name)  # (sub_B, 16, T_dataset)
                h2 = lsp_bank(view2, ds_name)

                sub_e1 = encoder(h1)  # (sub_B, 64) -- fixed-size regardless of T_dataset
                sub_e2 = encoder(h2)

                # Scatter this dataset's embeddings back into their
                # original batch positions. This is a standard
                # autograd-safe pattern (index_put under the hood) --
                # gradients flow correctly back through sub_e1/sub_e2
                # into this dataset's LSP matrix and the shared encoder.
                idx_tensor = torch.tensor(orig_indices, device=device)
                e1[idx_tensor] = sub_e1
                e2[idx_tensor] = sub_e2

            z1 = proj_head(e1)  # (B, 64)
            z2 = proj_head(e2)

            loss = loss_fn(z1, z2, subject_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(mean_epoch_loss)
        elapsed = time.time() - t0
        print(f"[train] epoch {epoch:3d} | loss {mean_epoch_loss:.4f} | "
              f"{elapsed:.1f}s | {len(epoch_losses)} batches")

        if mean_epoch_loss < best_loss - plateau_min_delta:
            best_loss = mean_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= plateau_patience:
            print(f"[train] Loss plateaued ({plateau_patience} epochs without "
                  f"improvement >{plateau_min_delta}). Stopping per spec Section 7's "
                  f"'train until InfoNCE loss plateaus' criterion.")
            break
    else:
        print(f"[train] WARNING: hit max_epochs={max_epochs} without a detected "
              f"plateau. This is the hard ceiling, not the intended stopping "
              f"condition -- worth checking whether the loss is still trending "
              f"down (might just need more epochs) or stuck (might indicate a "
              f"training issue) before raising max_epochs and re-running blindly.")

    # --- Checkpoint ---
    # Now stores ALL pooled datasets' info, not just one -- probe
    # scripts need updating to match this format before they can load a
    # pooled checkpoint (they currently expect a single "dataset_key"
    # string and single "channel_names" list).
    checkpoint = {
        "lsp_state_dict": lsp_bank.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "projection_head_state_dict": proj_head.state_dict(),
        "dataset_keys": dataset_keys,
        "channel_names_by_dataset": channel_names_by_dataset,
        "loss_history": loss_history,
        "final_loss": mean_epoch_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[train] Checkpoint saved to {checkpoint_path}")

    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARWIN Phase 0 training (multi-dataset pooled)")
    parser.add_argument("--datasets", type=str, nargs="+",
                         default=["bci_iv_2a", "bci_iv_2b", "physionet"],
                         choices=["bci_iv_2a", "bci_iv_2b", "physionet"],
                         help="One or more dataset keys to pool. openbmi "
                              "excluded -- unresolved download issue. "
                              "e.g. --datasets bci_iv_2a bci_iv_2b physionet")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--batch_size", type=int, default=64,
                         help="Spec wants >=64. With bci_iv_2a (9) + "
                              "bci_iv_2b (9) + physionet (~109) pooled, "
                              "~127 subjects are available -- more than "
                              "enough for a true P=32,K=2 batch_size=64 "
                              "run (was capped at 16 for single-dataset "
                              "runs on 2a/2b alone).")
    parser.add_argument("--max_epochs", type=int, default=150)
    args = parser.parse_args()

    # Real channel names, confirmed against MOABB raw data via
    # get_channel_names.py (see lsp.py for the same lists).
    CHANNEL_NAMES = {
        "bci_iv_2a": [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
            "P2", "POz",
        ],
        "bci_iv_2b": ["C3", "Cz", "C4"],
        # Confirmed via get_channel_names.py against real cached PhysioNet
        # data. Raw MOABB output reported 65 channels total, but the last
        # one ('STIM') is a stimulus/trigger channel, not real EEG -- it
        # has no scalp position and can't go through the standard_1020
        # montage lookup, so it's excluded here. 64 real EEG channels
        # remain, matching the (174, 64, 752) shape confirmed cached for
        # this dataset.
        "physionet": [
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            "Fp1", "Fpz", "Fp2",
            "AF7", "AF3", "AFz", "AF4", "AF8",
            "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FT8",
            "T7", "T8", "T9", "T10",
            "TP7", "TP8",
            "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8",
            "O1", "Oz", "O2",
            "Iz",
        ],
    }

    channel_names_by_dataset = {ds: CHANNEL_NAMES[ds] for ds in args.datasets}
    checkpoint_name = "phase0_checkpoint_" + "_".join(sorted(args.datasets)) + ".pt"

    train_phase0(
        dataset_keys=args.datasets,
        channel_names_by_dataset=channel_names_by_dataset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        checkpoint_path=checkpoint_name,
    )