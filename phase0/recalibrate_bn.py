"""
recalibrate_bn.py

Controlled experiment for the BatchNorm hypothesis: recompute the encoder's
BatchNorm RUNNING statistics on a balanced calibration set and save a new
checkpoint. NO weights are changed and nothing is retrained -- only the
`running_mean` / `running_var` buffers of the two BatchNorm2d layers (bn1, bn2)
in the encoder.

WHY
    train_phase0.py calls the encoder separately for every dataset group and for
    each view, so (a) during training each BatchNorm call normalizes with the
    statistics of ONE small single-dataset sub-batch, and (b) the running
    averages that eval() uses are an exponential average over roughly the last
    ~10 calls -- dominated by whichever dataset group was processed last, not by
    the pooled data distribution. The probes then run in eval() with those
    running statistics. If the running statistics are the problem, replacing
    them with properly balanced ones (same weights) should change the probe
    results. If it does not, the damage is in the trained weights, and the
    next step is to retrain with a normalization without running statistics.

WHAT IT DOES
    1. Loads a Phase 0 checkpoint (LSP bank + encoder; projection head ignored).
    2. Draws the SAME number of epochs from every dataset (default 2000 each,
       fixed seed), from the sessions Phase 0 was allowed to train on.
       Clean signal, no augmentation (what the probes feed the encoder).
    3. Layer by layer in forward order (bn1, then bn2), with everything in
       eval() mode (dropout off, earlier BN layers already finalized), measures
       the per-channel mean and variance of that layer's INPUT for each dataset.
       Sequential, so bn2's statistics are computed on inputs that went through
       the FINAL bn1 -- exactly what eval() will feed it.
    4. Pools them with EQUAL weight per dataset (mean of dataset means; variance
       = mean within-dataset variance + variance of the dataset means) and writes
       that into running_mean / running_var. This is exact, deterministic, and
       has no momentum or call-order effect.
    5. Prints a diagnostic: for each BN layer and dataset, how far that
       dataset's true input statistics were from the running statistics BEFORE
       and AFTER (standardized mean shift, and median variance ratio).
    6. Saves a new checkpoint (all other keys copied unchanged, plus a
       `bn_recalibration` record).

MODES
    --mode pooled       (default) one checkpoint, statistics pooled over all
                        datasets with equal weight.
    --mode per_dataset  one checkpoint PER dataset, each calibrated on that
                        dataset alone (domain-specific statistics, "AdaBN").
                        Probe each with --dataset <that dataset>. Diagnostic of
                        how much dataset-specific normalization matters; the
                        pooled checkpoint is the cleaner test of contamination.

Run it on the UNTRAINED checkpoint too (a fair control: a fresh encoder has
identity BN statistics), then compare the probes on all four checkpoints.

Usage:
    python recalibrate_bn.py --checkpoint phase0_interim_3ds.pt
    python recalibrate_bn.py --checkpoint phase0_untrained.pt
    python recalibrate_bn.py --checkpoint phase0_interim_3ds.pt --mode per_dataset
    (writes phase0_interim_3ds_bncal.pt, ..._bncal_<dataset>.pt)
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from lsp import LSPBank
from encoder import Phase0Encoder
from sampler import PairDataset
from cache_config import CACHE_DIR


# ---------------------------------------------------------------------------
# Pure arithmetic (works on numpy arrays and torch tensors alike)
# ---------------------------------------------------------------------------
def pool_moments(means, variances):
    """
    Pool per-dataset (mean, biased variance) with EQUAL weight per dataset.
    means, variances: lists of per-channel arrays, one per dataset.
    Returns (pooled_mean, pooled_var), where
        pooled_mean = mean_d(mu_d)
        pooled_var  = mean_d(var_d + (mu_d - pooled_mean)^2)
    i.e. the exact variance of the equal-weight mixture of the datasets.
    """
    d = len(means)
    pooled_mean = sum(means) / d
    pooled_var = sum(v + (m - pooled_mean) ** 2 for m, v in zip(means, variances)) / d
    return pooled_mean, pooled_var


# ---------------------------------------------------------------------------
# Measuring BatchNorm input statistics
# ---------------------------------------------------------------------------
@torch.no_grad()
def bn_input_moments(lsp_bank, encoder, bn, ds_name, x_all, device, batch_size=128):
    """Per-channel mean and (biased) variance of the INPUT to `bn` for one
    dataset, measured in eval() mode. x_all: (N, C_dataset, T) tensor."""
    lsp_bank.eval()
    encoder.eval()
    c = bn.num_features
    s = torch.zeros(c, dtype=torch.float64, device=device)
    ss = torch.zeros(c, dtype=torch.float64, device=device)
    count = [0]

    def hook(_module, inputs, _output):
        x = inputs[0].double()                      # (B, C, H, W)
        s.add_(x.sum(dim=(0, 2, 3)))
        ss.add_((x * x).sum(dim=(0, 2, 3)))
        count[0] += x.numel() // c

    handle = bn.register_forward_hook(hook)
    try:
        for i in range(0, x_all.shape[0], batch_size):
            xb = x_all[i:i + batch_size].to(device).float()
            encoder(lsp_bank(xb, ds_name))
    finally:
        handle.remove()
    mean = s / count[0]
    var = (ss / count[0] - mean ** 2).clamp(min=0.0)
    return mean, var


def batchnorm_layers(encoder):
    """BatchNorm2d layers in registration (= forward) order."""
    return [(n, m) for n, m in encoder.named_modules() if isinstance(m, nn.BatchNorm2d)]


def calibrate(lsp_bank, encoder, data_by_ds, ds_list, device, batch_size):
    """Overwrite running_mean / running_var of every BN layer, sequentially in
    forward order, from the datasets in `ds_list` (equal weight each)."""
    encoder.eval()
    for name, bn in batchnorm_layers(encoder):
        moms = [bn_input_moments(lsp_bank, encoder, bn, ds, data_by_ds[ds],
                                 device, batch_size) for ds in ds_list]
        mean, var = pool_moments([m for m, _ in moms], [v for _, v in moms])
        with torch.no_grad():
            bn.running_mean.copy_(mean.float())
            bn.running_var.copy_(var.float())


@torch.no_grad()
def diagnose(lsp_bank, encoder, data_by_ds, ds_list, device, batch_size):
    """For each BN layer and dataset: how far the dataset's true input
    statistics are from the layer's CURRENT running statistics.
        shift     = mean over channels of |mu_d - run_mean| / sqrt(run_var + eps)
        var_ratio = median over channels of var_d / run_var
    shift ~ 0 and var_ratio ~ 1 mean the running statistics fit that dataset."""
    out = {}
    for name, bn in batchnorm_layers(encoder):
        run_mean = bn.running_mean.double()
        run_var = bn.running_var.double()
        for ds in ds_list:
            mu, var = bn_input_moments(lsp_bank, encoder, bn, ds, data_by_ds[ds],
                                       device, batch_size)
            shift = ((mu - run_mean).abs() / (run_var + bn.eps).sqrt()).mean().item()
            ratio = (var / (run_var + 1e-12)).median().item()
            out[(name, ds)] = (shift, ratio)
    return out


def print_diagnostic(title, diag, ds_list):
    print(f"\n{title}")
    print(f"  {'layer':<22}{'dataset':<12}{'mean shift':>12}{'var ratio':>12}")
    for (name, ds), (shift, ratio) in diag.items():
        print(f"  {name:<22}{ds:<12}{shift:>12.3f}{ratio:>12.3f}")


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def draw_calibration_data(cache_dir, ds_list, sessions, n_per_dataset, seed):
    """Same number of clean epochs from every dataset, fixed seed. Returns
    ({ds: (N, C, T) float tensor}, n_used)."""
    sets = {ds: PairDataset(cache_dir, [ds], sessions=sessions) for ds in ds_list}
    n = min([n_per_dataset] + [len(s) for s in sets.values()])
    rng = np.random.RandomState(seed)
    data = {}
    for ds in ds_list:
        idx = np.sort(rng.choice(len(sets[ds]), size=n, replace=False))
        data[ds] = torch.stack([sets[ds][int(i)] for i in idx]).float()
        sets[ds]._cache.clear()
        print(f"[calib] {ds}: {n} epochs, shape {tuple(data[ds].shape)}")
    return data, n


def main():
    parser = argparse.ArgumentParser(
        description="Recompute BatchNorm running statistics on balanced data "
                    "(no retraining, weights unchanged)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: <checkpoint>_bncal.pt). In "
                             "per_dataset mode: <stem>_<dataset>.pt.")
    parser.add_argument("--mode", choices=["pooled", "per_dataset"], default="pooled")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--n_per_dataset", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib_sessions", choices=["train", "all"], default="train",
                        help="train (default): only the sessions Phase 0 was "
                             "allowed to see (checkpoint's train_sessions).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using cache_dir: {args.cache_dir}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "dataset_keys" not in ckpt:
        raise SystemExit("This script needs a pooled checkpoint (key 'dataset_keys').")
    dataset_keys = list(ckpt["dataset_keys"])
    chans = ckpt["channel_names_by_dataset"]

    lsp_bank = LSPBank({d: chans[d] for d in dataset_keys}, n_shared=16).to(device)
    lsp_bank.load_state_dict(ckpt["lsp_state_dict"])
    encoder = Phase0Encoder(n_channels=16, sample_rate=250).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    lsp_bank.eval()
    encoder.eval()
    for module in (lsp_bank, encoder):
        for p in module.parameters():
            p.requires_grad = False

    layers = batchnorm_layers(encoder)
    print(f"BatchNorm layers (forward order): {[n for n, _ in layers]}")
    train_sessions = ckpt.get("train_sessions")
    sessions = train_sessions if args.calib_sessions == "train" else None
    print(f"Checkpoint datasets: {dataset_keys}; final Phase 0 loss: "
          f"{ckpt.get('final_loss', 'n/a')}")
    print(f"Calibration sessions: {sessions if sessions else 'all'}")

    data, n_used = draw_calibration_data(
        args.cache_dir, dataset_keys, sessions, args.n_per_dataset, args.seed)

    # Weights must not change -- remember them to verify afterwards.
    weights_before = {k: v.clone() for k, v in encoder.state_dict().items()
                      if "running_" not in k and "num_batches" not in k}

    print_diagnostic("BEFORE (current running statistics vs each dataset's true "
                     "BN-input statistics):",
                     diagnose(lsp_bank, encoder, data, dataset_keys, device,
                              args.batch_size), dataset_keys)

    base = (os.path.splitext(args.out)[0] if args.out
            else os.path.splitext(args.checkpoint)[0] + "_bncal")
    if args.mode == "pooled":
        jobs = [(None, dataset_keys, base + ".pt")]
    else:
        jobs = [(d, [d], f"{base}_{d}.pt") for d in dataset_keys]

    for tag, ds_list, out_path in jobs:
        enc = copy.deepcopy(encoder)
        calibrate(lsp_bank, enc, data, ds_list, device, args.batch_size)

        for k, v in enc.state_dict().items():
            if k in weights_before:
                assert torch.equal(v, weights_before[k]), f"weight {k} changed!"

        label = f"AFTER ({'pooled over ' + ', '.join(ds_list) if tag is None else 'calibrated on ' + tag}):"
        print_diagnostic(label, diagnose(lsp_bank, enc, data, dataset_keys, device,
                                         args.batch_size), dataset_keys)

        new = dict(ckpt)
        new["encoder_state_dict"] = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
        new["bn_recalibration"] = {
            "source_checkpoint": args.checkpoint, "mode": args.mode,
            "calibrated_on": ds_list, "n_per_dataset": n_used, "seed": args.seed,
            "calib_sessions": args.calib_sessions, "weighting": "equal_per_dataset",
        }
        torch.save(new, out_path)
        print(f"\nWrote {out_path}")
        if tag is None:
            print("Probe it with the same commands as the original checkpoint, e.g.\n"
                  f"  python phase0\\probe_task_decodability.py --checkpoint {out_path} "
                  f"--json_out task_{os.path.basename(out_path)}.json")
        else:
            print(f"Probe it with --dataset {tag}.")


if __name__ == "__main__":
    main()