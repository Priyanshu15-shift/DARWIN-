"""
lsp.py

LSP (per-dataset channel mixer / "Local Spatial Projection") for DARWIN
Phase 0, per spec v3 Section 4 and Section 7.

WHAT THIS SOLVES:
Your four datasets have different channel counts (3 / 22 / 62 / 64) and
different physical electrode layouts. LSP learns a per-dataset linear
projection W_dataset that maps each dataset's raw channels down to a
shared C_shared=16 channel space, so the EEGNet encoder downstream
always sees a fixed 16-channel input regardless of which dataset an
epoch came from.

SHAPE CONVENTION — flagging a spec inconsistency:
Spec v3 Section 3 states "W_dataset in R^(C_dataset x 16)", but Section
4's pseudocode is "W_dataset @ x -> (16, T)" where x is (C_dataset, T).
Those are only dimensionally consistent if W_dataset is (16, C_dataset),
not (C_dataset, 16). The initialization description in Section 7 ("each
row peaks at one physical channel") also only makes semantic sense under
the (16, C_dataset) orientation: each of the 16 OUTPUT rows is a
weighted combination over the C_dataset physical channels, peaking at
one anchor. This module uses W_dataset with shape (16, C_dataset).
Flag this to whoever owns spec v3 Section 3's notation — it's likely
just a documentation inconsistency, not an intentional different design.

INITIALIZATION (spec v3 Section 7):
"Soft channel selector: each row of W_dataset peaks at one physical
channel, decays with Gaussian kernel over spatial neighbors."

Concretely:
    1. Define 16 canonical scalp target positions (a standard 16-channel
       10-20 layout, spread across the scalp for broad coverage).
    2. For a given dataset's actual channel montage, find the physical
       channel nearest each of the 16 targets -> that's the "peak"
       channel for that output row.
    3. Each output row's weight over ALL C_dataset physical channels is
       a Gaussian function of 3D Euclidean distance from that row's peak
       channel: near the peak = high weight, far = decays toward 0.
    4. Weights are L2-normalized per row so initial output scale is
       comparable across datasets with very different channel counts.

This initialization is NOT frozen — LSP weights are explicitly listed as
trained in spec Section 4/7 ("Backpropagation scope: LSP matrices +
EEGNet encoder + projection head"). The Gaussian init just gives training
a sensible, spatially-coherent starting point instead of random noise.

V3.6 CHANGE -- RIGID CHANNEL-INTERSECTION MODE (new `rigid_intersection` flag):
Spec v3.6 requires that LSP can be reverted to a rigid channel intersection
with ONE flag flip, as the fallback if the learned projection fails across
datasets (risk table: "LSP fails across datasets -> flag flip to rigid channel
intersection -> single-montage results"). `LSPBank(..., rigid_intersection=True)`
does exactly that:
    - finds the electrode names common to EVERY dataset in the bank
      (case-insensitive, trailing dots/whitespace ignored);
    - builds a FIXED (16, C_dataset) selection matrix per dataset: one
      one-hot row per common channel, remaining rows all-zero, so the
      encoder still receives its fixed 16-channel input;
    - stores that matrix as a BUFFER, not a Parameter: nothing is trained,
      total_param_count() is 0, and the optimizer in train_phase0.py simply
      gets an empty LSP parameter list.
The state_dict key names are unchanged ("layers.<dataset>.W"), so existing
learned checkpoints still load into a default (rigid_intersection=False)
bank. Do NOT load a learned checkpoint's lsp_state_dict into a rigid bank:
it would overwrite the selection matrix. Rigid mode is a separate Phase 0
run with its own checkpoint. With bci_iv_2a + bci_iv_2b + physionet the
intersection is just C3, Cz, C4 (3 live channels, 13 zero channels), which
is exactly the "single-montage" degradation the risk table accepts.
Pass the result of `bank.config()` into the checkpoint so downstream code
can tell which mode produced it.

DEPENDENCY: requires `mne` for standard 10-20/10-10 electrode positions
(mne is very likely already installed as a MOABB dependency in your
venv). If missing: `uv pip install mne`.
"""

import math
import torch
import torch.nn as nn


# 16 canonical target positions, chosen for broad scalp coverage using
# standard 10-20 system electrode names. This defines the "meaning" of
# each of the 16 shared output channels — every dataset's LSP maps onto
# this SAME set of 16 target locations, which is what makes the shared
# embedding space (spec's "z is the only embedding space" invariant)
# comparable across datasets with completely different montages.
LSP_TARGET_CHANNELS_16 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "P4", "P8",
]
assert len(LSP_TARGET_CHANNELS_16) == 16


def _get_montage_positions(channel_names):
    """
    Looks up 3D positions for the given channel names using MNE's
    standard 10-20 montage (covers standard 10-20 AND extended 10-10
    names, which is what PhysioNet/OpenBMI-style 62/64-channel caps use).

    Returns: dict {channel_name: (x, y, z)} for every name that was
    found. Names not found in the standard montage are silently omitted
    — caller must check for missing names and decide how to handle them
    (this happens for non-standard or manufacturer-specific channel
    labels; worth checking the returned dict length against the input
    length rather than assuming full coverage).
    """
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_positions = montage.get_positions()["ch_pos"]  # {name: np.array([x,y,z])}

    positions = {}
    missing = []
    for name in channel_names:
        # MNE montage names are case-sensitive and dataset channel naming
        # conventions vary (e.g. "FC3" vs "Fc3" vs "fc3") — try a few
        # reasonable variants before giving up on a channel.
        candidates = [name, name.upper(), name.capitalize(),
                      name[0].upper() + name[1:].lower() if len(name) > 1 else name.upper()]
        found = None
        for cand in candidates:
            if cand in montage_positions:
                found = cand
                break
        if found is not None:
            positions[name] = tuple(montage_positions[found])
        else:
            missing.append(name)

    if missing:
        print(f"[lsp] WARNING: {len(missing)} channel name(s) not found in "
              f"standard_1020 montage and will be excluded from spatial "
              f"initialization: {missing}. Check for naming mismatches "
              f"(e.g. trailing whitespace, non-standard labels).")

    return positions


def _gaussian_init_weights(dataset_channel_names, n_shared=16,
                             target_channels=None, sigma_frac=0.3):
    """
    Builds the (n_shared, C_dataset) initialization matrix per spec v3
    Section 7's "soft channel selector" description.

    dataset_channel_names: ordered list of this dataset's channel names,
        length C_dataset. ORDER MATTERS — must match the channel order
        in your cached 'X' arrays exactly, or the wrong physical channel
        gets wired to the wrong LSP output row.
    n_shared: number of output shared channels (spec: 16)
    target_channels: the n_shared canonical target electrode names to
        anchor each output row to. Defaults to LSP_TARGET_CHANNELS_16.
    sigma_frac: Gaussian kernel bandwidth as a fraction of the maximum
        pairwise distance in this dataset's montage. Larger = smoother/
        wider mixing across channels, smaller = closer to a hard
        one-hot channel selector. 0.3 is a moderate starting point, not
        spec-specified — reasonable to tune later if LSP convergence
        (spec Section 10: "non-degenerate W_dataset") looks off.

    Returns: torch.Tensor of shape (n_shared, C_dataset)
    """
    if target_channels is None:
        target_channels = LSP_TARGET_CHANNELS_16
    assert len(target_channels) == n_shared

    dataset_positions = _get_montage_positions(dataset_channel_names)
    target_positions = _get_montage_positions(target_channels)

    missing_dataset = [c for c in dataset_channel_names if c not in dataset_positions]
    missing_targets = [c for c in target_channels if c not in target_positions]
    if missing_dataset:
        raise ValueError(
            f"Cannot compute spatial init: {len(missing_dataset)} of this "
            f"dataset's channels have no known position: {missing_dataset}. "
            f"Either fix the channel name (likely a naming-convention "
            f"mismatch with the standard_1020 montage) or fall back to "
            f"random initialization for this dataset's LSP layer."
        )
    if missing_targets:
        raise ValueError(
            f"Cannot compute spatial init: {len(missing_targets)} of the 16 "
            f"canonical target channels have no known position: "
            f"{missing_targets}. This shouldn't happen with standard 10-20 "
            f"names — check LSP_TARGET_CHANNELS_16 for typos."
        )

    C = len(dataset_channel_names)
    dataset_pos = torch.tensor(
        [dataset_positions[c] for c in dataset_channel_names], dtype=torch.float32
    )  # (C, 3)
    target_pos = torch.tensor(
        [target_positions[c] for c in target_channels], dtype=torch.float32
    )  # (n_shared, 3)

    # Pairwise distances: (n_shared, C)
    dists = torch.cdist(target_pos, dataset_pos, p=2)

    # For each target, find nearest physical channel — that's the "peak"
    # this output row's weight will be centered on.
    peak_idx = dists.argmin(dim=1)  # (n_shared,)
    peak_dist_to_all = dists[torch.arange(n_shared), :]  # (n_shared, C) — reuse dists directly

    # Gaussian kernel bandwidth: scale relative to this dataset's own
    # montage spread, so datasets with very different physical head
    # coverage (a 3-channel motor-strip-only cap vs a 64-channel
    # whole-scalp cap) get a sensibly-scaled decay rather than a fixed
    # absolute distance that means something different for each.
    max_pairwise = torch.cdist(dataset_pos, dataset_pos, p=2).max()
    sigma = max(sigma_frac * max_pairwise.item(), 1e-6)  # avoid div-by-zero for degenerate montages

    # Distance from each target's PEAK channel (not the canonical target
    # position itself) to every physical channel — this is what "peaks
    # at one physical channel, decays over spatial neighbors" means.
    peak_positions = dataset_pos[peak_idx]  # (n_shared, 3)
    dist_from_peak = torch.cdist(peak_positions, dataset_pos, p=2)  # (n_shared, C)

    weights = torch.exp(-(dist_from_peak ** 2) / (2 * sigma ** 2))  # (n_shared, C)

    # L2-normalize each row so initial output scale doesn't scale with
    # C_dataset (a 64-channel dataset's rows would otherwise sum to a
    # much larger initial magnitude than a 3-channel dataset's).
    row_norms = weights.norm(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights / row_norms

    return weights  # (n_shared, C_dataset)


def _normalize_name(name):
    """Case-insensitive channel key; ignores trailing dots/whitespace
    (some MOABB datasets label channels like 'C3.' or 'Fc5.')."""
    return name.strip().rstrip(".").upper()


def _common_channel_names(dataset_channel_names, target_channels=None, n_shared=16):
    """
    Normalized names present in EVERY dataset's montage, ordered so that
    channels in the canonical 16-target list come first (in that list's
    order), then any other common channels alphabetically. Truncated to
    n_shared if the intersection is larger than the shared width.
    """
    if target_channels is None:
        target_channels = LSP_TARGET_CHANNELS_16
    sets = [{_normalize_name(c) for c in names}
            for names in dataset_channel_names.values()]
    common = set.intersection(*sets) if sets else set()
    canon_order = [_normalize_name(c) for c in target_channels]
    ordered = [c for c in canon_order if c in common]
    ordered += sorted(common - set(ordered))
    return ordered[:n_shared]


def _rigid_selection_matrix(channel_names, common, n_shared=16):
    """
    Fixed (n_shared, C_dataset) one-hot selector: row i picks the dataset
    channel whose normalized name equals common[i]; rows past len(common)
    stay all-zero. Channel ORDER in channel_names must match the cached X.
    """
    index = {_normalize_name(c): i for i, c in enumerate(channel_names)}
    W = torch.zeros(n_shared, len(channel_names))
    for row, name in enumerate(common):
        W[row, index[name]] = 1.0
    return W


class LSPLayer(nn.Module):
    """
    One dataset's LSP: a trainable (16, C_dataset) linear channel mixer.
    forward(x) implements the spec's "W_dataset @ x -> (16, T)".

    x: (C_dataset, T) or (B, C_dataset, T)
    returns: (16, T) or (B, 16, T)

    rigid_common: None (default) = normal learned LSP. A list of normalized
    common channel names = RIGID mode (v3.6 fallback): W is a fixed one-hot
    selection buffer, not trained. See module docstring.
    """

    def __init__(self, channel_names, n_shared=16, sigma_frac=0.3,
                 use_spatial_init=True, rigid_common=None):
        super().__init__()
        self.channel_names = channel_names
        self.c_dataset = len(channel_names)
        self.n_shared = n_shared
        self.rigid = rigid_common is not None

        if self.rigid:
            # Buffer (not Parameter): keeps the "W" state_dict key but is
            # excluded from .parameters(), so the optimizer never sees it.
            self.register_buffer(
                "W", _rigid_selection_matrix(channel_names, rigid_common, n_shared))
            return

        if use_spatial_init:
            init_weights = _gaussian_init_weights(
                channel_names, n_shared=n_shared, sigma_frac=sigma_frac
            )
        else:
            # Fallback: small random init. Only use this if spatial
            # positions genuinely can't be resolved for some channel
            # naming scheme — spec explicitly wants the Gaussian init,
            # so this path should be the exception, not the default.
            init_weights = torch.randn(n_shared, self.c_dataset) * 0.01
            print(f"[lsp] WARNING: using random init, NOT the spec-required "
                  f"spatial Gaussian init, for a {self.c_dataset}-channel dataset.")

        self.W = nn.Parameter(init_weights)  # (n_shared, C_dataset), trained

    def forward(self, x):
        # x: (..., C_dataset, T) -> (..., n_shared, T)
        # torch.matmul broadcasts correctly whether x is (C,T) or (B,C,T)
        return torch.matmul(self.W, x)

    def param_count(self):
        # TRAINABLE parameters only: a rigid layer has none.
        return 0 if self.rigid else self.W.numel()


class LSPBank(nn.Module):
    """
    Holds one LSPLayer per dataset. Spec Section 4: "~1K params total,
    trained" across ALL datasets combined — with n_shared=16, a dataset
    contributes 16 * C_dataset params, so e.g. 16*22 (bci_iv_2a) +
    16*3 (bci_iv_2b) + 16*62 (openbmi) + 16*64 (physionet) = 2,416
    params total (16 * 151 channels; an earlier version of this docstring said
    2,432, which was an arithmetic slip). That's above the spec's "~1K" figure — flag this to
    whoever wrote spec Section 4, since it's a real discrepancy, not
    just rounding. Possible resolutions: spec's ~1K estimate assumed
    fewer/smaller datasets, or a shared low-rank factorization instead
    of a full dense matrix per dataset. Proceeding with full dense
    matrices here since that's what Section 4's exact pseudocode
    describes; revisit if the param budget turns out to matter.
    """

    def __init__(self, dataset_channel_names, n_shared=16, sigma_frac=0.3,
                 rigid_intersection=False):
        """
        dataset_channel_names: dict {dataset_name: [channel_name, ...]}
            ORDER of each list must match the channel order in that
            dataset's cached 'X' arrays exactly.
        rigid_intersection: v3.6 one-flag fallback. False (default) = the
            learned, spatially-initialised LSP. True = fixed channel
            intersection across all datasets in this bank, nothing trained.
        """
        super().__init__()
        self.n_shared = n_shared
        self.rigid_intersection = rigid_intersection

        if rigid_intersection:
            self.common_channels = _common_channel_names(
                dataset_channel_names, n_shared=n_shared)
            if not self.common_channels:
                raise ValueError(
                    "rigid_intersection=True but the datasets share no channel "
                    "names: " + str({k: len(v) for k, v in dataset_channel_names.items()})
                    + ". Check naming (case, trailing dots) or drop a dataset.")
            print(f"[lsp] RIGID intersection mode: {len(self.common_channels)} common "
                  f"channel(s) {self.common_channels}; remaining "
                  f"{n_shared - len(self.common_channels)} of {n_shared} shared "
                  f"channels are zero. No LSP parameters are trained.")
            self.layers = nn.ModuleDict({
                ds: LSPLayer(names, n_shared=n_shared, rigid_common=self.common_channels)
                for ds, names in dataset_channel_names.items()
            })
        else:
            self.common_channels = None
            self.layers = nn.ModuleDict({
                ds: LSPLayer(names, n_shared=n_shared, sigma_frac=sigma_frac)
                for ds, names in dataset_channel_names.items()
            })

    def config(self):
        """Small dict to store in the checkpoint so downstream code knows
        which LSP mode produced it."""
        return {"rigid_intersection": self.rigid_intersection,
                "common_channels": self.common_channels,
                "n_shared": self.n_shared}

    def forward(self, x, dataset_name):
        return self.layers[dataset_name](x)

    def total_param_count(self):
        return sum(layer.param_count() for layer in self.layers.values())


if __name__ == "__main__":
    # Channel names confirmed against real MOABB raw data (see
    # get_channel_names.py output). physionet and openbmi are still
    # placeholder-pending — OpenBMI/PhysioNet downloads were running
    # when this was last updated; re-run get_channel_names.py once those
    # finish and update DATASET_CHANNEL_NAMES below.
    #
    # ASSUMPTION FLAGGED: these lists assume preprocessing.py selects
    # EEG channels by simply taking the first N channels in MNE's
    # original raw order (dropping trailing EOG/STI channels), which is
    # MOABB's typical pick_types behavior. This matches the cached
    # channel COUNTS (22 for bci_iv_2a, 3 for bci_iv_2b) but the exact
    # channel-selection logic in preprocessing.py should be checked to
    # confirm the ORDER also matches — a count match doesn't guarantee
    # the same order was used.
    DATASET_CHANNEL_NAMES = {
        "bci_iv_2a": [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
            "P2", "POz",
        ],  # 22 channels, EOG1/EOG2/EOG3/STI dropped
        "bci_iv_2b": ["C3", "Cz", "C4"],  # EOG1/EOG2/EOG3/STI dropped
        # "physionet": [...],  # pending download
        # "openbmi": [...],    # pending download
    }

    print("Building LSP bank for datasets with confirmed channel names.\n")
    bank = LSPBank(DATASET_CHANNEL_NAMES, n_shared=16)
    print(f"Total LSP params (bci_iv_2a + bci_iv_2b): {bank.total_param_count()}")
    print(f"  (expect 16*22 + 16*3 = {16*22 + 16*3})")

    for ds_key, n_channels in [("bci_iv_2a", 22), ("bci_iv_2b", 3)]:
        x = torch.randn(n_channels, 1001)
        out = bank(x, ds_key)
        print(f"\n{ds_key}: input {tuple(x.shape)} -> output {tuple(out.shape)} "
              f"(expect (16, 1001))")

        x_batch = torch.randn(8, n_channels, 1001)
        out_batch = bank(x_batch, ds_key)
        print(f"{ds_key} batched: input {tuple(x_batch.shape)} -> "
              f"output {tuple(out_batch.shape)} (expect (8, 16, 1001))")
        

    # ------------------------------------------------------------------
    # v3.6: rigid channel-intersection mode checks
    # ------------------------------------------------------------------
    print("\n--- rigid_intersection=True ---")
    rigid = LSPBank(DATASET_CHANNEL_NAMES, n_shared=16, rigid_intersection=True)
    assert rigid.total_param_count() == 0, "rigid LSP must have no trainable params"
    assert len(list(rigid.parameters())) == 0, "rigid LSP must expose no Parameters"
    assert rigid.common_channels == ["C3", "CZ", "C4"], rigid.common_channels
    print(f"config: {rigid.config()}")

    # Output row 0/1/2 must equal the raw C3/Cz/C4 rows; rows 3..15 must be zero.
    names_2a = DATASET_CHANNEL_NAMES["bci_iv_2a"]
    x = torch.randn(4, len(names_2a), 500)
    y = rigid(x, "bci_iv_2a")
    assert y.shape == (4, 16, 500)
    for row, nm in enumerate(["C3", "Cz", "C4"]):
        assert torch.equal(y[:, row], x[:, names_2a.index(nm)]), nm
    assert torch.count_nonzero(y[:, 3:]) == 0
    print("rigid mode: selected rows match raw channels, other rows are zero. OK")

    # Default mode must be unchanged: same state_dict keys as before.
    keys = list(bank.state_dict().keys())
    assert keys == ["layers.bci_iv_2a.W", "layers.bci_iv_2b.W"], keys
    assert list(rigid.state_dict().keys()) == keys
    print("state_dict key names identical in both modes. OK")