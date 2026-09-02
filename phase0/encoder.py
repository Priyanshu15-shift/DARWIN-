"""
encoder.py

EEGNet 9-band spectral encoder for DARWIN Phase 0, per spec v3 Section 4:

    Raw EEG (C_dataset, T)
      -> LSP: W_dataset @ x -> (16, T)               [~1K params]
      -> EEGNet Encoder: 9-band spectral -> R^d       [~18K params]
      -> Projection Head: 2-layer MLP -> R^64          [projection_head.py]
      -> z in R^64 (unified embedding space)

This file implements the SECOND block only (encoder). LSP (lsp.py)
must run first; this module consumes LSP's fixed 16-channel output,
never raw per-dataset channels directly — this is what makes the
encoder dataset-agnostic once LSP has done its job.

SPEC GAP BEING FILLED IN: v3 Section 4 specifies the block diagram and
an exact parameter budget (~18K) but does NOT specify literal layer
hyperparameters (filter counts, kernel sizes) for the EEGNet backbone.
The architecture below is a design choice made to satisfy the stated
constraints (9-band spectral input, ~18K trainable params, output
feeding a 64-dim projection head) — not something dictated verbatim by
the spec. If your team has a more specific EEGNet variant in mind
(e.g. from a cited paper), swap the backbone accordingly; the band
decomposition front-end and the ~18K budget are the two hard
constraints this must satisfy either way.

DESIGN: "9-band spectral" is implemented as an FBCNet-style fixed
filter bank front-end (Mane et al., FBCNet) rather than a single
learned temporal conv (as in vanilla EEGNet). The 9 bands are computed
via FFT-based band-pass filtering using the SAME band edges as
augmentations.py's freq_mask (EEG_9_BANDS_HZ, imported directly to
avoid the two ever drifting out of sync — freq_mask perturbing bands
the encoder isn't structured around would defeat the point of that
augmentation). This decomposition has ZERO trainable parameters,
consistent with spec Section 7's backprop scope ("LSP matrices +
EEGNet encoder + projection head. Nothing else.") — the band-pass
filtering itself is a fixed, deterministic transform, not a learned
layer.

After decomposition, the encoder is a compact EEGNet-style stack:
    1. Depthwise spatial conv (per-band, collapses the 16 LSP channels
       to a single spatial dimension per band) -- analogous to
       standard EEGNet's spatial filter, but grouped per band since
       each band arrives as a separate "channel" from the front-end.
    2. Depthwise temporal conv + pointwise conv (separable conv,
       standard EEGNet block 2) -- mixes information across bands and
       extracts temporal features.
    3. Global average pool + linear projection to a fixed-size
       embedding, which the projection head (projection_head.py) then
       maps to the spec's non-negotiable 64-dim z.

Default hyperparameters (D=13 depth multiplier, F2=52 separable-conv
output filters, kernel_len2=48) were chosen by grid search in this
build to land at 17,302 trainable params -- within ~4% of the spec's
~18K target. Tune D/F2/kernel_len2 if you want to hit it more exactly,
but this is close enough to treat as spec-compliant.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from augmentations import EEG_9_BANDS_HZ  # single source of truth for band edges,
                                            # shared with freq_mask so both operate
                                            # on the SAME frequency partition


def spectral_decompose(x, bands=None, sample_rate=250):
    """
    Fixed (non-trainable) 9-band filter bank front-end. Takes LSP's
    16-channel output and returns 9 separately band-pass-filtered
    versions, stacked as a new leading dimension -- the "9-band
    spectral" input the EEGNet encoder consumes.

    x: (C, T) or (B, C, T) -- LSP output, C=16 by spec
    bands: list of (low_hz, high_hz); defaults to EEG_9_BANDS_HZ
        (imported from augmentations.py so this ALWAYS matches what
        freq_mask perturbs -- do not pass a different band list here
        without also updating freq_mask, or the two will silently
        disagree about what "band-specific drift" means).
    sample_rate: MUST match the actual rate your cached epochs were
        resampled to (data_loader.py's TARGET_SFREQ=250.0 per your
        pipeline -- verify this hasn't changed before trusting the
        default here).

    Returns: (n_bands, C, T) or (B, n_bands, C, T)
    """
    if bands is None:
        bands = EEG_9_BANDS_HZ

    orig_ndim = x.ndim
    if orig_ndim == 2:
        x = x.unsqueeze(0)  # (1, C, T) -- uniform batched path below

    B, C, T = x.shape
    X_freq = torch.fft.rfft(x, dim=-1)  # (B, C, n_bins), complex
    n_bins = X_freq.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / sample_rate).to(x.device)  # (n_bins,)

    band_outputs = []
    for lo_hz, hi_hz in bands:
        bin_mask = (freqs >= lo_hz) & (freqs < hi_hz)  # (n_bins,)
        X_band = X_freq * bin_mask.view(1, 1, -1)  # zero out everything outside this band
        x_band = torch.fft.irfft(X_band, n=T, dim=-1)  # (B, C, T)
        band_outputs.append(x_band)

    out = torch.stack(band_outputs, dim=1)  # (B, n_bands, C, T)

    if orig_ndim == 2:
        out = out.squeeze(0)  # back to (n_bands, C, T)
    return out


class EEGNet9Band(nn.Module):
    """
    EEGNet-style encoder consuming 9-band-decomposed, LSP-projected
    input. Outputs a fixed-size embedding (default 64-dim) that feeds
    into projection_head.py.

    Input:  (B, n_bands, n_channels, T) -- from spectral_decompose(),
            n_channels=16 by spec (LSP's shared channel count)
    Output: (B, embedding_dim)
    """

    def __init__(self, n_bands=9, n_channels=16, D=13, F2=52,
                 kernel_len2=48, pool1=4, pool2=8, dropout=0.25,
                 embedding_dim=64):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels

        F1 = n_bands  # one input "channel" per band, fixed by the front-end

        # Block 1: depthwise spatial conv, one filter group per band --
        # collapses the 16 LSP-shared channels down to a single spatial
        # response per band, D filters per band.
        self.spatial_conv = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise temporal + pointwise across
        # bands) -- standard EEGNet block 2, lets the network combine
        # information across the 9 bands after block 1 has extracted
        # per-band spatial structure.
        self.depthwise_temporal = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, kernel_len2),
            groups=F1 * D, padding=(0, kernel_len2 // 2), bias=False
        )
        self.pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(F2, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: (B, n_bands, n_channels, T)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.depthwise_temporal(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.global_pool(x)   # (B, F2, 1, 1)
        x = x.flatten(1)          # (B, F2)
        x = self.fc(x)            # (B, embedding_dim)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class Phase0Encoder(nn.Module):
    """
    Convenience wrapper combining spectral_decompose() + EEGNet9Band
    into a single callable module, so the training loop can just do
    `embedding = encoder(lsp_output)` without manually invoking the
    band decomposition step every time.
    """

    def __init__(self, n_channels=16, sample_rate=250, bands=None, **eegnet_kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.bands = bands or EEG_9_BANDS_HZ
        self.eegnet = EEGNet9Band(
            n_bands=len(self.bands), n_channels=n_channels, **eegnet_kwargs
        )

    def forward(self, x):
        # x: (B, C=16, T) -- LSP output
        x_bands = spectral_decompose(x, bands=self.bands, sample_rate=self.sample_rate)
        return self.eegnet(x_bands)

    def param_count(self):
        return self.eegnet.param_count()


if __name__ == "__main__":
    # Smoke test: verify shapes end-to-end and confirm param count is
    # close to the spec's ~18K budget.
    encoder = Phase0Encoder(n_channels=16, sample_rate=250)
    print(f"Encoder trainable params: {encoder.param_count()} "
          f"(spec target: ~18,000)")

    # Simulate LSP's output: (B, 16, T)
    x = torch.randn(8, 16, 1001)
    out = encoder(x)
    print(f"Input shape:  {tuple(x.shape)} (simulated LSP output)")
    print(f"Output shape: {tuple(out.shape)} (expect (8, 64))")
    print(f"Output nan: {torch.isnan(out).any().item()}")

    # Verify spectral_decompose in isolation too
    single = torch.randn(16, 1001)
    decomposed = spectral_decompose(single, sample_rate=250)
    print(f"\nspectral_decompose single-epoch: {tuple(single.shape)} -> "
          f"{tuple(decomposed.shape)} (expect (9, 16, 1001))")