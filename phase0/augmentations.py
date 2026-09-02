"""
augmentations.py

Phase 0 augmentation pipeline for DARWIN self-supervised pretraining.

Each function takes a single epoch tensor of shape (C, T) — channels x
timepoints — and returns an augmented version of the same shape. The
training loop calls `augment()` TWICE per epoch (once per view) to form
the positive pair for InfoNCE.

Augmentation choices here follow common practice in EEG self-supervised
learning (e.g. BENDR, EEG-SimCLR-style setups): each augmentation
perturbs a different axis of variation the encoder needs to be invariant
to — sensor noise, small temporal misalignment, single-channel dropout/
failure, and local amplitude drift — without destroying the underlying
neural signal structure. Defaults are conservative starting points, not
tuned to your specific datasets; treat them as a first pass to validate
the pipeline, then adjust based on what the eval probe (step 7) tells you.

Design principle: every augmentation is individually toggleable and
seeded off a single config object, so you can ablate ("does removing
channel dropout change downstream probe accuracy?") without touching
this file.
"""

import torch


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def time_shift(x, max_shift=50):
    """
    Circularly shifts the signal along the time axis by a random offset in
    [-max_shift, max_shift]. Simulates small misalignment in epoch
    windowing/trigger timing across sessions.

    x: (C, T)
    max_shift: max shift in timepoints (e.g. 50 samples @ 250Hz = 200ms)
    """
    if max_shift <= 0:
        return x
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)


def channel_dropout(x, p=0.1, fill_value=0.0):
    """
    Randomly zeroes out entire channels with probability p per channel.
    Simulates electrode dropout / bad contact, a real failure mode in EEG
    recording, and forces the encoder not to over-rely on any single
    channel — useful for cross-subject robustness since channel
    reliability varies by subject/session.

    x: (C, T)
    p: per-channel dropout probability
    """
    if p <= 0:
        return x
    C = x.shape[0]
    mask = (torch.rand(C) > p).float().unsqueeze(-1)  # (C, 1)
    return x * mask + fill_value * (1 - mask)


def gaussian_noise(x, std=0.05):
    """
    Adds i.i.d. Gaussian noise scaled relative to the signal's own std.
    Simulates sensor/amplifier noise. Since data is pre-normalized
    (std ~1), std=0.05 means noise at ~5% of signal amplitude — mild by
    design; too much noise here can destroy the very structure the
    encoder needs to learn from.

    x: (C, T)
    std: noise standard deviation, relative to normalized signal scale
    """
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def amplitude_scale(x, scale_range=(0.8, 1.2)):
    """
    Scales the whole epoch by a single random factor. Simulates
    inter-subject/inter-session gain variation (electrode impedance,
    amplifier calibration differences) — this is one of the exact
    sources of cross-subject variance DARWIN is meant to be robust to,
    so encouraging invariance to it here is directly on-purpose.

    x: (C, T)
    scale_range: (min, max) uniform sampling range for the scale factor
    """
    lo, hi = scale_range
    if lo == hi == 1.0:
        return x
    scale = torch.empty(1).uniform_(lo, hi).item()
    return x * scale


def time_mask(x, max_mask_len=100, n_masks=1):
    """
    Zeroes out one or more contiguous time segments. Simulates transient
    artifacts (blinks, muscle artifacts, brief signal loss) and encourages
    the encoder to rely on distributed temporal structure rather than any
    single window.

    x: (C, T)
    max_mask_len: max length of each masked segment, in timepoints
    n_masks: number of masked segments to apply
    """
    if max_mask_len <= 0 or n_masks <= 0:
        return x
    x = x.clone()
    T = x.shape[-1]
    for _ in range(n_masks):
        mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
        if mask_len >= T:
            continue
        start = torch.randint(0, T - mask_len, (1,)).item()
        x[:, start:start + mask_len] = 0.0
    return x


# Canonical 9-band split used by the EEGNet 9-band spectral decomposition.
# Spec v3 Section 6 specifies "max_bands_to_mask=2" but does not give exact
# band edges. These follow the standard EEG sub-band convention (splitting
# the conventional delta/theta/alpha/beta/gamma ranges finer to reach 9
# bands spanning 0.5-45Hz). IMPORTANT: if the actual EEGNet backbone
# implementation uses different cutoffs, replace this list with the exact
# values so freq_mask masks the SAME bands the encoder decomposes into —
# otherwise this augmentation and the encoder's spectral structure are
# talking about different frequency partitions, which defeats the point.
EEG_9_BANDS_HZ = [
    (0.5, 4),    # delta
    (4, 6),      # theta-low
    (6, 8),      # theta-high
    (8, 10),     # alpha-low
    (10, 13),    # alpha-high
    (13, 20),    # beta-low
    (20, 30),    # beta-high
    (30, 40),    # gamma-low
    (40, 45),    # gamma-high
]


def freq_mask(x, max_bands_to_mask=2, sample_rate=250, bands=None):
    """
    Band-specific frequency masking (spec v3, Section 6, F2). Zeroes out
    ALL frequency content within 1-2 randomly chosen physiological bands
    (of the 9 the EEGNet backbone spectrally decomposes into) — not
    arbitrary FFT bin ranges.

    This distinction matters: since the encoder consumes 9 separate
    spectral bands, masking needs to align with THOSE band boundaries to
    actually pressure invariance to band-specific drift (e.g. a session
    where alpha power is shifted). Masking a random bin range that
    straddles band boundaries doesn't target the encoder's actual input
    structure the way the spec intends.

    x: (C, T)
    max_bands_to_mask: mask between 1 and this many bands (spec: 2)
    sample_rate: Hz, needed to convert band edges (Hz) into FFT bin
        indices correctly. MUST match the actual sample rate your cached
        epochs were recorded/resampled at — verify against
        preprocessing.py rather than assuming 250Hz.
    bands: list of (low_hz, high_hz) tuples; defaults to EEG_9_BANDS_HZ.
        Pass your EEGNet backbone's exact band edges here if they differ
        from the assumed convention above.
    """
    if max_bands_to_mask <= 0:
        return x

    if bands is None:
        bands = EEG_9_BANDS_HZ

    T = x.shape[-1]
    X_freq = torch.fft.rfft(x, dim=-1)  # (C, T//2 + 1), complex
    n_bins = X_freq.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / sample_rate)  # (n_bins,), Hz per bin

    n_to_mask = torch.randint(1, max_bands_to_mask + 1, (1,)).item()
    n_to_mask = min(n_to_mask, len(bands))
    chosen_band_idxs = torch.randperm(len(bands))[:n_to_mask].tolist()

    X_freq = X_freq.clone()
    for band_idx in chosen_band_idxs:
        lo_hz, hi_hz = bands[band_idx]
        bin_mask = (freqs >= lo_hz) & (freqs < hi_hz)  # (n_bins,) bool
        X_freq[:, bin_mask] = 0.0

    x_out = torch.fft.irfft(X_freq, n=T, dim=-1)
    return x_out


def temporal_crop(x, crop_ratio_range=(0.7, 0.95)):
    """
    Temporal cropping (spec v3, Section 7.2). Crops a random contiguous
    sub-window of the epoch, then resizes back to the original length via
    linear interpolation, so downstream shape stays fixed at (C, T).

    Distinct from `time_mask`: time_mask zeroes a segment but keeps the
    full-length context around it, so the encoder can still "see" the
    whole trial and just has to fill in a gap. `temporal_crop` actually
    reduces the observed trial duration, forcing the encoder to recognize
    task structure from a genuinely partial observation — this is the
    property the trial/class probe diagnostic needs: if the encoder can
    only recognize trials at full length, it hasn't learned robust,
    duration-invariant task structure, which is a real risk if this
    augmentation is skipped.

    x: (C, T)
    crop_ratio_range: (min, max) fraction of original length to keep,
        e.g. 0.7 means keep a random contiguous 70% of the epoch before
        resizing back up to T.
    """
    lo, hi = crop_ratio_range
    T = x.shape[-1]
    ratio = torch.empty(1).uniform_(lo, hi).item()
    crop_len = max(1, int(T * ratio))
    if crop_len >= T:
        return x

    start = torch.randint(0, T - crop_len, (1,)).item()
    cropped = x[:, start:start + crop_len]  # (C, crop_len)

    # Resize back to original T via linear interpolation so shape stays
    # fixed for downstream batching — interpolate expects (N, C, L)
    cropped = cropped.unsqueeze(0)  # (1, C, crop_len)
    resized = torch.nn.functional.interpolate(
        cropped, size=T, mode="linear", align_corners=False
    )
    return resized.squeeze(0)  # (C, T)


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class AugmentationConfig:
    """
    Central place to enable/disable and parameterize each augmentation.
    Pass an instance of this (or override defaults) into `augment()`.

    Defaults are a reasonable starting point for normalized EEG epochs.
    Treat these as hyperparameters to tune once the eval probe (step 7)
    gives you a signal on what helps vs. hurts.
    """
    def __init__(
        self,
        use_time_shift=True,
        max_shift=50,

        use_channel_dropout=True,
        channel_dropout_p=0.25,

        use_gaussian_noise=True,
        noise_std=0.08,

        use_amplitude_scale=True,
        amplitude_scale_range=(0.6, 1.4),

        use_time_mask=True,
        max_mask_len=150,
        n_masks=2,

        use_freq_mask=True,
        max_bands_to_mask=2,  # spec v3 Section 6
        sample_rate=250,      # VERIFY against preprocessing.py — do not assume

        use_temporal_crop=True,
        crop_ratio_range=(0.5, 0.8),  # spec v3 Section 6: crop_ratio=(0.7, 0.9)
    ):
        self.use_time_shift = use_time_shift
        self.max_shift = max_shift

        self.use_channel_dropout = use_channel_dropout
        self.channel_dropout_p = channel_dropout_p

        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std

        self.use_amplitude_scale = use_amplitude_scale
        self.amplitude_scale_range = amplitude_scale_range

        self.use_time_mask = use_time_mask
        self.max_mask_len = max_mask_len
        self.n_masks = n_masks

        self.use_freq_mask = use_freq_mask
        self.max_bands_to_mask = max_bands_to_mask
        self.sample_rate = sample_rate

        self.use_temporal_crop = use_temporal_crop
        self.crop_ratio_range = crop_ratio_range


def augment(x, config=None):
    """
    Applies the full augmentation pipeline to a single epoch, in the
    EXACT order spec v3 Section 6 specifies ("For each view, apply in
    this fixed order:"):

        time_shift -> amplitude_scale -> gaussian_noise ->
        channel_dropout -> time_mask -> temporal_crop -> freq_mask

    CORRECTION NOTE: an earlier version of this function used a
    different order (reasoned from pipeline-design intuition rather
    than re-checked against the spec table), which was a real
    deviation from spec. Fixed here to match Section 6 exactly.

    x: (C, T) tensor -- a single raw cached epoch
    config: AugmentationConfig instance; uses defaults if None

    Returns: (C, T) tensor, same shape, augmented.
    """
    if config is None:
        config = AugmentationConfig()

    if config.use_time_shift:
        x = time_shift(x, max_shift=config.max_shift)

    if config.use_amplitude_scale:
        x = amplitude_scale(x, scale_range=config.amplitude_scale_range)

    if config.use_gaussian_noise:
        x = gaussian_noise(x, std=config.noise_std)

    if config.use_channel_dropout:
        x = channel_dropout(x, p=config.channel_dropout_p)

    if config.use_time_mask:
        x = time_mask(x, max_mask_len=config.max_mask_len, n_masks=config.n_masks)

    if config.use_temporal_crop:
        x = temporal_crop(x, crop_ratio_range=config.crop_ratio_range)

    if config.use_freq_mask:
        x = freq_mask(x, max_bands_to_mask=config.max_bands_to_mask, sample_rate=config.sample_rate)

    return x


def augment_batch(batch, config=None):
    """
    Applies `augment()` independently to each epoch in a batch.
    Each epoch gets its OWN random augmentation draw — this is
    important, batching should not share augmentation parameters
    across epochs or you lose the intended diversity.

    batch: (B, C, T) tensor
    Returns: (B, C, T) tensor, same shape.
    """
    return torch.stack([augment(x, config) for x in batch], dim=0)


if __name__ == "__main__":
    # Smoke test — run directly to sanity check on a real cached epoch.
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from sampler import make_phase0_dataloader

    loader = make_phase0_dataloader(
        cache_dir="./cache",
        datasets=["bci_iv_2a"],
        batch_size=8,
        num_workers=0,
    )
    batch = next(iter(loader))

    print(f"Original batch  — shape: {batch.shape}, "
          f"mean: {batch.mean().item():.4f}, std: {batch.std().item():.4f}, "
          f"nan: {torch.isnan(batch).any().item()}")

    cfg = AugmentationConfig()
    aug_batch = augment_batch(batch, cfg)

    print(f"Augmented batch — shape: {aug_batch.shape}, "
          f"mean: {aug_batch.mean().item():.4f}, std: {aug_batch.std().item():.4f}, "
          f"nan: {torch.isnan(aug_batch).any().item()}")

    # Two independent views of the same batch should differ from each other
    aug_batch_2 = augment_batch(batch, cfg)
    diff = (aug_batch - aug_batch_2).abs().mean().item()
    print(f"Mean abs diff between two independent augmented views: {diff:.4f} "
          f"(should be > 0 — confirms randomness is actually applied)")

    # Isolate freq_mask and temporal_crop specifically — these are the two
    # riskiest additions (FFT round-trip, interpolation resize), worth
    # checking in isolation rather than trusting the combined pipeline
    single_epoch = batch[0]
    print(f"\nSingle epoch shape: {single_epoch.shape}")

    fm = freq_mask(single_epoch, max_bands_to_mask=2, sample_rate=250)
    print(f"freq_mask      — shape: {fm.shape} (should match input), "
          f"nan: {torch.isnan(fm).any().item()}, "
          f"mean/std: {fm.mean().item():.4f}/{fm.std().item():.4f}")

    tc = temporal_crop(single_epoch, crop_ratio_range=(0.7, 0.95))
    print(f"temporal_crop  — shape: {tc.shape} (should match input), "
          f"nan: {torch.isnan(tc).any().item()}, "
          f"mean/std: {tc.mean().item():.4f}/{tc.std().item():.4f}")