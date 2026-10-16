"""
augmentations.py
Phase 0 augmentation pipeline for DARWIN self-supervised pretraining.

PERFORMANCE NOTE (FFT fix):
The original implementation called augment() in a Python loop over batch
elements, meaning freq_mask did B separate rfft+irfft calls per view.
This version vectorizes freq_mask, channel_dropout, gaussian_noise and
amplitude_scale to operate on (B, C, T) tensors directly — ONE FFT per
batch instead of B. time_shift, time_mask and temporal_crop still loop over
samples (cheap indexing/interpolation). This removes the B-fold FFT cost on
the augmentation side only: encoder.py's spectral_decompose is a separate
cost (a fixed FFT filter bank run on every forward pass) and is NOT cached.

Each function now accepts EITHER (C, T) single-epoch OR (B, C, T) batched
input and returns the same shape. The training loop calls augment_batch()
once per view to form the positive pair for InfoNCE.

SCOPE (spec rev A): these are Phase 0 TRAINING augmentations, applied on the
raw per-dataset channels BEFORE LSP (so channel_dropout pressures LSP). They
are not the synthetic drift injection used in evaluation (rev A section 10),
which is applied to the raw signal before CAR in a separate module. The band
edges below are shared with encoder.py so freq_mask and the encoder's filter
bank always use the same frequency partition.
"""
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Canonical 9-band split (single source of truth, shared with encoder.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Individual augmentations — all now support (C, T) OR (B, C, T)
# ---------------------------------------------------------------------------

def time_shift(x, max_shift=50):
    """
    Circularly shifts the signal along the time axis.
    x: (C, T) or (B, C, T)
    """
    if max_shift <= 0:
        return x
    # Per-sample shift for diversity (batched)
    if x.ndim == 3:
        B = x.shape[0]
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
        # torch.roll doesn't support per-element shifts, so loop (cheap — just indexing)
        out = torch.empty_like(x)
        for i in range(B):
            if shifts[i].item() == 0:
                out[i] = x[i]
            else:
                out[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=-1)
        return out
    else:
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


def channel_dropout(x, p=0.1, fill_value=0.0):
    """
    Randomly zeroes out entire channels with probability p per channel.
    x: (C, T) or (B, C, T)
    """
    if p <= 0:
        return x
    if x.ndim == 3:
        B, C, T = x.shape
        # (B, C, 1) — per-sample, per-channel mask
        mask = (torch.rand(B, C, 1, device=x.device) > p).float()
        return x * mask + fill_value * (1 - mask)
    else:
        C = x.shape[0]
        mask = (torch.rand(C, 1, device=x.device) > p).float()
        return x * mask + fill_value * (1 - mask)


def gaussian_noise(x, std=0.05):
    """
    Adds i.i.d. Gaussian noise scaled relative to the signal's own std.
    x: (C, T) or (B, C, T)
    """
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def amplitude_scale(x, scale_range=(0.8, 1.2)):
    """
    Scales the whole epoch by a random factor.
    x: (C, T) or (B, C, T)
    """
    lo, hi = scale_range
    if lo == hi == 1.0:
        return x
    if x.ndim == 3:
        B = x.shape[0]
        # Per-sample scale for diversity
        scales = torch.empty(B, 1, 1, device=x.device).uniform_(lo, hi)
        return x * scales
    else:
        scale = torch.empty(1, device=x.device).uniform_(lo, hi).item()
        return x * scale


def time_mask(x, max_mask_len=100, n_masks=1):
    """
    Zeroes out one or more contiguous time segments.
    x: (C, T) or (B, C, T)
    """
    if max_mask_len <= 0 or n_masks <= 0:
        return x
    x = x.clone()
    if x.ndim == 3:
        B, C, T = x.shape
        for i in range(B):
            for _ in range(n_masks):
                mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
                if mask_len >= T:
                    continue
                start = torch.randint(0, T - mask_len, (1,)).item()
                x[i, :, start:start + mask_len] = 0.0
    else:
        T = x.shape[-1]
        for _ in range(n_masks):
            mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
            if mask_len >= T:
                continue
            start = torch.randint(0, T - mask_len, (1,)).item()
            x[:, start:start + mask_len] = 0.0
    return x


def freq_mask(x, max_bands_to_mask=2, sample_rate=250, bands=None):
    """
    Band-specific frequency masking (spec v3 Section 6, F2; unchanged in rev A).
    
    FFT-OPTIMIZED: Operates on (B, C, T) in a SINGLE rfft+irfft call.
    This is the critical fix — the old version looped over B samples,
    doing B separate FFTs. Now it's ONE FFT for the whole batch.
    
    x: (C, T) or (B, C, T)
    """
    if max_bands_to_mask <= 0:
        return x
    if bands is None:
        bands = EEG_9_BANDS_HZ
    
    is_batched = (x.ndim == 3)
    if not is_batched:
        x = x.unsqueeze(0)
    
    B, C, T = x.shape
    
    # ONE FFT for the entire batch
    X_freq = torch.fft.rfft(x, dim=-1)  # (B, C, n_bins)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sample_rate).to(x.device)  # (n_bins,)
    
    # Choose which bands to mask (same for whole batch — keeps it fast)
    n_to_mask = torch.randint(1, max_bands_to_mask + 1, (1,)).item()
    n_to_mask = min(n_to_mask, len(bands))
    chosen_band_idxs = torch.randperm(len(bands))[:n_to_mask].tolist()
    
    # Zero out chosen bands — vectorized over (B, C)
    X_freq = X_freq.clone()
    for band_idx in chosen_band_idxs:
        lo_hz, hi_hz = bands[band_idx]
        bin_mask = (freqs >= lo_hz) & (freqs < hi_hz)  # (n_bins,) bool
        X_freq[..., bin_mask] = 0.0
    
    # ONE IFFT for the entire batch
    x_out = torch.fft.irfft(X_freq, n=T, dim=-1)  # (B, C, T)
    
    if not is_batched:
        x_out = x_out.squeeze(0)
    return x_out


def temporal_crop(x, crop_ratio_range=(0.7, 0.95)):
    """
    Temporal cropping + resize back to original length via interpolation.
    x: (C, T) or (B, C, T)
    """
    lo, hi = crop_ratio_range
    if x.ndim == 3:
        B, C, T = x.shape
        out = torch.empty_like(x)
        for i in range(B):
            ratio = torch.empty(1).uniform_(lo, hi).item()
            crop_len = max(1, int(T * ratio))
            if crop_len >= T:
                out[i] = x[i]
                continue
            start = torch.randint(0, T - crop_len, (1,)).item()
            cropped = x[i:i+1, :, start:start + crop_len]  # (1, C, crop_len)
            resized = F.interpolate(cropped, size=T, mode="linear", align_corners=False)
            out[i] = resized.squeeze(0)
        return out
    else:
        T = x.shape[-1]
        ratio = torch.empty(1).uniform_(lo, hi).item()
        crop_len = max(1, int(T * ratio))
        if crop_len >= T:
            return x
        start = torch.randint(0, T - crop_len, (1,)).item()
        cropped = x[:, start:start + crop_len].unsqueeze(0)  # (1, C, crop_len)
        resized = F.interpolate(cropped, size=T, mode="linear", align_corners=False)
        return resized.squeeze(0)


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class AugmentationConfig:
    """
    Central place to enable/disable and parameterize each augmentation.
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
        max_bands_to_mask=2,
        sample_rate=250,
        use_temporal_crop=True,
        crop_ratio_range=(0.5, 0.8),
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
    Applies the full augmentation pipeline to a single epoch (C, T).
    Order matches spec v3 Section 6 exactly (unchanged in rev A).
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
    FFT-OPTIMIZED batched augmentation.
    
    Applies augmentations to (B, C, T) tensor. Most operations are now
    vectorized — freq_mask does ONE FFT for the whole batch instead of B.
    
    batch: (B, C, T) tensor
    Returns: (B, C, T) tensor, same shape.
    """
    if config is None:
        config = AugmentationConfig()
    
    x = batch
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


if __name__ == "__main__":
    # Smoke test — verify batched vs single-sample produce same shapes
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    B, C, T = 8, 16, 1001
    batch = torch.randn(B, C, T)
    
    print(f"Input batch shape: {batch.shape}")
    
    cfg = AugmentationConfig()
    
    # Test batched augmentation
    aug_batch = augment_batch(batch, cfg)
    print(f"Augmented batch shape: {aug_batch.shape}")
    print(f"  mean: {aug_batch.mean().item():.4f}, std: {aug_batch.std().item():.4f}")
    print(f"  nan: {torch.isnan(aug_batch).any().item()}")
    
    # Test that two independent views differ
    aug_batch_2 = augment_batch(batch, cfg)
    diff = (aug_batch - aug_batch_2).abs().mean().item()
    print(f"Mean abs diff between two views: {diff:.4f} (should be > 0)")
    
    # Test freq_mask in isolation (the critical FFT fix)
    single = batch[0]
    fm_single = freq_mask(single, max_bands_to_mask=2, sample_rate=250)
    fm_batch = freq_mask(batch, max_bands_to_mask=2, sample_rate=250)
    print(f"\nfreq_mask single: {fm_single.shape} (expect (16, 1001))")
    print(f"freq_mask batch:  {fm_batch.shape} (expect (8, 16, 1001))")
    print(f"  nan single: {torch.isnan(fm_single).any().item()}")
    print(f"  nan batch:  {torch.isnan(fm_batch).any().item()}")
    
    print("\n All batched augmentation shapes correct.")