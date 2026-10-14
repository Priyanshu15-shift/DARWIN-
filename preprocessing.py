"""
DARWIN preprocessing pipeline: the epoch-level stages that run on top of
MOABB-loaded data (resampling and epoch extraction are already handled by
data_loader.py via the MotorImagery paradigm).

Pipeline order, matching the finalized spec:
    ... (resample -- handled upstream in data_loader.py) ...
    -> CAR (common average reference, per-epoch)
    -> Notch filtering (mains + harmonics)
    -> MAD artifact clipping (per-epoch, robust, logs % clipped)
    -> Robust amplitude norm (per-epoch median/MAD, logs pre-norm stats)
    -> [LSP happens next, in a separate module -- not yet built]

Note on CAR and notch filtering: for full rigor these should be applied at
the raw, pre-epoch level (cleaner filtering, no edge artifacts at epoch
boundaries). MOABB's default paradigm does not expose raw-level CAR/notch
hooks without a custom paradigm subclass. For this first working version,
we apply both CAR and notch filtering at the epoch level, which is close
enough to unblock Phase 0 development -- upgrading to raw-level CAR/notch
is a reasonable, low-risk refinement to make once the rest of the pipeline
is running end to end, not a blocker to starting.

FIX LOGGED: an earlier version of this module's docstring promised CAR as
part of the pipeline ("Raw EEG -> CAR+MAD -> LSP -> Encoder" per the
blueprint) but the actual preprocess() chain never called it -- notch,
MAD clip, and amplitude norm ran on un-referenced data. car_reference()
below is the fix; preprocess() now calls it first, matching the pipeline
order documented above and in the blueprint.
"""

import numpy as np
from scipy import signal


def car_reference(X):
    """
    Common Average Reference, applied per-epoch: subtracts each epoch's
    across-channel mean (at every timepoint) from every channel. This is
    the piece that was documented in this module's own pipeline-order
    docstring and in the blueprint ("Raw EEG -> CAR+MAD -> LSP ->
    Encoder") but never actually implemented -- preprocess() previously
    ran notch/MAD-clip/amplitude-norm directly on amplifier-referenced
    data.

    Per-epoch (not per-recording/raw-level) CAR is a deliberate scope
    choice, not full rigor: raw-level CAR before epoching would avoid
    any epoch-boundary edge effects, but MOABB's default paradigm here
    doesn't expose a raw-level hook without a custom paradigm subclass
    (see module docstring). Per-epoch CAR is the same math applied at a
    coarser grain and is enough to unblock the rest of the pipeline.

    X : np.ndarray, shape (n_epochs, n_channels, n_times)

    Returns
    -------
    X_car : np.ndarray, same shape as input. For every epoch and every
        timepoint, the mean across channels has been subtracted, so each
        timepoint's channel values sum to (approximately) zero.
    """
    channel_mean = X.mean(axis=1, keepdims=True)  # (n_epochs, 1, n_times)
    return X - channel_mean


def notch_filter_epochs(X, sfreq, base_freq=50.0, n_harmonics=2, bandwidth=4.0):
    """
    Apply notch filtering at base_freq and its harmonics to remove mains
    interference. base_freq should be 50.0 for European-recorded datasets
    (BCI-IV 2a/2b) and 60.0 for US/Korean-recorded datasets (PhysioNet,
    OpenBMI) -- pass the correct value per dataset, do not assume one value
    for all four.

    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    out = X.copy()
    for k in range(1, n_harmonics + 1):
        freq = base_freq * k
        # narrow bandstop around the mains frequency and each harmonic
        b, a = signal.iirnotch(freq, Q=freq / bandwidth, fs=sfreq)
        out = signal.filtfilt(b, a, out, axis=-1)
    return out


def mad_clip_epoch(X, n_mad=5.0):
    """
    Robust per-epoch artifact clipping using median absolute deviation.
    Returns the clipped array and the percentage of samples clipped per
    epoch -- log this per your finalized pipeline spec, it's a cheap
    data-quality diagnostic.

    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    med = np.median(X, axis=-1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=-1, keepdims=True) + 1e-8

    lower = med - n_mad * mad
    upper = med + n_mad * mad

    was_clipped = (X < lower) | (X > upper)
    pct_clipped_per_epoch = was_clipped.mean(axis=(1, 2)) * 100.0  # per epoch

    clipped = np.clip(X, lower, upper)
    return clipped, pct_clipped_per_epoch


def robust_amplitude_norm(X):
    """
    Per-epoch robust amplitude normalization using median/MAD (not
    whole-recording stats, per the finalized spec -- this is deliberate:
    we want DTKF later tracking genuine representational drift, not
    trivial amplitude-scale drift that gets normalized away here).

    Returns the normalized array plus the pre-normalization median and MAD
    per epoch, which should be logged as metadata -- this is what lets you
    later check whether any observed DTKF drift just re-derives what got
    normalized away here.

    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    med = np.median(X, axis=-1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=-1, keepdims=True) + 1e-8

    normed = (X - med) / (mad * 1.4826)  # 1.4826 makes MAD ~= std for Gaussian data
    return normed, med, mad


def preprocess(X, sfreq, notch_base_freq=50.0, n_mad_clip=5.0, apply_car=True):
    """
    Run the full epoch-level pipeline on a batch of epochs from one dataset.

    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    sfreq : float, the sampling rate X is currently at (should already be
        TARGET_SFREQ from data_loader.py by this point)
    notch_base_freq : 50.0 or 60.0 depending on the dataset's country of
        origin -- pass explicitly per dataset, do not hardcode
    apply_car : bool, default True. Runs car_reference() first, matching
        the documented "CAR -> notch -> MAD clip -> amplitude norm"
        order. Exposed as a flag (rather than hardcoded) so an ablation
        can isolate CAR's effect if that's ever useful, but the default
        matches the spec and should stay on for any real run.

    Returns
    -------
    X_processed : np.ndarray, same shape as input
    diagnostics : dict with 'pct_clipped_per_epoch', 'pre_norm_median',
        'pre_norm_mad' -- log these per epoch as recommended in the spec
    """
    X_referenced = car_reference(X) if apply_car else X
    X_notched = notch_filter_epochs(X_referenced, sfreq, base_freq=notch_base_freq)
    X_clipped, pct_clipped = mad_clip_epoch(X_notched, n_mad=n_mad_clip)
    X_normed, pre_norm_med, pre_norm_mad = robust_amplitude_norm(X_clipped)

    diagnostics = {
        "pct_clipped_per_epoch": pct_clipped,
        "pre_norm_median": pre_norm_med,
        "pre_norm_mad": pre_norm_mad,
    }
    return X_normed, diagnostics


if __name__ == "__main__":
    # Quick synthetic smoke test -- doesn't need real data, just checks
    # the pipeline runs and produces sane shapes before you point it at
    # actual loaded epochs.
    rng = np.random.default_rng(0)
    fake_X = rng.normal(size=(5, 22, 250)).astype(np.float64)  # 5 epochs, 22 ch, 1s @ 250Hz

    X_out, diag = preprocess(fake_X, sfreq=250.0, notch_base_freq=50.0)
    print("Output shape:", X_out.shape)
    print("Pct clipped per epoch:", diag["pct_clipped_per_epoch"])
    assert X_out.shape == fake_X.shape
    print("Smoke test passed.")

    # Sanity check CAR itself in isolation: after car_reference, every
    # epoch's per-timepoint channel mean should be ~0.
    car_out = car_reference(fake_X)
    max_residual_mean = np.abs(car_out.mean(axis=1)).max()
    print(f"CAR check: max |per-timepoint channel mean| after CAR = "
          f"{max_residual_mean:.2e} (expect ~0)")
    assert max_residual_mean < 1e-10
    print("CAR smoke test passed.")