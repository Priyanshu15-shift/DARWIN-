"""
Unified loader for DARWIN's four source datasets, via MOABB.

MOABB handles the messy per-dataset file formats (GDF, EDF, custom) so we
don't write four separate parsers. It also standardizes resampling and
epoch extraction so every dataset comes back as a plain numpy array with
consistent (subject, session) metadata attached -- which is exactly what
the preprocessing pipeline and the later train/val/test split need.

Start here: run this file directly to sanity-check that one subject from
one dataset loads correctly before touching anything else.
"""

from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np

# Maps DARWIN's dataset names to their MOABB dataset classes.
# NOTE: class names verified against moabb==1.5.0 specifically -- these have
# changed across moabb versions before (e.g. BNCI2014001 -> BNCI2014_001),
# so if you upgrade moabb later and this import breaks again, run:
#   python3 -c "import moabb.datasets as d; print([n for n in dir(d) if not n.startswith('_')])"
# and update the names below to match.
DATASET_REGISTRY = {
    "bci_iv_2a": BNCI2014_001,  # 9 subjects, 2 sessions, 22 channels, 4-class
    "bci_iv_2b": BNCI2014_004,  # 9 subjects, 5 sessions, 3 channels,  2-class
    "openbmi": Lee2019_MI,      # 54 subjects, 2 sessions, larger montage, 2-class
    "physionet": PhysionetMI,   # 109 subjects, ~1 session, larger montage
}

# Target sampling rate for all datasets -- matches the preprocessing pipeline's
# resample stage. PhysioNet's native rate (160 Hz) has the lowest Nyquist
# (80 Hz) of the four; this is the ceiling the 9-band filter bank must respect.
TARGET_SFREQ = 250.0

# --------------------------------------------------------------------------
# BANDPASS FIX (was the root cause of the dead-band problem):
# moabb.paradigms.MotorImagery defaults to fmin=8.0, fmax=32.0 -- a
# mu/beta-only bandpass meant for classic CSP-style motor imagery
# pipelines. Left at that default, 5 of augmentations.py's/encoder.py's
# 9 canonical bands (EEG_9_BANDS_HZ: delta 0.5-4, theta-low 4-6,
# theta-high 6-8, gamma-low 30-40, gamma-high 40-45) arrive dead or
# nearly dead, and preprocessing.py's 50/60 Hz notch filter becomes a
# no-op since nothing above 32 Hz survives to notch out in the first
# place.
#
# TARGET_FMAX=48.0 is chosen to stay safely under every dataset's
# post-resample Nyquist (125 Hz at TARGET_SFREQ=250.0) AND under
# PhysioNet's native-rate Nyquist (160 Hz native -> 80 Hz Nyquist,
# the tightest ceiling of the four datasets pre-resampling) -- see
# EEG_9_BANDS_HZ's top edge (45 Hz) for why 48 leaves headroom rather
# than clipping the last band exactly at its boundary.
# TARGET_FMIN=0.5 matches EEG_9_BANDS_HZ's own low edge (delta band
# starts at 0.5 Hz) -- no point requesting content below what any
# downstream band actually uses.
#
# Changing these invalidates any cache built before this fix -- a full
# re-cache (full_pool_preprocess.py, force=True or a fresh cache dir)
# and Phase 0 retrain is required after updating this.
TARGET_FMIN = 0.5
TARGET_FMAX = 48.0


def load_dataset(dataset_key, subjects=None, resample=TARGET_SFREQ,
                  fmin=TARGET_FMIN, fmax=TARGET_FMAX):
    """
    Load one of the four DARWIN source datasets via MOABB.

    Parameters
    ----------
    dataset_key : str
        One of "bci_iv_2a", "bci_iv_2b", "openbmi", "physionet".
    subjects : list[int] or None
        Which subjects to load. None loads all subjects in the dataset --
        expensive for openbmi/physionet, so pass a small list while developing.
    resample : float
        Target sampling rate in Hz. MOABB resamples during epoch extraction.
    fmin, fmax : float
        Bandpass edges in Hz, passed straight to MOABB's MotorImagery
        paradigm. Explicitly overriding MOABB's own defaults (8/32 Hz)
        here is required -- left unset, MOABB silently restricts every
        dataset to a mu/beta-only bandpass, starving 5 of the 9 bands
        EEG_9_BANDS_HZ (augmentations.py/encoder.py) expects to see.
        Defaults to TARGET_FMIN/TARGET_FMAX above; pass explicit values
        only if you have a specific reason to deviate per-call.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
        Raw (not yet DARWIN-preprocessed) epoched EEG.
    y : np.ndarray, shape (n_epochs,)
        Class labels as strings (e.g. "left_hand", "right_hand").
    metadata : pandas.DataFrame
        One row per epoch, with at least 'subject' and 'session' columns,
        plus a 'dataset_id' column added here for downstream tagging.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Choose from {list(DATASET_REGISTRY)}"
        )

    dataset = DATASET_REGISTRY[dataset_key]()
    paradigm = MotorImagery(resample=resample, fmin=fmin, fmax=fmax)

    subj_list = subjects if subjects is not None else dataset.subject_list
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subj_list)

    metadata = metadata.copy()
    metadata["dataset_id"] = dataset_key
    return X, y, metadata


def load_dataset_with_retry(dataset_key, subjects=None, resample=TARGET_SFREQ,
                             fmin=TARGET_FMIN, fmax=TARGET_FMAX,
                             max_retries=3, retry_delay=15):
    """
    Same as load_dataset, but retries on failure with a delay between
    attempts. Use this instead of load_dataset directly for any dataset
    with a large single-file download (OpenBMI especially -- its files
    run several hundred MB each) where a transient network stall is more
    likely than with the smaller BCI-IV files.

    fmin, fmax : float
        Forwarded to load_dataset -- see its docstring. Kept as explicit
        parameters here (not just **kwargs) so a bandpass override is
        visible at the call site, not buried.
    max_retries : int
        Total attempts before giving up and re-raising the last error.
    retry_delay : float
        Seconds to wait between attempts. Kept short here since most
        transient network issues resolve within seconds to a couple of
        minutes -- if it's a sustained block (e.g. the network actively
        throttling a specific host), retrying won't help and you'll see
        the same failure repeat max_retries times before raising.
    """
    import time
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(dataset_key, subjects=subjects, resample=resample,
                                 fmin=fmin, fmax=fmax)
        except Exception as e:
            last_exception = e
            print(f"  Attempt {attempt}/{max_retries} failed: "
                  f"{type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)
    raise last_exception


def load_all_datasets(subjects_per_dataset=None, fmin=TARGET_FMIN, fmax=TARGET_FMAX):
    """
    Load all four datasets and return them as a dict keyed by dataset_id.

    subjects_per_dataset : dict[str, list[int]] or None
        Optional per-dataset subject subset, e.g. {"physionet": [1, 2, 3]}
        to keep early development runs fast. Datasets not in the dict load
        every subject.
    fmin, fmax : float
        Forwarded to load_dataset for every dataset in the registry --
        see load_dataset's docstring for why these must be set explicitly.
    """
    subjects_per_dataset = subjects_per_dataset or {}
    out = {}
    for key in DATASET_REGISTRY:
        subs = subjects_per_dataset.get(key, None)
        print(f"Loading {key} (subjects={'all' if subs is None else subs}) ...")
        X, y, meta = load_dataset(key, subjects=subs, fmin=fmin, fmax=fmax)
        out[key] = (X, y, meta)
        print(f"  -> X shape {X.shape}, {len(np.unique(y))} classes, "
              f"{meta['subject'].nunique()} subjects, "
              f"{meta['session'].nunique()} unique session labels")
    return out


if __name__ == "__main__":
    # Smallest possible sanity check: one subject, one dataset.
    # This is the very first thing to run tonight -- if this works,
    # the rest of the loading logic will work the same way for the others.
    X, y, meta = load_dataset("bci_iv_2a", subjects=[1])
    print("X shape:", X.shape)          # (n_epochs, n_channels, n_times)
    print("labels:", np.unique(y))
    print(f"bandpass: {TARGET_FMIN}-{TARGET_FMAX} Hz (was MOABB default 8-32 Hz)")
    print(meta[["subject", "session"]].head())