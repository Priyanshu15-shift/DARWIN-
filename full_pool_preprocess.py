"""
Full-pool preprocessing: run every subject of every dataset through the
DARWIN preprocessing pipeline once, and cache the result to disk as .npy
files -- one file per (dataset, subject) pair, so a partial run can be
resumed without re-processing everything from scratch.

Structured to run on whichever datasets are ready. OpenBMI is included in
the registry but can simply be skipped for now (via the DATASETS_TO_RUN
list below) if its download hasn't finished yet -- add it back in and
rerun once it has, no need to redo the other three.

Cache layout:
    cache/<dataset_id>/subject_<N>.npz
        contains: X (preprocessed epochs), y (labels), subject, session
        per-epoch arrays, plus the clipping/normalization diagnostics.
"""

import os
import time
import numpy as np
from data_loader import load_dataset_with_retry, DATASET_REGISTRY, TARGET_SFREQ
from preprocessing import preprocess

CACHE_DIR = "cache"

NOTCH_FREQ_BY_DATASET = {
    "bci_iv_2a": 50.0,
    "bci_iv_2b": 50.0,
    "openbmi": 60.0,
    "physionet": 60.0,
}

# Edit this list to control which datasets actually run right now.
# Drop "openbmi" from here until its download finishes, then add it back
# and rerun -- the other three won't be touched again since they're
# already cached (see the skip-if-cached check below).
DATASETS_TO_RUN = ["bci_iv_2a", "bci_iv_2b", "physionet","openbmi"]


def get_subject_list(dataset_key, limit=None):
    """Returns the subject list for a dataset, optionally capped for testing."""
    dataset = DATASET_REGISTRY[dataset_key]()
    subjects = dataset.subject_list
    if limit is not None:
        subjects = subjects[:limit]
    return subjects


def cache_path(dataset_key, subject):
    return os.path.join(CACHE_DIR, dataset_key, f"subject_{subject}.npz")


def process_and_cache_subject(dataset_key, subject, force=False):
    """
    Load + preprocess one subject, cache to disk. Skips work entirely if
    a cache file already exists for this (dataset, subject) pair, unless
    force=True -- this is what makes the script resumable: rerunning it
    after an interruption just skips everyone already done.
    """
    out_path = cache_path(dataset_key, subject)

    if os.path.exists(out_path) and not force:
        print(f"  [skip] {dataset_key} subject {subject} already cached")
        return True

    try:
        X, y, meta = load_dataset_with_retry(dataset_key, subjects=[subject])
        notch_freq = NOTCH_FREQ_BY_DATASET[dataset_key]
        X_processed, diagnostics = preprocess(
            X, sfreq=TARGET_SFREQ, notch_base_freq=notch_freq
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            X=X_processed,
            y=y,
            subject=meta["subject"].values,
            session=meta["session"].values,
            pct_clipped_per_epoch=diagnostics["pct_clipped_per_epoch"],
        )
        print(f"  [done] {dataset_key} subject {subject}: "
              f"{X_processed.shape}, "
              f"mean clip {diagnostics['pct_clipped_per_epoch'].mean():.3f}%")
        return True

    except Exception as e:
        print(f"  [FAILED] {dataset_key} subject {subject}: "
              f"{type(e).__name__}: {e}")
        return False


def run_full_pool(datasets=DATASETS_TO_RUN, subject_limit=None):
    """
    subject_limit: cap subjects per dataset (useful for a quick end-to-end
    test before committing to the full multi-hour run across ~180 subjects).
    """
    overall_start = time.time()
    results = {}

    for dataset_key in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_key}")
        print('='*60)

        subjects = get_subject_list(dataset_key, limit=subject_limit)
        print(f"  {len(subjects)} subjects to process")

        dataset_start = time.time()
        ok_count = 0
        for subject in subjects:
            success = process_and_cache_subject(dataset_key, subject)
            ok_count += int(success)
        elapsed = time.time() - dataset_start

        results[dataset_key] = (ok_count, len(subjects))
        print(f"  {dataset_key}: {ok_count}/{len(subjects)} succeeded "
              f"in {elapsed/60:.1f} min")

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print("FULL POOL SUMMARY")
    print('='*60)
    for k, (ok, total) in results.items():
        print(f"  {k}: {ok}/{total}")
    print(f"  Total time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    # Start with subject_limit=2 the first time you run this -- confirms
    # the caching + resume logic works correctly on a small, fast run
    # before committing to the full ~180-subject pool, which will take
    # meaningfully longer.
    run_full_pool()
