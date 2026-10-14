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
    cache/<PIPELINE_VERSION>/<dataset_id>/subject_<N>.npz
        contains: X (preprocessed epochs, float32), y (labels), subject,
        session per-epoch arrays, pipeline_version (string, for a sanity
        check anywhere the cache is read), plus the clipping/
        normalization diagnostics: pct_clipped_per_epoch,
        pre_norm_median, pre_norm_mad.

FIX LOGGED (cache invalidation): earlier versions of this script cached
straight to cache/<dataset_id>/... with no version marker. The
skip-if-cached check has no way to know the preprocessing pipeline
changed underneath it (e.g. the bandpass 8-32Hz -> 0.5-48Hz fix and the
added CAR step) -- rerunning against the old flat cache path would
silently skip every subject and report success on stale, pre-fix data.
CACHE_DIR now includes PIPELINE_VERSION, so a pipeline change
automatically lands in a fresh directory with nothing to skip, rather
than relying on remembering to delete the old cache by hand. Bump
PIPELINE_VERSION any time preprocess()'s actual behavior changes.
"""

import os
import time
import numpy as np
from data_loader import load_dataset_with_retry, DATASET_REGISTRY, TARGET_SFREQ
from preprocessing import preprocess
from cache_config import PIPELINE_VERSION, CACHE_DIR

# PIPELINE_VERSION and CACHE_DIR now live in cache_config.py (stdlib-only,
# no moabb dependency) so probe scripts and anything else that only
# reads the cache can import the same path without pulling in this
# file's data_loader -> moabb chain. Bump PIPELINE_VERSION there, not
# here, any time preprocessing.py's actual behavior changes.

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
    a cache file already exists for this (dataset, subject) pair under
    the CURRENT PIPELINE_VERSION's directory, unless force=True -- this
    is what makes the script resumable: rerunning it after an
    interruption just skips everyone already done, and rerunning it
    after a preprocessing change lands in a fresh versioned directory
    instead of skipping stale data (see module docstring).

    Cache includes pre_norm_median and pre_norm_mad (shape
    (n_epochs, n_channels, 1) each, matching preprocess()'s diagnostics
    output) alongside pct_clipped_per_epoch, and pipeline_version as a
    string for an optional downstream sanity check.
    """
    out_path = cache_path(dataset_key, subject)

    if os.path.exists(out_path) and not force:
        print(f"  [skip] {dataset_key} subject {subject} already cached "
              f"(pipeline_version={PIPELINE_VERSION})")
        return True

    try:
        X, y, meta = load_dataset_with_retry(dataset_key, subjects=[subject])
        notch_freq = NOTCH_FREQ_BY_DATASET[dataset_key]
        X_processed, diagnostics = preprocess(
            X, sfreq=TARGET_SFREQ, notch_base_freq=notch_freq
        )

        # float32 halves disk/load time vs MOABB's native float64, with
        # no representational loss at this scale -- confirm the probe/
        # encoder path is fine with float32 tensors (torch.tensor(...,
        # dtype=torch.float32) in sampler.py already expects float32).
        X_processed = X_processed.astype(np.float32)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            pipeline_version=PIPELINE_VERSION,
            X=X_processed,
            y=y,
            subject=meta["subject"].values,
            session=meta["session"].values,
            pct_clipped_per_epoch=diagnostics["pct_clipped_per_epoch"],
            pre_norm_median=diagnostics["pre_norm_median"],
            pre_norm_mad=diagnostics["pre_norm_mad"],
        )
        print(f"  [done] {dataset_key} subject {subject}: "
              f"{X_processed.shape} ({X_processed.dtype}), "
              f"mean clip {diagnostics['pct_clipped_per_epoch'].mean():.3f}%, "
              f"pre_norm_median/mad shapes "
              f"{diagnostics['pre_norm_median'].shape}/"
              f"{diagnostics['pre_norm_mad'].shape}")
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
    print(f"Pipeline version: {PIPELINE_VERSION}")
    print(f"Cache dir: {CACHE_DIR}")

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
    # First run: subject_limit=2 to confirm the caching + resume logic,
    # and the new bandpass/CAR path specifically, work correctly on a
    # small, fast run before committing to the full ~180-subject pool.
    # The per-subject [done] printout now shows dtype, clip %, and
    # pre_norm shapes -- your first confirmation the new pipeline path
    # is healthy. Once that looks right, rerun with subject_limit=None
    # (or just call run_full_pool() with no args) for the full pool.
    run_full_pool()