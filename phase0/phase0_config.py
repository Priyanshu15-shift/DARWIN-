"""
phase0_config.py

One place for the Phase 0 SESSION POLICY (DARWIN spec rev A, R1):
Phase 0 must never see the held-out test session(s) of any subject, so the
sessions it may train on are listed here once instead of being retyped (and
mistyped) in every command.

WHAT THIS FILE IS
    - TRAIN_SESSIONS: the session labels Phase 0 may see, per multi-session
      dataset. This is the single source of truth.
    - Helpers that turn it into the `--sessions` arguments train_phase0.py
      expects (same DATASET=label[,label] format as sampler.parse_session_args).
    - A cache check (`--check`) that compares the labels written here with the
      labels actually stored in the cached .npz files, so a typo is caught
      BEFORE a training run rather than after.

WHAT IT IS NOT
    train_phase0.py does not import this file; nothing in training changes.
    It only generates / checks the arguments you pass to it. PhysioNet is
    deliberately absent from TRAIN_SESSIONS: it has a single session, no
    cross-session claim, and train_phase0.py exempts it from the session check.

VERIFY THE LABELS ONCE
    The labels below follow the MOABB naming used in the sampler docstrings
    ("0train", "1train", ...). They are the expected values, not confirmed
    against your cache. Run `python phase0_config.py --check` (or
    inspect_cache.py) once and fix TRAIN_SESSIONS if the printout differs.
    BCI IV-2b's test sessions (assumed to be the last two) must be confirmed
    against the release actually cached (spec rev A, section 16).

Usage:
    python phase0_config.py --check            # compare config with the cache
    python phase0_config.py --print_cmd        # print the train_phase0.py command
    python phase0_config.py --cli              # just the --sessions arguments
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from cache_config import CACHE_DIR  # stdlib-only, shared versioned cache path

# Datasets pooled in Phase 0 right now. openbmi is deferred (download incomplete).
DATASETS = ["bci_iv_2a", "bci_iv_2b", "physionet"]

# Session labels Phase 0 MAY see, per multi-session dataset. EDIT HERE ONLY.
TRAIN_SESSIONS = {
    "bci_iv_2a": ["0train"],
    "bci_iv_2b": ["0train", "1train", "2train"],
}

# Expected held-out (test) labels. Reference and sanity check only: `--check`
# warns if the cache disagrees. Never passed to training.
EXPECTED_HELD_OUT = {
    "bci_iv_2a": ["1test"],
    "bci_iv_2b": ["3test", "4test"],
}

# Single-session datasets: no session list needed.
SINGLE_SESSION_DATASETS = {"physionet"}


def sessions_dict(datasets=None):
    """{dataset: [labels]} for the requested datasets (single-session ones omitted)."""
    datasets = DATASETS if datasets is None else datasets
    return {d: list(TRAIN_SESSIONS[d]) for d in datasets if d in TRAIN_SESSIONS}


def sessions_cli(datasets=None):
    """['bci_iv_2a=0train', 'bci_iv_2b=0train,1train,2train'] -- the format
    train_phase0.py's --sessions takes (parsed by sampler.parse_session_args)."""
    return [f"{d}={','.join(labels)}" for d, labels in sessions_dict(datasets).items()]


def _cache_session_counts(cache_dir, dataset):
    """{label: n_epochs} over every cached subject of one dataset. Reads only
    the small 'session' array of each .npz, never X."""
    ds_dir = os.path.join(cache_dir, dataset)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"No cache dir for '{dataset}' at {ds_dir}")
    counts = {}
    files = sorted(f for f in os.listdir(ds_dir)
                   if f.startswith("subject_") and f.endswith(".npz"))
    for f in files:
        # allow_pickle=True: the cache stores 'session' as an object array.
        with np.load(os.path.join(ds_dir, f), allow_pickle=True) as d:
            labels, n = np.unique(d["session"].astype(str), return_counts=True)
        for lab, c in zip(labels.tolist(), n.tolist()):
            counts[lab] = counts.get(lab, 0) + int(c)
    return counts, len(files)


def check_against_cache(cache_dir=CACHE_DIR, datasets=None):
    """Print what Phase 0 would train on vs hold out, per dataset. Returns True
    if the config is consistent with the cache (no errors; warnings allowed)."""
    datasets = DATASETS if datasets is None else datasets
    ok = True
    print(f"Cache: {cache_dir}\n")
    for d in datasets:
        try:
            counts, n_subj = _cache_session_counts(cache_dir, d)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            ok = False
            continue
        present = sorted(counts)
        print(f"{d}: {n_subj} cached subject(s), session labels in cache: "
              f"{ {k: counts[k] for k in present} }")

        if d in SINGLE_SESSION_DATASETS:
            print("   single-session dataset: all epochs used, no session list needed.\n")
            continue
        if d not in TRAIN_SESSIONS:
            print(f"   [ERROR] '{d}' is multi-session but has no entry in TRAIN_SESSIONS.\n")
            ok = False
            continue

        train = list(TRAIN_SESSIONS[d])
        missing = [l for l in train if l not in counts]
        held_out = [l for l in present if l not in train]
        if missing:
            print(f"   [ERROR] TRAIN_SESSIONS labels not found in the cache: {missing}")
            ok = False
        if not held_out:
            print("   [ERROR] no held-out session: Phase 0 would train on every session "
                  "(not valid for cross-session claims).")
            ok = False
        n_train = sum(counts[l] for l in train if l in counts)
        n_held = sum(counts[l] for l in held_out)
        print(f"   Phase 0 trains on {train}: {n_train} epochs")
        print(f"   Held out (never seen by Phase 0): {held_out}: {n_held} epochs")
        expected = EXPECTED_HELD_OUT.get(d)
        if expected is not None and sorted(expected) != sorted(held_out):
            print(f"   [WARNING] expected held-out {expected} but the cache gives "
                  f"{held_out}. Confirm which sessions are the test sessions for the "
                  f"release you cached, then fix TRAIN_SESSIONS.")
        print()
    print("CONFIG OK" if ok else "CONFIG HAS ERRORS -- fix before training")
    return ok


def build_command(datasets=None, cache_dir=CACHE_DIR, export_path="phase0_embeddings.npz"):
    datasets = DATASETS if datasets is None else datasets
    parts = ["python train_phase0.py",
             f"--datasets {' '.join(datasets)}",
             f"--cache_dir {cache_dir}"]
    sess = sessions_cli(datasets)
    if sess:
        parts.append(f"--sessions {' '.join(sess)}")
    if export_path:
        parts.append(f"--export_embeddings {export_path}")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description="Phase 0 session policy: check and command builder")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--cache_dir", default=CACHE_DIR)
    p.add_argument("--check", action="store_true",
                   help="Compare TRAIN_SESSIONS with the labels in the cache")
    p.add_argument("--print_cmd", action="store_true",
                   help="Print the full train_phase0.py command")
    p.add_argument("--cli", action="store_true",
                   help="Print only the --sessions arguments")
    p.add_argument("--export_path", default="phase0_embeddings.npz")
    args = p.parse_args()

    if not (args.check or args.print_cmd or args.cli):
        p.print_help()
        return 0
    status = 0
    if args.check and not check_against_cache(args.cache_dir, args.datasets):
        status = 1
    if args.cli:
        print(" ".join(sessions_cli(args.datasets)))
    if args.print_cmd:
        print(build_command(args.datasets, args.cache_dir, args.export_path))
    return status


if __name__ == "__main__":
    sys.exit(main())