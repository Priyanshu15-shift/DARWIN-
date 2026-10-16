"""
sampler.py

Phase 0 pair sampler for DARWIN self-supervised pretraining.

Positive pair definition (Option A - SimCLR style):
    Each item returned is a single cached epoch. Two independently-
    augmented VIEWS of that same epoch form the positive pair. The
    augmentation itself happens in the training loop (train_phase0.py),
    not here — this module's only job is to serve the raw cached epoch
    reliably and efficiently.

Negatives are implicit: every other epoch in the same batch acts as a
negative for InfoNCE (handled in losses.py, not here).

SESSION FILTER (DARWIN spec rev A, R1): Phase 0 must never see the held-out
test session(s) of any subject. PairDataset takes `sessions`, a dict
{dataset_name: [session labels to KEEP]}. Labels are compared as strings
against the 'session' array stored in each cached .npz (run
inspect_cache.py to see the exact labels per dataset). A dataset missing
from the dict keeps ALL its sessions; an unmatched label raises, listing the
labels that do exist, so a typo can never silently train on everything.
Epoch indices in dataset.index are positions into the ORIGINAL cached X,
so filtering does not change how __getitem__ or train_phase0.py use them.

Cache format (matches full_pool_preprocess.py output, verified against
actual files on disk):
    - One file per subject: `{cache_dir}/{dataset_name}/subject_{id}.npz`
      e.g. cache/v2_bandpass0.5-48_car/bci_iv_2a/subject_1.npz -- see
      cache_config.py for the current CACHE_DIR; cache_dir here is
      caller-supplied (PairDataset takes no default), so pass
      cache_config.CACHE_DIR explicitly rather than hardcoding a path.
    - dataset_name is snake_case: bci_iv_2a, bci_iv_2b, physionet, openbmi
      (Phase 0 currently pools the first three; openbmi is deferred until
      its download completes)
    - Each .npz holds multiple arrays. Phase 0 training reads:
        - 'X': shape (n_epochs, n_channels, n_timepoints), float32
        - 'session': per-epoch session label, read once at indexing time
          for the session filter (spec rev A, R1)
      Read only on the evaluation side (get_meta, the embedding export hook
      in train_phase0.py, and the probes), never used to train Phase 0:
        - 'y' (task label) and, in caches that have them, 'run',
          'trial_idx', 'order_idx' (older caches lack the last three;
          get_meta then returns ""/-1)
      Present in the file but ignored here: 'subject', 'pct_clipped_per_epoch',
      'pre_norm_median', 'pre_norm_mad', 'pipeline_version'.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_session_args(pairs):
    """
    Turn CLI-style strings into a sessions dict, e.g.
        ["bci_iv_2a=0train", "bci_iv_2b=0train,1train,2train"]
        -> {"bci_iv_2a": ["0train"], "bci_iv_2b": ["0train","1train","2train"]}
    Returns None for None/empty input (= keep every session).
    """
    if not pairs:
        return None
    out = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected DATASET=label[,label...], got '{item}'")
        ds, labels = item.split("=", 1)
        out[ds.strip()] = [l.strip() for l in labels.split(",") if l.strip()]
    return out


class PairDataset(Dataset):
    """
    Wraps cached EEG epochs from one or more datasets and serves single
    epochs for Phase 0 contrastive pretraining. Augmentation is applied
    twice downstream (in the training loop) to form the positive pair.

    Args:
        cache_dir: root directory containing cached .npy files. No
            default here deliberately -- pass cache_config.CACHE_DIR
            explicitly at the call site so this always tracks the
            current pipeline version rather than silently defaulting
            to a stale or nonexistent path.
        datasets: list of dataset names to include, e.g.
                   ["bci_iv_2a", "physionet"]. Must match the snake_case
                   subfolder names under cache_dir (bci_iv_2a, bci_iv_2b,
                   physionet, openbmi).
        subjects: optional dict {dataset_name: [subject_ids]} to restrict
                   which subjects are pulled in (useful while a dataset's
                   caching is still in progress, e.g. OpenBMI, which is
                   deferred -- only pull subjects that are actually done --
                   and for the subject-disjoint splits of the G3 task probe).
        sessions: optional dict {dataset_name: [session labels to keep]}.
                   See the SESSION FILTER note in the module docstring.
                   None keeps every session of every dataset.
        dtype: torch dtype for returned tensors.
    """

    def __init__(self, cache_dir, datasets, subjects=None, sessions=None,
                 dtype=torch.float32):
        self.cache_dir = cache_dir
        self.datasets = datasets
        self.dtype = dtype
        self.sessions = sessions
        self._summary = {}  # {dataset: kept/dropped session labels + counts}
        self._meta_cache = {}
        self.index = []  # list of (dataset_name, subject_id, epoch_idx)
        self._cache = {}  # (dataset_name, subject_id) -> np.ndarray, lazy-loaded

        self._build_index(subjects)

    def _build_index(self, subjects):
        for ds in self.datasets:
            ds_dir = os.path.join(self.cache_dir, ds)
            if not os.path.isdir(ds_dir):
                raise FileNotFoundError(
                    f"Expected cache dir for dataset '{ds}' at {ds_dir}, "
                    f"but it doesn't exist. Check cache_dir / dataset name "
                    f"(and that it matches the current pipeline version in "
                    f"cache_config.py -- a stale or wrong cache_dir is the "
                    f"most common cause of this error)."
                )

            # Files are named subject_{id}.npz — extract the numeric/string id
            available_subjects = sorted(
                f[len("subject_"):-4]
                for f in os.listdir(ds_dir)
                if f.startswith("subject_") and f.endswith(".npz")
            )

            if subjects is not None and ds in subjects:
                wanted = set(str(s) for s in subjects[ds])
                available_subjects = [s for s in available_subjects if s in wanted]

            if len(available_subjects) == 0:
                print(f"[sampler] WARNING: no cached subjects found for '{ds}' "
                      f"(after filtering). Skipping.")
                continue

            keep_labels = None
            if self.sessions is not None and ds in self.sessions:
                keep_labels = set(str(l) for l in self.sessions[ds])

            seen_labels = set()
            kept_seen = set()
            n_kept, n_total = 0, 0
            n_subj_used = 0
            for subj in available_subjects:
                path = os.path.join(ds_dir, f"subject_{subj}.npz")
                # Only the small 'session' array is read here -- X stays
                # compressed on disk until a subject is actually loaded.
                sess = self._read_sessions(path)
                seen_labels |= set(sess.tolist())
                n_total += len(sess)
                if keep_labels is None:
                    keep_pos = range(len(sess))
                else:
                    keep_pos = [i for i, l in enumerate(sess) if l in keep_labels]
                if len(keep_pos) > 0:
                    n_subj_used += 1
                for i in keep_pos:
                    self.index.append((ds, subj, int(i)))
                    kept_seen.add(str(sess[i]))
                n_kept += len(keep_pos)

            self._summary[ds] = {
                "requested": None if keep_labels is None else sorted(keep_labels),
                "kept_labels": sorted(kept_seen),
                "dropped_labels": sorted(seen_labels - kept_seen),
                "n_epochs": int(n_kept),
                "n_epochs_total": int(n_total),
                "n_subjects": int(n_subj_used),
            }

            if keep_labels is not None:
                missing = keep_labels - seen_labels
                if missing:
                    raise ValueError(
                        f"[sampler] session label(s) {sorted(missing)} requested "
                        f"for '{ds}' but not found in the cache. Labels present: "
                        f"{sorted(seen_labels)}."
                    )
                print(f"[sampler] {ds}: sessions {sorted(keep_labels)} kept -> "
                      f"{n_kept}/{n_total} epochs, {n_subj_used} subjects "
                      f"(dropped labels: {sorted(seen_labels - keep_labels)})")
            else:
                print(f"[sampler] {ds}: all sessions kept ({sorted(seen_labels)}) "
                      f"-> {n_kept} epochs, {n_subj_used} subjects")

        if len(self.index) == 0:
            raise RuntimeError(
                "PairDataset index is empty. No cached data found for the "
                "requested datasets/subjects."
            )

        print(f"[sampler] total epochs indexed: {len(self.index)}")

    @staticmethod
    def _read_sessions(path):
        """Per-epoch session labels as a string array. Requires the 'session'
        key in each cached .npz (full_pool_preprocess.py output; stored since
        cache v2). Run inspect_cache.py to confirm the exact labels per
        dataset. ('run', 'trial_idx' and 'order_idx' exist only in newer
        caches; see get_meta.)"""
        # allow_pickle=True ONLY for this small metadata array: the cache
        # stores 'session' as an object array (strings), which numpy refuses to
        # read otherwise. Safe for your own cache files; 'X' is still loaded
        # with allow_pickle=False in _load_subject.
        with np.load(path, allow_pickle=True) as d:
            return d["session"].astype(str)

    def _load_subject(self, dataset_name, subject_id):
        key = (dataset_name, subject_id)
        if key not in self._cache:
            path = os.path.join(self.cache_dir, dataset_name, f"subject_{subject_id}.npz")
            with np.load(path, allow_pickle=False) as d:
                # Cache only the 'X' array in memory, not the whole NpzFile
                # (which would keep a file handle open and include y/session/etc
                # we don't need here).
                self._cache[key] = d["X"]
        return self._cache[key]

    def session_summary(self):
        """{dataset: {requested, kept_labels, dropped_labels, n_epochs,
        n_epochs_total, n_subjects}} -- what Phase 0 actually saw. Saved
        into the checkpoint by train_phase0.py."""
        return self._summary

    _META_KEYS = ("session", "y", "run", "trial_idx", "order_idx")

    def _load_meta(self, dataset_name, subject_id):
        key = (dataset_name, subject_id)
        if key not in self._meta_cache:
            path = os.path.join(self.cache_dir, dataset_name, f"subject_{subject_id}.npz")
            # allow_pickle=True: metadata arrays (session/run) may be object
            # arrays; see _read_sessions. Only these keys are read here.
            with np.load(path, allow_pickle=True) as d:
                self._meta_cache[key] = {
                    k: (d[k] if k in d.files else None) for k in self._META_KEYS
                }
        return self._meta_cache[key]

    def get_meta(self, positions):
        """
        Per-epoch metadata for the given positions, as numpy arrays:
        session, y, run (strings) and trial_idx, order_idx (ints; -1 if the
        cache predates v3). Used by the embedding export hook.
        """
        out = {k: [] for k in self._META_KEYS}
        for p in positions:
            ds, subj, i = self.index[p]
            m = self._load_meta(ds, subj)
            for k in self._META_KEYS:
                if m[k] is None:
                    out[k].append(-1 if k in ("trial_idx", "order_idx") else "")
                else:
                    out[k].append(m[k][i])
        res = {}
        for k, v in out.items():
            arr = np.array(v)
            res[k] = arr.astype(str) if k in ("session", "y", "run") else arr.astype(np.int64)
        return res

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_name, subject_id, epoch_idx = self.index[idx]
        arr = self._load_subject(dataset_name, subject_id)
        epoch = arr[epoch_idx]  # shape: (n_channels, n_timepoints)
        return torch.tensor(epoch, dtype=self.dtype)

    def subject_groups(self):
        """
        Returns {(dataset_name, subject_id): [index_positions]} — the
        positions in self.index belonging to each subject. Keyed by
        (dataset_name, subject_id) rather than just subject_id, since
        e.g. "subject_1" in bci_iv_2a and "subject_1" in physionet are
        different people and must never be pooled together.
        """
        groups = {}
        for pos, (ds, subj, _epoch_idx) in enumerate(self.index):
            key = (ds, subj)
            groups.setdefault(key, []).append(pos)
        return groups


class SubjectBalancedBatchSampler(torch.utils.data.Sampler):
    """
    P x K batch sampler satisfying spec v3 Section 3 & 7: every batch must
    contain >=2 clips from the same subject as the anchor (for
    subject-balanced negatives), with cross-subject negatives filling the
    remainder.

    Implementation: each batch is composed of P subjects x K clips per
    subject (K=2 by default, matching the spec's ">=2 clips" language).
    This satisfies the constraint by construction rather than by chance —
    every subject present in a batch contributes >=2 clips, guaranteeing
    intra-subject positive pairs exist, while having P>1 distinct
    subjects per batch supplies the cross-subject negatives InfoNCE needs.

    This also keeps some subject structure learnable in the embedding space,
    which SSRS donor matching needs (rev A section 9). The subject-ID linear
    probe is a DIAGNOSTIC, not a gate: the Phase 0 gate is G3, the task probe.

    NOTE (pooled runs): subjects are drawn uniformly over (dataset, subject)
    keys, so with bci_iv_2a + bci_iv_2b + physionet pooled, physionet's 109
    subjects dominate batch composition (roughly 2 subjects each from the two
    9-subject datasets per batch of 32). train_phase0.py routes each dataset's
    clips through its own LSP and a separate encoder sub-batch, so the small
    datasets see very small sub-batches.

    If a subject has fewer than K clips available (shouldn't happen with
    real EEG data, but defensive), that subject is skipped for that draw
    rather than sampling with replacement — sampling the same exact clip
    twice would make the "positive pair" trivially identical pre-
    augmentation, which is still valid but worth being an explicit choice
    rather than an accident.
    """

    def __init__(self, dataset, batch_size=64, k_per_subject=2, batches_per_epoch=None, seed=None):
        if batch_size % k_per_subject != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"k_per_subject ({k_per_subject}) for even P x K composition."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.k_per_subject = k_per_subject
        self.p_per_batch = batch_size // k_per_subject  # number of distinct subjects per batch

        self.groups = dataset.subject_groups()  # {(ds, subj): [positions]}
        self.group_keys = list(self.groups.keys())

        eligible = [k for k in self.group_keys if len(self.groups[k]) >= k_per_subject]
        ineligible = [k for k in self.group_keys if len(self.groups[k]) < k_per_subject]
        if ineligible:
            print(f"[sampler] WARNING: {len(ineligible)} subject(s) have fewer than "
                  f"{k_per_subject} clips and will never be sampled by the balanced "
                  f"batch sampler: {ineligible}")
        self.eligible_keys = eligible

        if len(self.eligible_keys) < self.p_per_batch:
            raise ValueError(
                f"Need at least {self.p_per_batch} subjects with >={k_per_subject} "
                f"clips each to form one P x K batch (batch_size={batch_size}, "
                f"k_per_subject={k_per_subject}), but only "
                f"{len(self.eligible_keys)} eligible subjects are available. "
                f"Reduce batch_size, reduce k_per_subject, or add more cached "
                f"subjects/datasets."
            )

        # Default: one pass roughly covering the dataset once per "epoch"
        if batches_per_epoch is None:
            total_eligible_clips = sum(len(self.groups[k]) for k in self.eligible_keys)
            batches_per_epoch = max(1, total_eligible_clips // batch_size)
        self.batches_per_epoch = batches_per_epoch

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            # Sample P distinct subjects for this batch
            perm = torch.randperm(len(self.eligible_keys), generator=self.generator)
            chosen_subject_idxs = perm[:self.p_per_batch].tolist()

            batch_positions = []
            for si in chosen_subject_idxs:
                key = self.eligible_keys[si]
                positions = self.groups[key]
                # Sample K positions from this subject, without replacement
                local_perm = torch.randperm(len(positions), generator=self.generator)
                chosen = [positions[i] for i in local_perm[:self.k_per_subject].tolist()]
                batch_positions.extend(chosen)

            yield batch_positions


def make_phase0_dataloader(cache_dir, datasets, subjects=None, sessions=None,
                             batch_size=64, k_per_subject=2,
                             batches_per_epoch=None, num_workers=4, seed=None):
    """
    Convenience factory implementing spec v3's required batch composition:
    every batch contains >=k_per_subject clips from each of several
    subjects (P x K sampling), guaranteeing intra-subject positives and
    cross-subject negatives per Section 3 & 7.

    Returns batches of shape (batch_size, n_channels, n_timepoints) — raw,
    unaugmented epochs, ordered as [subj_A_clip1, subj_A_clip2,
    subj_B_clip1, subj_B_clip2, ...]. Augmentation (twice, to form
    positive pairs) happens in the training loop, not here. The training
    loop's InfoNCE / negative-sampling logic needs the subject labels to
    know which positions in the batch share a subject — expose them via
    dataset.index if not already tracked elsewhere.

    sessions: optional {dataset: [session labels to keep]}, passed to
    PairDataset (see the SESSION FILTER note in the module docstring).
    Phase 0 must never see a held-out test session (spec rev A, R1).
    """
    dataset = PairDataset(cache_dir=cache_dir, datasets=datasets,
                          subjects=subjects, sessions=sessions)
    batch_sampler = SubjectBalancedBatchSampler(
        dataset,
        batch_size=batch_size,
        k_per_subject=k_per_subject,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Quick smoke test — run this file directly to sanity check indexing
    # and subject-balanced batch composition.
    from cache_config import CACHE_DIR  # was hardcoded "./cache" here --
                                          # that pointed at the pre-fix,
                                          # now-stale flat cache path and
                                          # would have silently smoke-
                                          # tested against old data.
    DATASETS = ["bci_iv_2a"]  # fully cached (9 subjects) — good first target

    # NOTE: batch_size here is small (16) because bci_iv_2a only has 9
    # subjects and k_per_subject=2 needs batch_size/2 distinct subjects
    # per batch (8 <= 9, OK). Once physionet is mixed in (openbmi is
    # deferred), batch_size can go up to the spec's >=64 — with more
    # subjects available, larger P x K batches become possible.
    loader = make_phase0_dataloader(
        cache_dir=CACHE_DIR,
        datasets=DATASETS,
        subjects=None,  # None = use all available cached subjects
        batch_size=16,
        k_per_subject=2,
        num_workers=0,  # 0 for the smoke test, bump up for real training
        seed=0,
    )

    dataset = loader.dataset
    batch_sampler = loader.batch_sampler

    # Pull ONE batch of positions directly from the sampler, then build
    # the tensor batch from those SAME positions — going through
    # loader/batch_sampler as two independent iterators would desync
    # them (both draw from the same underlying random generator), so we
    # deliberately derive both from a single draw here.
    batch_positions = next(iter(batch_sampler))
    batch = torch.stack([dataset[p] for p in batch_positions], dim=0)

    print(f"\nBatch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Batch mean/std: {batch.mean().item():.4f} / {batch.std().item():.4f}")

    # Verify subject-balanced composition: every subject in this batch
    # should appear exactly k_per_subject times.
    batch_subjects = [dataset.index[p][:2] for p in batch_positions]  # (ds, subj) per slot
    from collections import Counter
    counts = Counter(batch_subjects)
    print(f"\nSubject composition check (expect every count == 2):")
    for key, count in counts.items():
        print(f"  {key}: {count} clips")
    all_balanced = all(c == 2 for c in counts.values())
    print(f"Batch composition valid (every subject appears exactly "
          f"k_per_subject=2 times): {all_balanced}")