"""
losses.py

InfoNCE loss for DARWIN Phase 0 (spec v3 Section 5, carried unchanged into
the rev A / v3.6 specification):

    L_pretrain = L_InfoNCE with subject-balanced negative sampling.
    - Two augmented views per epoch (via augment() called twice).
    - Negatives are sampled such that the batch contains both:
        - Intra-subject positives: another view from the same subject.
        - Cross-subject negatives: clips from different subjects.
    - This explicitly prevents the encoder from exploiting subject
      identity as the sole discriminative feature.

CORRECTION (post early probe results -- read before touching this file):
An earlier version of this module excluded same-subject-different-clip
pairs from the negative pool entirely, so two clips from the same subject
were never pushed apart. That made subject identity a trivial shortcut. The
rev A / v3.6 specification names exactly this outcome as a risk: "Phase 0
learns a subject shortcut", detected by the G3 task probe being near chance,
with the fallback "fix augmentation, delay Phase 1".

WHAT THE SPEC ASKS FOR (rev A, section 9):
    - G3 is a TASK probe only: a linear probe on the frozen, PRE-PROJECTION
      64-d embeddings (encoder output, not the projection head) must reach
      chance + 5 percentage points on a subject-disjoint split
      (probe_task_decodability.py).
    - The subject-ID probe is a DIAGNOSTIC, never a gate
      (probe_subject_leakage.py). A collapsed subject probe (near chance) is
      a WARNING, because SSRS matches a new user to donor subjects by
      embedding similarity and needs some subject structure to exist. A very
      high subject probe on its own is informational: the concern is a high
      subject-ID probe COMBINED with a near-chance task probe.
    - The subject probe is plotted against the task probe for ablation 12
      (Pareto).
    So this loss does not try to drive subject-ID accuracy to chance. It only
    removes the trivial shortcut; whether the balance is right is judged by
    G3 plus the subject-ID and dataset-ID diagnostics.

Excluding same-subject-different-clip pairs from the loss meant every
single comparison this loss ever constructed reduced to "same subject"
(positive) vs. "different subject" (negative) -- nothing ever required
the encoder to distinguish between two DIFFERENT clips from the SAME
subject. Since raw EEG carries strong subject-specific low-level
statistics (skull geometry, electrode impedance, baseline spectral
power) that are far easier to exploit than subtle task-related
structure, this made subject-identity clustering a trivial, fully-
sufficient shortcut for satisfying the objective -- with zero
counter-pressure against it. A real training run confirmed this: 40%
task decodability (weak) alongside 97% subject-ID leakage (near-total),
i.e. the G3 failure signal.

FIX: same-subject-different-clip pairs are now treated as ordinary
negatives, same as cross-subject pairs (standard SimCLR-style InfoNCE,
with only self-similarity excluded from the denominator). This gives
the loss real pressure against subject clustering: the encoder must
now also distinguish between different trials from the same subject,
not just between different subjects.

NOTE ON WHAT THIS DOESN'T CHANGE: sampler.py's SubjectBalancedBatchSampler
still deliberately puts >=2 clips from the same subject into every batch
(per spec Section 3/7). That's still correct and still needed -- it's
what guarantees genuine intra-subject structure is present in every
batch for the loss to act on. The bug was in how this loss function
USED that structure (excluding it), not in the batch composition itself.

Deliberately NOT fixed via an adversarial subject classifier / gradient-
reversal layer, despite that being the standard alternative in the
domain-adaptation literature: DARWIN's own problem statement explicitly
rejects adversarial training's instability at small subject counts as
one of the things this architecture is meant to avoid. Staying within
plain contrastive-loss territory keeps this fix consistent with that.

INPUT ORDERING CONTRACT (matches sampler.py's SubjectBalancedBatchSampler):
Batch positions are ordered [subj_A_clip1, subj_A_clip2, subj_B_clip1,
subj_B_clip2, ...] -- i.e. grouped by subject, k_per_subject consecutive
positions per subject. subject_ids passed to info_nce_subject_balanced
must reflect this same ordering (one id per batch position, BEFORE the
augmented-view doubling that happens inside this loss function).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_subject_balanced(z1, z2, subject_ids, temperature=0.1):
    """
    z1: (B, D) -- embeddings from augmented view 1 of each anchor epoch
    z2: (B, D) -- embeddings from augmented view 2 of the SAME anchor
        epochs, same order as z1 (z1[i] and z2[i] are the positive pair)
    subject_ids: (B,) -- subject id (as an integer/hashable-turned-int
        tensor) for each anchor position. z1[i] and z2[i] share
        subject_ids[i] by construction, since they're two views of one
        epoch from one subject. NOTE: since the fix below, this function
        only shape-checks subject_ids (only self-similarity is masked);
        the argument is kept for the batch-ordering contract and for
        diagnostics.
    temperature: NOT specified by spec v3 -- 0.1 is the standard
        SimCLR/contrastive-learning default, used here as a starting
        point. Worth treating as a tunable hyperparameter if the G3 task
        probe (or the subject-ID / dataset-ID diagnostics) suggests the
        embedding space is too diffuse
        (raise temperature) or collapsing (lower it), rather than
        assuming 0.1 is correct for EEG data specifically.

    Positive pair: the other augmented view of the same anchor epoch.
    Negatives: every other position in the batch, INCLUDING same-
    subject-different-clip pairs (see module docstring's CORRECTION
    section for why this changed from an earlier, buggy version that
    excluded them). Only exact self-similarity is excluded.

    Returns: scalar loss (mean over all 2B anchor positions)
    """
    B = z1.shape[0]
    assert z2.shape[0] == B
    assert subject_ids.shape[0] == B

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # True positive for position i: the OTHER view of the same anchor.
    # View1 positions [0, B) pair with view2 positions [B, 2B) at the
    # same offset, and vice versa.
    pos_idx = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])  # (2B,)

    # Build the "exclude from denominator" mask.
    #
    # FIX (see module docstring "CORRECTION" section): only self-
    # similarity (i == i) is excluded, standard SimCLR practice.
    # Same-subject-different-clip pairs are now ordinary negatives --
    # NOT masked out. The previous version excluded them, which meant
    # every comparison this loss ever constructed reduced to "same
    # subject" (positive) vs. "different subject" (negative), making
    # subject-identity clustering a trivial, fully-sufficient solution
    # to the objective with zero counter-pressure against it. G3
    # needs task structure to survive, so same-subject-different-clip
    # pairs must be pushed apart just like cross-subject pairs --
    # otherwise nothing in the loss ever requires the encoder to
    # represent anything beyond subject identity.
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    exclude_mask = self_mask

    sim = sim.masked_fill(exclude_mask, float("-inf"))

    loss = F.cross_entropy(sim, pos_idx)
    return loss


class InfoNCELoss(nn.Module):
    """
    Thin nn.Module wrapper around info_nce_subject_balanced, for
    convenient use in a training loop (so temperature can live as a
    module attribute rather than being threaded through every call).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, subject_ids):
        return info_nce_subject_balanced(z1, z2, subject_ids, temperature=self.temperature)


if __name__ == "__main__":
    # Smoke tests -- verify correctness in isolation with synthetic
    # tensors BEFORE trusting this against real encoder output. Per
    # your team's own build practice: unit-test loss functions in
    # isolation to catch sign errors / wrong-axis bugs cheaply.
    torch.manual_seed(0)
    loss_fn = InfoNCELoss(temperature=0.1)

    B = 8
    D = 64
    # subject_ids matching sampler.py's grouped ordering: k=2 per
    # subject, e.g. [0,0,1,1,2,2,3,3] for P=4 subjects, K=2 each
    subject_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    # Test 1: identical positive pairs should give LOW loss
    z_base = torch.randn(B, D)
    loss_identical = loss_fn(z_base, z_base.clone(), subject_ids)
    print(f"Test 1 -- identical positive pairs: loss = {loss_identical.item():.4f} "
          f"(expect LOW, near 0)")

    # Test 2: completely random, unrelated pairs should give HIGHER loss
    z1_rand = torch.randn(B, D)
    z2_rand = torch.randn(B, D)
    loss_random = loss_fn(z1_rand, z2_rand, subject_ids)
    print(f"Test 2 -- random unrelated pairs:    loss = {loss_random.item():.4f} "
          f"(expect HIGHER than test 1)")
    assert loss_identical.item() < loss_random.item(), \
        "FAIL: identical pairs should have lower loss than random pairs"
    print("  PASS: identical < random")

    # Test 3 (rewritten post-fix): verify same-subject-different-clip
    # pairs are now treated SYMMETRICALLY with cross-subject pairs, not
    # specially excluded. Method: force anchor 0's view1 embedding to
    # look identical to another position's view1 embedding -- once with
    # a SAME-subject position (position 1, also subject 0), once with a
    # DIFFERENT-subject position (position 2, subject 1). Under the fix,
    # both should act as a "hard negative" for anchor 0 and raise the
    # loss versus baseline, by a comparable amount -- confirming
    # same-subject negatives are no longer masked out.
    z1_test = torch.randn(B, D)
    z2_test = torch.randn(B, D)
    loss_baseline = loss_fn(z1_test, z2_test, subject_ids)

    z1_same_subj_confound = z1_test.clone()
    z1_same_subj_confound[1] = z1_test[0].clone()  # subj 0's clip2 <- subj 0's clip1
    loss_same_subj = loss_fn(z1_same_subj_confound, z2_test, subject_ids)

    z1_cross_subj_confound = z1_test.clone()
    z1_cross_subj_confound[2] = z1_test[0].clone()  # subj 1's clip1 <- subj 0's clip1
    loss_cross_subj = loss_fn(z1_cross_subj_confound, z2_test, subject_ids)

    print(f"\nTest 3 -- symmetric negative treatment after the fix:")
    print(f"  baseline loss:              {loss_baseline.item():.4f}")
    print(f"  same-subject confound loss: {loss_same_subj.item():.4f} "
          f"(delta {loss_same_subj.item() - loss_baseline.item():+.4f})")
    print(f"  cross-subject confound loss:{loss_cross_subj.item():.4f} "
          f"(delta {loss_cross_subj.item() - loss_baseline.item():+.4f})")

    assert loss_same_subj.item() > loss_baseline.item(), (
        "FAIL: a same-subject-different-clip pair made highly similar to "
        "the anchor should now raise the loss (it's an ordinary negative) "
        "-- if this doesn't hold, same-subject pairs may still be masked."
    )
    assert loss_cross_subj.item() > loss_baseline.item(), (
        "sanity: a cross-subject confound should also raise the loss."
    )
    print("  PASS: both same-subject and cross-subject confounds raise the "
          "loss -- same-subject pairs are no longer masked out, and both "
          "kinds of negative are treated symmetrically as intended.")

    print("\nAll shape/sanity checks passed.")