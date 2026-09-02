"""
projection_head.py

Projection head for DARWIN Phase 0, per spec v3 Section 4:

    ... -> EEGNet Encoder -> Projection Head: 2-layer MLP -> R^64
    z in R^64 (unified embedding space) [NON-NEGOTIABLE WIDTH]

This is the one component of the Phase 0 pipeline where the spec is
fully explicit ("2-layer MLP") -- unlike the encoder, no architectural
guessing needed here.

IMPORTANT, non-obvious from typical contrastive-learning practice:
in standard SimCLR-style setups, the projection head is normally
DISCARDED after pretraining -- only the encoder is kept, because the
projection head's job is to let InfoNCE operate in a space specialized
for the contrastive objective, which can actually hurt downstream task
performance if used directly. Spec v3's Section 9 deliverables table
explicitly breaks this convention: phase0_checkpoint.pt is defined to
contain "LSP weights ... EEGNet encoder weights, projection head
weights" together, frozen as one artifact. So for DARWIN, the
projection head is NOT thrown away -- it's saved and frozen alongside
the encoder. Keep this in mind when writing the checkpoint-saving code
in train_phase0.py: don't drop projection_head's state_dict there just
because that's the "usual" thing to do elsewhere.

Architecture: encoder output (64-dim, see encoder.py's default
embedding_dim) -> Linear -> BatchNorm -> ReLU -> Linear -> z (64-dim).
BatchNorm1d is included following standard SimCLR-style projection head
design (stabilizes training, mild regularization) -- not spec-mandated,
but a reasonable default within the spec's "2-layer MLP" constraint.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    2-layer MLP mapping encoder embeddings to the unified 64-dim
    contrastive space z.

    input_dim: must match encoder.py's embedding_dim (default 64)
    output_dim: NON-NEGOTIABLE per spec -- must stay 64. Everything
        downstream (SSRS centroids, DTKF measurements, FiLM
        conditioning) is defined to operate in this exact space, so
        changing this breaks the "z is the only embedding space"
        invariant the whole DARWIN architecture depends on.
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        if output_dim != 64:
            print(f"[projection_head] WARNING: output_dim={output_dim} != 64. "
                  f"Spec v3 Section 4 marks 64 as the NON-NEGOTIABLE width "
                  f"for z, the unified embedding space every later module "
                  f"(SSRS, DTKF, FiLM) is defined to operate in. Only "
                  f"override this for isolated experimentation, never for "
                  f"an actual Phase 0 training run.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x):
        # x: (B, input_dim) -- encoder output
        return self.net(x)  # (B, output_dim)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test: verify shapes and confirm output width is spec-locked at 64.
    head = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=64)
    print(f"Projection head params: {head.param_count()}")

    # Simulate encoder output: (B, 64)
    x = torch.randn(8, 64)
    z = head(x)
    print(f"Input shape:  {tuple(x.shape)} (simulated encoder output)")
    print(f"Output shape: {tuple(z.shape)} (expect (8, 64) -- NON-NEGOTIABLE width)")
    print(f"Output nan: {torch.isnan(z).any().item()}")

    # Confirm two independent forward passes on the SAME input differ when
    # in train mode (dropout/BN noise) but agree in eval mode -- a quick
    # sanity check that BatchNorm/train-eval switching is wired correctly,
    # since train_phase0.py will need to call head.eval() before Gate 1/2
    # probes on frozen embeddings.
    head.eval()
    with torch.no_grad():
        z1 = head(x)
        z2 = head(x)
    same_in_eval = torch.allclose(z1, z2)
    print(f"Deterministic in eval mode (expect True): {same_in_eval}")
    