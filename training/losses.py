"""Loss functions used for KDS-Former training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """CTC loss for gloss-sequence supervision."""

    def __init__(
        self,
        blank: int = 0,
        lambda_ctc: float = 1.0,
    ):
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(
        self,
        student_gloss_logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        actual_frames: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float]:
        batch_size, frames, _ = student_gloss_logits.shape
        log_probs = F.log_softmax(student_gloss_logits, dim=-1).permute(1, 0, 2)
        input_lengths = actual_frames if actual_frames is not None else torch.full(
            (batch_size,),
            frames,
            dtype=torch.long,
            device=log_probs.device,
        )

        loss_ctc = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        total_loss = self.lambda_ctc * loss_ctc
        return {
            "total": total_loss,
            "ctc": float(loss_ctc.detach().item()),
        }
