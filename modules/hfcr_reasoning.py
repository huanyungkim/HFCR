"""
HFCR reasoning modules (Historical / Future / Cross-Video).

This file intentionally contains only PyTorch / torchvision compatible code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossVideoSamplingConfig:
    """Controls token sampling for cross-video attention."""

    num_tokens: int = 64
    exclude_self: bool = True


def _global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W] -> [B, C]
    return F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)


class HistoricalBacktracking(nn.Module):
    """
    Historical backtracking with a lightweight temporal attention and a learnable EMA update.

    Shapes:
        - cur:  [B, C, H, W]
        - past: Sequence[[B, C, H, W]] (most-recent last)
    """

    def __init__(self, channels: int, max_history: int = 5, pool: str = "avg") -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_history = int(max_history)
        self.pool = pool

        # Query/key projection for temporal attention (global descriptors).
        self.q_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.k_proj = nn.Linear(self.channels, self.channels, bias=False)

        # Lightweight transform T(F_{t-k}, F_t) -> gate in [0, 1].
        self.gate = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable decay factor gamma in (0, 1).
        self._gamma_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ≈ 0.88

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self._gamma_logit)

    def forward(
        self,
        cur: torch.Tensor,
        past: Sequence[torch.Tensor],
        prev_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            E_his:        [B, C, H, W]
            memory_next:  [B, C, 1, 1]
        """
        if len(past) == 0:
            e_his = torch.zeros_like(cur)
            pooled = F.adaptive_avg_pool2d(cur, (1, 1))
            memory_next = pooled if prev_memory is None else prev_memory
            return e_his, memory_next

        # Only keep the last K frames.
        past = past[-self.max_history :]

        # Temporal attention over past frames using global descriptors.
        q = self.q_proj(_global_avg_pool(cur))  # [B, C]
        keys = torch.stack([self.k_proj(_global_avg_pool(p)) for p in past], dim=1)  # [B, K, C]
        attn = torch.einsum("bc,bkc->bk", q, keys) / math.sqrt(self.channels)  # [B, K]
        alpha = F.softmax(attn, dim=1)  # [B, K]

        # Embedding: sum_k alpha_{t,k} * T(past_k, cur) ⊙ past_k
        e_his = 0.0
        for k, p in enumerate(past):
            gate = self.gate(torch.cat([p, cur], dim=1))  # [B, C, H, W]
            e_his = e_his + alpha[:, k].view(-1, 1, 1, 1) * (gate * p)
        e_his = e_his  # [B, C, H, W]

        pooled = F.adaptive_avg_pool2d(e_his, (1, 1))  # [B, C, 1, 1]
        if prev_memory is None:
            prev_memory = torch.zeros_like(pooled)

        g = self.gamma.to(dtype=pooled.dtype, device=pooled.device)
        memory_next = g * prev_memory + (1.0 - g) * pooled  # [B, C, 1, 1]
        return e_his, memory_next


class FutureInference(nn.Module):
    """
    Predicts the next-frame foreground mask (logits) given current features, mask, and flow.

    Inputs:
        feat_t:  [B, C, H, W]
        mask_t:  [B, 1, H, W] (probability or logits - treated as a feature)
        flow_t:  [B, Cf, H, W]
    """

    def __init__(self, feat_channels: int, flow_channels: int = 3, hidden: int = 128) -> None:
        super().__init__()
        in_ch = int(feat_channels) + 1 + int(flow_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat_t: torch.Tensor, mask_t: torch.Tensor, flow_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_t, mask_t, flow_t], dim=1)  # [B, C+1+Cf, H, W]
        return self.net(x)  # [B, 1, H, W] (logits)


def spatial_smoothness_l2(x: torch.Tensor) -> torch.Tensor:
    """
    L2 penalty on spatial gradients for smoothness.

    x: [B, 1, H, W] (logits or probabilities)
    """
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dx.square().mean() + dy.square().mean())


class CrossVideoReasoning(nn.Module):
    """
    Cross-video attention with token sampling + an optional InfoNCE-style contrastive loss.

    Notes:
        - This is designed to be lightweight and batch-friendly.
        - When video_ids are missing or no positives exist in the batch, the contrastive loss is 0.
    """

    def __init__(
        self,
        channels: int,
        attn_dim: int = 128,
        sampling: CrossVideoSamplingConfig = CrossVideoSamplingConfig(),
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.attn_dim = int(attn_dim)
        self.sampling = sampling
        self.temperature = float(temperature)

        self.q = nn.Linear(self.channels, self.attn_dim, bias=False)
        self.k = nn.Linear(self.channels, self.attn_dim, bias=False)
        self.v = nn.Linear(self.channels, self.attn_dim, bias=False)
        self.out = nn.Linear(self.attn_dim, self.channels, bias=False)

    def _sample_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> tokens: [B, T, C]
        b, c, h, w = x.shape
        t = min(self.sampling.num_tokens, h * w)
        flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        # Random sampling per-sample keeps compute bounded for large batches.
        idx = torch.randint(0, h * w, size=(b, t), device=x.device)
        idx = idx.unsqueeze(-1).expand(-1, -1, c)  # [B, T, C]
        return flat.gather(dim=1, index=idx)  # [B, T, C]

    def forward(
        self,
        feat: torch.Tensor,
        video_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: [B, C, H, W]
            video_ids: [B] int64 identifiers. Same id => positive pairs for contrastive loss.

        Returns:
            context: [B, C, 1, 1] (broadcast-friendly)
            con_loss: scalar tensor
        """
        b, c, h, w = feat.shape
        tokens = self._sample_tokens(feat)  # [B, T, C]

        q = self.q(tokens)  # [B, T, D]
        k = self.k(tokens)  # [B, T, D]
        v = self.v(tokens)  # [B, T, D]

        # Flatten keys/values for a single attention pool.
        k_all = k.reshape(b * k.size(1), -1)  # [B*T, D]
        v_all = v.reshape(b * v.size(1), -1)  # [B*T, D]

        # Attention: each sample attends to all sampled tokens (optionally excluding itself).
        context_tokens = []
        for i in range(b):
            qi = q[i]  # [T, D]
            logits = qi @ k_all.t() / math.sqrt(k_all.size(1))  # [T, B*T]

            if self.sampling.exclude_self:
                start = i * q.size(1)
                end = (i + 1) * q.size(1)
                logits[:, start:end] = -float("inf")

            attn = F.softmax(logits, dim=-1)  # [T, B*T]
            ctx = attn @ v_all  # [T, D]
            context_tokens.append(ctx.mean(dim=0))  # [D]

        ctx = torch.stack(context_tokens, dim=0)  # [B, D]
        ctx = self.out(ctx).view(b, c, 1, 1)  # [B, C, 1, 1]

        # InfoNCE-style contrastive loss on global descriptors.
        con_loss = feat.new_tensor(0.0)
        if video_ids is not None and b >= 2:
            if video_ids.dim() != 1 or video_ids.numel() != b:
                raise ValueError(f"video_ids must be shape [B], got {tuple(video_ids.shape)}")
            desc = F.normalize(_global_avg_pool(feat), dim=1)  # [B, C]
            sim = (desc @ desc.t()) / self.temperature  # [B, B]

            # Mask out self-similarities.
            sim.fill_diagonal_(-float("inf"))

            # positives: same video id; negatives: different
            labels = video_ids.to(device=feat.device)
            pos_mask = labels.view(-1, 1).eq(labels.view(1, -1))  # [B, B]
            pos_mask.fill_diagonal_(False)
            has_pos = pos_mask.any(dim=1)  # [B]
            if has_pos.any():
                # For each anchor, pick the hardest positive (max sim) as the positive logit.
                pos_logits = sim.masked_fill(~pos_mask, -float("inf")).max(dim=1).values  # [B]
                # Denominator over all non-self pairs.
                con_loss = (-pos_logits[has_pos] + torch.logsumexp(sim[has_pos], dim=1)).mean()

        return ctx, con_loss
