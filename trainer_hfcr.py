from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import zlib

import os

import numpy
import torch
import torch.nn.functional as F

from evaluation import metrics
from modules.hfcr_reasoning import spatial_smoothness_l2
from utils import AverageMeter, get_iou


@dataclass
class LossWeights:
    # Matches HFCR.md: L_total = λ1 L_ce + λ2 L_his + λ3 L_fut + λ4 L_con
    ce: float = 1.0
    his: float = 1e-3
    fut: float = 1e-3
    con: float = 1e-3

    # Smoothness regularizer for future prediction.
    fut_reg: float = 1e-3


def _as_video_ids(v: Any, batch_size: int) -> Sequence[str]:
    """
    Normalizes "video" metadata into a list[str] with length B.
    """
    if v is None:
        return [f"sample_{i}" for i in range(batch_size)]
    if isinstance(v, str):
        return [v for _ in range(batch_size)]
    if isinstance(v, (list, tuple)) and len(v) == batch_size:
        return [str(x) for x in v]
    return [f"sample_{i}" for i in range(batch_size)]


def _video_ids_to_tensor(video_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    """
    Converts video id strings into a deterministic int64 tensor that can be scattered by DataParallel.
    """
    ids = [zlib.crc32(str(x).encode("utf-8")) & 0xFFFFFFFF for x in video_ids]
    return torch.tensor(ids, device=device, dtype=torch.long)


def _sanitize_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Ensures masks are [B, L, H, W] long tensors (foreground=1, background=0).
    """
    if masks.dim() == 5 and masks.size(2) == 1:
        masks = masks.squeeze(2)  # [B, L, H, W]
    if masks.dim() == 4:
        return masks.long()
    raise ValueError(f"Unexpected mask shape: {tuple(masks.shape)}")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_set: Optional[torch.utils.data.Dataset],
        save_name: str,
        save_step: int,
        val_step: int,
        loss_w: LossWeights = LossWeights(),
    ) -> None:
        self.model = model.cuda()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_set = val_set
        self.save_name = save_name
        self.save_step = int(save_step)
        self.val_step = int(val_step)
        self.loss_w = loss_w

        self.epoch = 1
        self.best_score = 0.0
        self.score = 0.0
        self.stats = {"loss": AverageMeter(), "iou": AverageMeter()}

    def train(self, max_epochs: int, path: str) -> None:
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.save_step == 0:
                print("saving checkpoint\n")
                self.save_checkpoint(folder_path=path)
            if self.score > self.best_score:
                print(f"new best checkpoint, after epoch {self.epoch}\n")
                self.save_checkpoint(alt_name="best", folder_path=path)
                self.best_score = self.score
        print("finished training!\n", flush=True)

    def train_epoch(self) -> None:
        # Train
        self.model.train()
        self.cycle_dataset(mode="train")

        # Val
        self.model.eval()
        if self.epoch % self.val_step == 0 and self.val_set is not None:
            with torch.no_grad():
                self.score = self.cycle_dataset(mode="val")

        # Reset meters
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def _compute_losses(
        self, out: Dict[str, Any], masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        out["scores"]: [B, L, 2, H, W]
        out["aux"]["his_logits"]: [B, L, 1, h, w]
        out["aux"]["fut_logits"]: [B, L, 1, h, w]
        """
        scores = out["scores"]
        aux = out.get("aux", {})

        b, l, _, h, w = scores.shape

        # Main segmentation loss (CE).
        loss_ce = torch.nn.CrossEntropyLoss()(
            scores.view(b * l, 2, h, w),
            masks.view(b * l, h, w),
        )

        # Historical loss: supervise history logits with current mask at feature scale.
        loss_his = scores.new_tensor(0.0)
        his_logits = aux.get("his_logits", None)
        if isinstance(his_logits, torch.Tensor) and l >= 2:
            # Skip t=0 (no history available).
            his_logits = his_logits[:, 1:]  # [B, L-1, 1, h16, w16]
            h16, w16 = his_logits.shape[-2:]
            tgt = F.interpolate(masks[:, 1:].float().unsqueeze(2), size=(h16, w16), mode="nearest").squeeze(2)
            loss_his = F.binary_cross_entropy_with_logits(
                his_logits.flatten(0, 1),
                tgt.flatten(0, 1).unsqueeze(1),
            )

        # Future loss: predict M_{t+1} from time t.
        loss_fut = scores.new_tensor(0.0)
        fut_logits = aux.get("fut_logits", None)
        if isinstance(fut_logits, torch.Tensor) and l >= 2:
            h16, w16 = fut_logits.shape[-2:]
            tgt_next = masks[:, 1:].float().unsqueeze(2)  # [B, L-1, 1, H, W]
            tgt_next = F.interpolate(tgt_next.flatten(0, 1), size=(h16, w16), mode="nearest")  # [(B*(L-1)),1,h16,w16]
            pred = fut_logits[:, :-1].flatten(0, 1)  # [(B*(L-1)),1,h16,w16]

            bce = F.binary_cross_entropy_with_logits(pred, tgt_next)
            reg = spatial_smoothness_l2(pred)
            loss_fut = bce + self.loss_w.fut_reg * reg

        # Contrastive loss (already computed in the model).
        loss_con = aux.get("con_loss", scores.new_tensor(0.0))
        if not isinstance(loss_con, torch.Tensor):
            loss_con = scores.new_tensor(0.0)

        loss_total = (
            self.loss_w.ce * loss_ce
            + self.loss_w.his * loss_his
            + self.loss_w.fut * loss_fut
            + self.loss_w.con * loss_con
        )
        losses = {"ce": loss_ce, "his": loss_his, "fut": loss_fut, "con": loss_con}
        return loss_total, losses

    def cycle_dataset(self, mode: str) -> Optional[float]:
        if mode == "train":
            for vos_data in self.train_loader:
                imgs = vos_data["imgs"].cuda(non_blocking=True)   # [B, L, 3, H, W]
                flows = vos_data["flows"].cuda(non_blocking=True)  # [B, L, 3, H, W]
                masks = _sanitize_masks(vos_data["masks"].cuda(non_blocking=True))  # [B, L, H, W]

                b, l, _, h, w = imgs.size()
                video_ids = _as_video_ids(vos_data.get("video", None), b)
                video_ids_t = _video_ids_to_tensor(video_ids, device=imgs.device)

                out = self.model(imgs, flows, video_ids=video_ids_t)
                loss, _ = self._compute_losses(out, masks)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # Metrics (IoU uses logits directly).
                self.stats["loss"].update(loss.detach().cpu().item(), b)
                iou = torch.mean(get_iou(out["scores"].view(b * l, 2, h, w), masks.view(b * l, h, w))[:, 1:])
                self.stats["iou"].update(iou.detach().cpu().item(), b)

            print(f"[ep{self.epoch:04d}] loss: {self.stats['loss'].avg:.5f}, iou: {self.stats['iou'].avg:.5f}")
            return None

        if mode == "val":
            metrics_res: Dict[str, list] = {"J": [], "F": []}
            for _video_name, video_parts in self.val_set.get_videos():
                for vos_data in video_parts:
                    imgs = vos_data["imgs"].cuda(non_blocking=True)   # [1, L, 3, H, W]
                    flows = vos_data["flows"].cuda(non_blocking=True)  # [1, L, 3, H, W]
                    masks = vos_data["masks"].cuda(non_blocking=True)  # [1, L, 1, H, W] or [1, L, H, W]

                    out = self.model(imgs, flows, video_ids=_video_ids_to_tensor([_video_name], device=imgs.device))
                    res_masks = out["masks"].squeeze(2)  # [1, L, H, W]

                    if masks.dim() == 5:
                        gt_masks = masks.squeeze(2)  # [1, L, H, W]
                    else:
                        gt_masks = masks  # [1, L, H, W]

                    # Keep consistent with the original evaluation: ignore the first/last frame if desired.
                    res_masks = res_masks[:, 1:-1]
                    gt_masks = gt_masks[:, 1:-1]

                    b, l, h, w = res_masks.shape
                    object_ids = numpy.unique(gt_masks.cpu()).tolist()
                    if 0 in object_ids:
                        object_ids.remove(0)
                    if len(object_ids) == 0:
                        continue

                    all_res_masks = numpy.zeros((len(object_ids), l, h, w))
                    all_gt_masks = numpy.zeros((len(object_ids), l, h, w))
                    for k in object_ids:
                        res_masks_k = res_masks.detach().cpu().numpy().copy()
                        res_masks_k[res_masks_k != k] = 0
                        res_masks_k[res_masks_k != 0] = 1
                        all_res_masks[k - 1] = res_masks_k[0]

                        gt_masks_k = gt_masks.detach().cpu().numpy().copy()
                        gt_masks_k[gt_masks_k != k] = 0
                        gt_masks_k[gt_masks_k != 0] = 1
                        all_gt_masks[k - 1] = gt_masks_k[0]

                    j_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    f_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    for i in range(all_gt_masks.shape[0]):
                        j_metrics_res[i] = metrics.db_eval_iou(all_gt_masks[i], all_res_masks[i])
                        f_metrics_res[i] = metrics.db_eval_boundary(all_gt_masks[i], all_res_masks[i])
                        [jm, _, _] = metrics.db_statistics(j_metrics_res[i])
                        [fm, _, _] = metrics.db_statistics(f_metrics_res[i])
                        metrics_res["J"].append(jm)
                        metrics_res["F"].append(fm)

            j, f = metrics_res["J"], metrics_res["F"]
            final_mean = (numpy.mean(j) + numpy.mean(f)) / 2.0
            print(f"[ep{self.epoch:04d}] J&F score: {final_mean:.5f}\n")
            return float(final_mean)

        raise ValueError(f"Unknown mode: {mode}")

    def save_checkpoint(self, alt_name: Optional[str] = None, folder_path: str = "") -> None:
        save_dir = f"weights/{folder_path}" if folder_path else "weights"
        os.makedirs(save_dir, exist_ok=True)

        if alt_name is not None:
            file_path = f"{save_dir}/{self.save_name}_{alt_name}.pth"
        else:
            file_path = f"{save_dir}/{self.save_name}_{self.epoch:04d}.pth"

        # DataParallel wraps the model in `.module`.
        state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        torch.save(state, file_path)
