from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from convnextv2.build import build_convnextv2_model
from modules.FreqFuse import FreqFusion
from modules.hfcr_reasoning import CrossVideoReasoning, FutureInference, HistoricalBacktracking

# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.ver = ver

        # ResNet-101 backbone
        if ver == 'rn101':
            backbone = tv.models.resnet101(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ConvNeXtV2_tiny backbone
        if ver == 'convnextv2':
            self.backbone = build_convnextv2_model(model_type='convnextv2_tiny.fcmae_ft_in22k_in1k')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, img):

        # ResNet-101 backbone
        if self.ver == 'rn101':
            x = (img - self.mean) / self.std
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            s4 = x
            x = self.layer2(x)
            s8 = x
            x = self.layer3(x)
            s16 = x
            x = self.layer4(x)
            s32 = x
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # ConvNeXtV2_tiny backbone
        if self.ver == 'convnextv2':
            x = (img - self.mean) / self.std
            x = self.backbone(x)
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}


# decoding module
class Decoder(nn.Module):
    def __init__(self, ver):
        super().__init__()

        # ResNet-101 backbone
        if ver == 'rn101':
            self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.ff1 = FreqFusion(256, 256)
            self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.ff2 = FreqFusion(256, 256)
            self.conv3 = ConvRelu(512, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.ff3 = FreqFusion(256, 256)
            self.conv4 = ConvRelu(256, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        # # ConvNeXtV2_tiny backbone
        if ver == 'convnextv2':
            self.conv1 = ConvRelu(768, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.ff1 = FreqFusion(256, 256)
            self.conv2 = ConvRelu(384, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.ff2 = FreqFusion(256, 256)
            self.conv3 = ConvRelu(192, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.ff3 = FreqFusion(256, 256)
            self.conv4 = ConvRelu(96, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats:
                - s4:  [B, C4,  H/4,  W/4]
                - s8:  [B, C8,  H/8,  W/8]
                - s16: [B, C16, H/16, W/16]
                - s32: [B, C32, H/32, W/32]
        Returns:
            logits: [B, 2, H, W]
        """
        x = self.conv1(feats['s32'])  # [B, 256, H/32, W/32]
        x = self.cbam1(self.blend1(x))
        # Use FreqFusion for upsampling to the next scale.
        _, x_hr, x_lr = self.ff1(hr_feat=self.conv2(feats['s16']), lr_feat=x)  # -> H/16
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.cbam2(self.blend2(x))
        _, x_hr, x_lr = self.ff2(hr_feat=self.conv3(feats['s8']), lr_feat=x)  # -> H/8
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.cbam3(self.blend3(x))
        _, x_hr, x_lr = self.ff3(hr_feat=self.conv4(feats['s4']), lr_feat=x)  # -> H/4
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        logits = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)  # [B, 2, H, W]
        return logits



# VOS model
class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        self.decoder = Decoder(ver)


@dataclass
class HFCRConfig:
    # A fixed internal resolution simplifies feature caching and cross-video token sampling.
    model_input_size: int = 512

    # Historical backtracking.
    max_history: int = 5
    detach_history: bool = True

    # Future inference.
    fut_hidden: int = 128
    detach_future_mask: bool = True

    # Cross-video reasoning.
    cross_attn_dim: int = 128


class HFCR(nn.Module):
    """
    HFCR: Unified UVOS with Historical, Future, and Cross-Video Reasoning.

    Forward inputs:
        imgs:  [B, L, 3, H, W]
        flows: [B, L, 3, H, W]

    Forward outputs:
        scores: [B, L, 2, H, W] (logits)
        masks:  [B, L, 1, H, W] (hard argmax, provided for convenience)
        aux:
            his_logits: [B, L, 1, H/16, W/16]
            fut_logits: [B, L, 1, H/16, W/16]
            con_loss:   scalar
    """

    def __init__(self, ver: str, cfg: Optional[HFCRConfig] = None):
        super().__init__()
        self.vos = VOS(ver)
        self.cfg = cfg or HFCRConfig()

        # ConvNeXtV2-Tiny has s16 channels = 384. For rn101, s16 channels = 1024.
        s16_channels = 384 if ver == "convnextv2" else 1024

        self.his = HistoricalBacktracking(channels=s16_channels, max_history=self.cfg.max_history)
        self.his_head = nn.Conv2d(s16_channels, 1, kernel_size=1)

        self.lowres_mask_head = nn.Conv2d(s16_channels, 1, kernel_size=1)
        self.fut = FutureInference(feat_channels=s16_channels, flow_channels=3, hidden=self.cfg.fut_hidden)
        self.fut_proj = nn.Conv2d(1, s16_channels, kernel_size=1)

        self.cross = CrossVideoReasoning(channels=s16_channels, attn_dim=self.cfg.cross_attn_dim)

        # Feature fusion at s16 (residual).
        self.fuse_s16 = nn.Sequential(
            nn.Conv2d(s16_channels * 4, s16_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(s16_channels, s16_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self,
        imgs: torch.Tensor,
        flows: torch.Tensor,
        video_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, l, _, h_in, w_in = imgs.size()

        # Resize to a fixed internal size for stable multi-scale feature maps.
        s = int(self.cfg.model_input_size)
        if (h_in, w_in) != (s, s):
            imgs_ = F.interpolate(imgs.view(b * l, -1, h_in, w_in), size=(s, s), mode="bicubic", align_corners=False)
            flows_ = F.interpolate(flows.view(b * l, -1, h_in, w_in), size=(s, s), mode="bicubic", align_corners=False)
            imgs = imgs_.view(b, l, -1, s, s)
            flows = flows_.view(b, l, -1, s, s)

        score_lst: List[torch.Tensor] = []
        mask_lst: List[torch.Tensor] = []
        his_logits_lst: List[torch.Tensor] = []
        fut_logits_lst: List[torch.Tensor] = []

        past_s16: List[torch.Tensor] = []
        memory: Optional[torch.Tensor] = None
        con_losses: List[torch.Tensor] = []

        for t in range(l):
            # Backbone features.
            app_feats = self.vos.app_encoder(imgs[:, t])  # dict of multi-scale features
            mo_feats = self.vos.mo_encoder(flows[:, t])

            feats = {
                "s4": app_feats["s4"] + mo_feats["s4"],    # [B, C4,  s/4,  s/4]
                "s8": app_feats["s8"] + mo_feats["s8"],    # [B, C8,  s/8,  s/8]
                "s16": app_feats["s16"] + mo_feats["s16"],  # [B, C16, s/16, s/16]
                "s32": app_feats["s32"] + mo_feats["s32"],  # [B, C32, s/32, s/32]
            }

            s16 = feats["s16"]  # [B, C16, s/16, s/16]

            # Historical backtracking.
            e_his, memory = self.his(cur=s16, past=past_s16, prev_memory=memory)
            his_logits = self.his_head(e_his)  # [B, 1, s/16, s/16]
            his_logits_lst.append(his_logits)

            # Future inference (next-frame mask prediction).
            lowres_mask_logits = self.lowres_mask_head(s16)  # [B, 1, s/16, s/16]
            mask_t = torch.sigmoid(lowres_mask_logits)
            if self.cfg.detach_future_mask:
                mask_t = mask_t.detach()
            flow_lr = F.interpolate(flows[:, t], size=s16.shape[-2:], mode="bilinear", align_corners=False)
            fut_logits = self.fut(feat_t=s16, mask_t=mask_t, flow_t=flow_lr)  # [B, 1, s/16, s/16]
            fut_logits_lst.append(fut_logits)

            # Cross-video reasoning.
            cross_ctx, con_loss_t = self.cross(feat=s16, video_ids=video_ids)  # [B, C16, 1, 1], scalar
            con_losses.append(con_loss_t)

            # Fuse: s16 + f(s16, E_his, CrossCtx, FutPred)
            # Include the EMA memory as an additional broadcast context (cheap but effective).
            mem_map = memory.expand_as(s16) if memory is not None else torch.zeros_like(s16)
            cross_map = cross_ctx.expand_as(s16) + mem_map  # [B, C16, s/16, s/16]
            fut_map = self.fut_proj(fut_logits)   # [B, C16, s/16, s/16]
            fuse_in = torch.cat([s16, e_his, cross_map, fut_map], dim=1)  # [B, 4*C16, s/16, s/16]
            feats["s16"] = s16 + self.fuse_s16(fuse_in)  # [B, C16, s/16, s/16]

            # Decoder predicts logits at the internal resolution, then upscale back.
            logits = self.vos.decoder(feats)  # [B, 2, s, s]
            logits = F.interpolate(logits, size=(h_in, w_in), mode="bicubic", align_corners=False)  # [B, 2, H, W]
            score_lst.append(logits)

            # Hard masks (kept for evaluation/inference convenience).
            pred_mask = logits.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
            mask_lst.append(pred_mask)

            # Cache current s16 for history.
            past_s16.append(s16.detach() if self.cfg.detach_history else s16)
            if len(past_s16) > self.cfg.max_history:
                past_s16.pop(0)

        scores = torch.stack(score_lst, dim=1)  # [B, L, 2, H, W]
        masks = torch.stack(mask_lst, dim=1)    # [B, L, 1, H, W]
        aux = {
            "his_logits": torch.stack(his_logits_lst, dim=1),  # [B, L, 1, s/16, s/16]
            "fut_logits": torch.stack(fut_logits_lst, dim=1),  # [B, L, 1, s/16, s/16]
            "con_loss": torch.stack(con_losses).mean() if len(con_losses) else scores.new_tensor(0.0),
        }
        return {"scores": scores, "masks": masks, "aux": aux}

