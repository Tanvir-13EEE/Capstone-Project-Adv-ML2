"""
Multi-Stream Spectral Transformer (MSST)
Core model definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    HaarDWT2d, StatisticalPooling,
    MultiStreamFrequencyExtractor,
    MultiScaleDWTBranch,
    ExpertPhysicsStreamProjector,
    MultiStreamFusion,
)
from .heads import BinaryHead, MultiClassHead, LocalizationHead


class MultiStreamSpectralTransformer(nn.Module):
    """
    Multi-Stream Spectral Transformer for AI-generated image detection.

    Three processing priorities:
        P1 — Frequency domain: HH conv, patch moments,
             global statistics, spectral slope
        P2 — Multi-scale DWT: 3-level Mallat pyramid
        P3 — Gated fusion: input-dependent stream weighting

    Args:
        img_size    : input image resolution (any square size)
        d_model     : transformer embedding dimension
        num_heads   : number of attention heads
        num_layers  : transformer encoder depth
        task        : 'binary' | 'multiclass' | 'localization'
        num_classes : number of classes (multiclass only)
    """
    def __init__(
        self,
        img_size    : int = 224,
        d_model     : int = 256,
        num_heads   : int = 8,
        num_layers  : int = 6,
        task        : str = 'binary',
        num_classes : int = 2,
    ):
        super().__init__()
        self.img_size = img_size
        self.task     = task
        self._d_model = d_model

        if   img_size <= 32: ps, stride = 4,  4
        elif img_size <= 64: ps, stride = 8,  8
        else:                ps, stride = 16, 16

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d_model, ps, stride),
        )
        self.freq_extractor   = \
            MultiStreamFrequencyExtractor(d_model, img_size)
        dwt_levels            = 3 if img_size >= 64 else 2
        self.dwt_branch       = \
            MultiScaleDWTBranch(3, d_model, dwt_levels, 7)
        self.expert_projector = \
            ExpertPhysicsStreamProjector(d_model)
        self.fusion           = MultiStreamFusion(d_model)
        self.cls_token        = \
            nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.head        = self._build_head(
            task, d_model, num_classes)
        self._init_weights()

    def _build_head(self, task, d_model, num_classes):
        if task == 'binary':
            return BinaryHead(d_model)
        if task == 'multiclass':
            return MultiClassHead(d_model, num_classes)
        if task == 'localization':
            return LocalizationHead(d_model)
        raise ValueError(f"Unknown task: {task}")

    def swap_head(self, task, num_classes=2):
        """Replace classification head without touching backbone."""
        self.task = task
        self.head = self._build_head(
            task, self._d_model, num_classes)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img, phys):
        B         = img.size(0)
        sp_tokens = self.patch_embed(img).flatten(2).transpose(1, 2)

        hh_p1, patch_p1, stat_p1, slope_p1 = \
            self.freq_extractor(img, phys[:, :25])

        sp_p2, stat_p2, cs_p2 = self.dwt_branch(img)

        ex    = self.expert_projector(phys)
        cls_t = self.cls_token.expand(B, -1, -1)

        seq, gates = self.fusion(
            hh_p1, patch_p1, stat_p1, slope_p1,
            sp_p2, stat_p2, cs_p2,
            ex['psd'], ex['prnu_snr'],
            ex['phase'], ex['forensic'],
            cls_t,
        )
        seq    = torch.cat([seq, sp_tokens], dim=1)
        out    = self.norm(self.transformer(seq))
        return self.head(out[:, 0]), gates
