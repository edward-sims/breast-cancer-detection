import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import timm
from transformers import ViTModel, ViTConfig
import einops
from typing import Dict, List, Tuple, Optional
import math

from utils.constants import MODEL_CONFIG, MULTI_VIEW_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention fusion for combining different views"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # Project queries, keys, values
        Q = (
            self.query_proj(x1)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key_proj(x2)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value_proj(x2)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        )

        return self.out_proj(context)


class VisionTransformerBackbone(nn.Module):
    """Vision Transformer backbone with medical domain adaptation"""

    def __init__(
        self, model_name: str = "vit_base_patch16_224", pretrained: bool = True
    ):
        super().__init__()

        # Load pretrained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="token",  # Use CLS token
        )

        # Medical domain adaptation layers
        self.medical_adaptation = nn.Sequential(
            nn.Linear(self.vit.num_features, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)  # [batch_size, num_features]

        # Apply medical domain adaptation
        adapted_features = self.medical_adaptation(features)

        return adapted_features


class MultiViewFusionModel(nn.Module):
    """Cutting-edge multi-view fusion model for breast cancer detection"""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.1,
        fusion_method: str = "attention",
    ):
        super().__init__()

        self.fusion_method = fusion_method
        self.backbone = VisionTransformerBackbone(backbone_name, pretrained)

        # Feature dimensions - After medical domain adaptation, output is 512
        self.feature_dim = 512

        # Multi-view fusion
        if fusion_method == "attention":
            self.fusion_layer = MultiHeadAttentionFusion(
                hidden_dim=self.feature_dim,
                num_heads=MULTI_VIEW_CONFIG["attention_heads"],
                dropout=dropout,
            )
            # For attention fusion: fused_features + cc_features + mlo_features = 3 * feature_dim
            self.fusion_proj = nn.Linear(self.feature_dim * 3, self.feature_dim)
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(self.feature_dim * 2, self.feature_dim)
        elif fusion_method == "weighted_avg":
            self.view_weights = nn.Parameter(torch.ones(2) / 2)
            self.fusion_proj = nn.Linear(self.feature_dim, self.feature_dim)

        # Patient-level aggregation
        self.patient_aggregation = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(self, cc_view, mlo_view, patient_metadata=None):
        """
        Forward pass with multi-view fusion

        Args:
            cc_view: Cranio-caudal view images [batch_size, channels, height, width]
            mlo_view: Medio-lateral oblique view images [batch_size, channels, height, width]
            patient_metadata: Optional patient metadata for conditioning
        """

        # Extract features from both views
        cc_features = self.backbone(cc_view)  # [batch_size, feature_dim]
        mlo_features = self.backbone(mlo_view)  # [batch_size, feature_dim]

        # Multi-view fusion
        if self.fusion_method == "attention":
            # Use attention to fuse views
            fused_features = self.fusion_layer(
                cc_features.unsqueeze(1), mlo_features.unsqueeze(1)
            )
            fused_features = fused_features.squeeze(1)
            # Concatenate with original features for richer representation
            combined_features = torch.cat(
                [fused_features, cc_features, mlo_features], dim=1
            )
            fused_features = self.fusion_proj(combined_features)

        elif self.fusion_method == "concat":
            # Simple concatenation
            combined_features = torch.cat([cc_features, mlo_features], dim=1)
            fused_features = self.fusion_proj(combined_features)

        elif self.fusion_method == "weighted_avg":
            # Weighted average
            weights = F.softmax(self.view_weights, dim=0)
            fused_features = weights[0] * cc_features + weights[1] * mlo_features
            fused_features = self.fusion_proj(fused_features)

        # Patient-level aggregation
        patient_features = self.patient_aggregation(fused_features)

        # Classification
        logits = self.classifier(patient_features)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(patient_features)

        return logits, uncertainty


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights))

    def forward(self, cc_view, mlo_view, patient_metadata=None):
        outputs = []
        uncertainties = []

        for model in self.models:
            logits, uncertainty = model(cc_view, mlo_view, patient_metadata)
            outputs.append(logits)
            uncertainties.append(uncertainty)

        # Weighted ensemble
        weights = F.softmax(self.weights, dim=0)
        ensemble_logits = sum(w * out for w, out in zip(weights, outputs))
        ensemble_uncertainty = sum(w * unc for w, unc in zip(weights, uncertainties))

        return ensemble_logits, ensemble_uncertainty


def create_model(model_type: str = "multiview_fusion", **kwargs) -> nn.Module:
    """Factory function to create different model architectures"""

    if model_type == "multiview_fusion":
        return MultiViewFusionModel(**kwargs)
    elif model_type == "ensemble":
        models = [
            MultiViewFusionModel(fusion_method="attention", **kwargs),
            MultiViewFusionModel(fusion_method="concat", **kwargs),
            MultiViewFusionModel(fusion_method="weighted_avg", **kwargs),
        ]
        return EnsembleModel(models)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Legacy model for compatibility
class se_resnext101_32x4d(nn.Module):
    """Legacy model for backward compatibility"""

    def __init__(self):
        super().__init__()
        import pretrainedmodels
        import ssl

        # Workaround for SSL errors
        ssl._create_default_https_context = ssl._create_unverified_context

        self.model_ft = nn.Sequential(
            *list(
                pretrainedmodels.__dict__["se_resnext101_32x4d"](
                    num_classes=1000, pretrained="imagenet"
                ).children()
            )[:-2]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        output = self.binary_head(fea)
        return output
