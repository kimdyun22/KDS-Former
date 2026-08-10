"""KDS-Former architecture for Korean continuous sign language recognition.

The core structural contribution separates 137 keypoints into hand and
body/face streams with independent Transformer encoders and cross-stream fusion.

Keypoint Layout (OpenPose Body-25 + Hands + Face):
  - Body+Face: joints 0~94  (25 body + 70 face)
  - Left Hand:  joints 95~115 (21 joints)
  - Right Hand: joints 116~136 (21 joints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NormBothLinear(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.eps = eps
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        return torch.matmul(
            F.normalize(x, dim=-1, eps=self.eps),
            F.normalize(self.weight, dim=0, eps=self.eps),
        )


class KDSFormer(nn.Module):
    """
    Hand-body decoupled transformer with late fusion.
    
    Architecture:
        Hand(42 joints) -> HandProj -> HandTransformer(2L)  ────────┐
                                                                    ├─> Late Fusion Encoder -> Gloss Head
        Body+Face(95 joints) -> BodyProj -> BodyTransformer(2L)  ───┘
    """
    
    # Joint group indices (Canonical Order: Pose+Face 0-94, Hands 95-136)
    BODY_FACE_RANGE = (0, 95)   # Pose(25) + Face(70) = 95 joints
    HAND_RANGE = (95, 137)      # Left Hand(21) + Right Hand(21) = 42 joints
    
    def __init__(
        self,
        num_classes: int,
        hand_joints=42,
        body_joints=95,
        d_hand=128,
        d_body=128,
        d_fusion=256,
        nhead=8,
        branch_layers=2,
        fusion_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        input_channels=2,
        classifier="linear",
        norm_scale=32.0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "num_classes must include the CTC blank class and at least one gloss class."
            )
        self.num_classes = num_classes
        self.d_fusion = d_fusion
        self.classifier_type = classifier
        self.norm_scale = norm_scale
        
        # === Branch Projections ===
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_joints * input_channels, d_hand),
            nn.LayerNorm(d_hand),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.body_proj = nn.Sequential(
            nn.Linear(body_joints * input_channels, d_body),
            nn.LayerNorm(d_body),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # === Branch Positional Encodings ===
        self.hand_pe = PositionalEncoding(d_hand, dropout=dropout)
        self.body_pe = PositionalEncoding(d_body, dropout=dropout)
        
        # === Branch Transformers ===
        hand_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_hand, nhead=nhead // 2,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.hand_encoder = nn.TransformerEncoder(hand_enc_layer, num_layers=branch_layers)
        body_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_body, nhead=nhead // 2,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.body_encoder = nn.TransformerEncoder(body_enc_layer, num_layers=branch_layers)
        
        # === Fusion Mechanism ===
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_hand + d_body, d_fusion),
            nn.LayerNorm(d_fusion),
            nn.GELU(),
        )
        self.fusion_pe = PositionalEncoding(d_fusion, dropout=dropout)
        fusion_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=nhead,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_enc_layer, num_layers=fusion_layers)
        
        # === Gloss Head ===
        if classifier == "linear":
            self.gloss_head = nn.Linear(d_fusion, num_classes)
        elif classifier == "normboth":
            self.gloss_head = NormBothLinear(d_fusion, num_classes)
        else:
            raise ValueError(f"unsupported classifier: {classifier}")
        
    def forward(self, x, actual_frames=None):
        """
        Args:
            x: (B, T, 137, 2) keypoint input
            actual_frames: (B,) tensor with valid sequence lengths
        """
        B, T, J, C = x.shape
        
        # 0. Generate Padding Mask
        # src_key_padding_mask: (B, T) True for padded positions
        mask = None
        if actual_frames is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= actual_frames.unsqueeze(1)
        # 1. Split into Hand and Body groups
        body = x[:, :, self.BODY_FACE_RANGE[0]:self.BODY_FACE_RANGE[1], :]  # (B, T, 95, 2)
        hand = x[:, :, self.HAND_RANGE[0]:self.HAND_RANGE[1], :]            # (B, T, 42, 2)
        
        # Flatten joint dimensions
        body = body.reshape(B, T, -1)
        hand = hand.reshape(B, T, -1)
        
        # 2. Branch Projections + Positional Encoding
        hand_feat = self.hand_proj(hand)
        body_feat = self.body_proj(body)
        
        hand_feat = self.hand_pe(hand_feat)
        body_feat = self.body_pe(body_feat)
        
        # 3. Branch Transformers with Padding Mask
        hand_feat = self.hand_encoder(hand_feat, src_key_padding_mask=mask)
        body_feat = self.body_encoder(body_feat, src_key_padding_mask=mask)
        
        # 4. Fusion
        fused = torch.cat([hand_feat, body_feat], dim=-1)
        fused = self.fusion_proj(fused)
        fused = self.fusion_pe(fused)
        fused = self.fusion_encoder(fused, src_key_padding_mask=mask)
        
        # 5. Gloss Head
        gloss_logits = self.gloss_head(fused)
        if self.classifier_type == "normboth":
            gloss_logits = gloss_logits * self.norm_scale
        return gloss_logits
