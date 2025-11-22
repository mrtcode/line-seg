# Python
"""
Layout-Aware Transformer for line-type classification with a linear-chain CRF.

Expected JSON sample format:
{
  "lines": [
    [x1, y1, x2, y2,
     width, height, area, char_area_ratio, uppercase_ratio,
     first_char_cat, last_char_cat,
     some_font_to_prev_ratio, deltaXToPrev, deltaYToPrev,
     has_caption_label, bold_italic_ratio],
    ...
  ],
  "types": [0, 1, 2, ...]  # Direct type prediction: 0..10 inclusive
}
"""

import argparse
import json
import random
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# Constants and Fixed Training Configuration
# =============================================================================

# Labeling: 0..10 inclusive
NUM_CLASSES = 11

# Input feature layout per line
NUM_LINE_FEATURES = 16

# Padding / batching
PAD_LABEL = -100
MAX_TOKENS = 256
DEFAULT_BATCH_SIZE = 8

# Data split and reproducibility
SPLIT_RATIO = 0.9
GLOBAL_SEED = 42

# Training hparams
HPARAMS = dict(
    lr=1e-3,
    weight_decay=0.01,
    prior_lambda=1e-2,
    prior_margin=1.0,
    patience=3,
    min_lr=1e-5,
    oversample_factor=6,
)


# =============================================================================
# Relative Position Utilities
# =============================================================================

def _pairwise_rel_pos(bboxes: torch.Tensor, num_buckets: int = 32) -> torch.Tensor:
    """
    Convert pairwise Δx, Δy offsets of word-centres into integer buckets.

    Args:
        bboxes: (B, T, 4) – [x1, y1, x2, y2] in normalized coords [0,1]
        num_buckets: number of log-scaled buckets per axis
    Returns:
        bias_ids: (B, T, T) – integer bucket indices
    """
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2  # (B, T)
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2  # (B, T)

    dx = cx[:, :, None] - cx[:, None, :]  # (B, T, T)
    dy = cy[:, :, None] - cy[:, None, :]  # (B, T, T)

    def _bucket(d):
        sign = (d < 0).long()
        d = d.abs() + 1e-6  # avoid log(0)
        logd = torch.floor(torch.log(d + 1) / math.log(2))  # log-scale
        logd = logd.clamp(max=num_buckets - 1)
        return sign * num_buckets + logd  # 0…(2N-1)

    bx = _bucket(dx)
    by = _bucket(dy)
    return bx * (2 * num_buckets) + by  # unique id per (Δx, Δy)


# =============================================================================
# Attention + Transformer Blocks
# =============================================================================

class RelPosMultiheadAttention(nn.Module):
    """Multi-head attention with relative 2D position bias."""

    def __init__(self, d_model: int, n_head: int, n_rel: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rel_bias = nn.Embedding(n_rel, n_head)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, rel_ids, key_padding_mask=None):
        """
        Args:
            x: (B, T, D)
            rel_ids: (B, T, T) – output of _pairwise_rel_pos
            key_padding_mask: (B, T) – True for padding positions
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        rel_bias = self.rel_bias(rel_ids.long())  # (B, T, T, H)
        rel_bias = rel_bias.permute(0, 3, 1, 2)   # (B, H, T, T)
        scores = scores + rel_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out_proj(out)


# =============================================================================
# Geometry + Feature Embeddings
# =============================================================================

class BBoxEmbedding(nn.Module):
    """
    Discretise each coordinate into integer bins and look it up in its own
    trainable embedding table. Concatenate and project to d_model.
    """

    def __init__(self, d_model: int, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
        slice_dim = d_model // 4
        self.emb_x1 = nn.Embedding(num_bins, slice_dim)
        self.emb_y1 = nn.Embedding(num_bins, slice_dim)
        self.emb_x2 = nn.Embedding(num_bins, slice_dim)
        self.emb_y2 = nn.Embedding(num_bins, slice_dim)
        self.proj = nn.Linear(slice_dim * 4, d_model)

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: (B, T, 4) – normalised floats in [0, 1]
        Returns:
            (B, T, d_model) trainable embeddings
        """
        idx = (bboxes * (self.num_bins - 1)).long().clamp(min=0, max=self.num_bins - 1)
        x1_idx, y1_idx, x2_idx, y2_idx = idx.unbind(dim=-1)
        emb = torch.cat(
            (
                self.emb_x1(x1_idx),
                self.emb_y1(y1_idx),
                self.emb_x2(x2_idx),
                self.emb_y2(y2_idx),
            ),
            dim=-1,
        )
        return self.proj(emb)


class MultiResBBoxEmbedding(nn.Module):
    """
    Concatenate fine and coarse discretisations to capture both granular and
    global layout cues.
    """

    def __init__(self, d_model: int, fine_bins: int = 256, coarse_bins: int = 16):
        super().__init__()
        half = d_model // 2
        self.fine = BBoxEmbedding(d_model=half, num_bins=fine_bins)
        self.coarse = BBoxEmbedding(d_model=half, num_bins=coarse_bins)

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        emb_fine = self.fine(bboxes)
        emb_coarse = self.coarse(bboxes)
        return torch.cat([emb_fine, emb_coarse], dim=-1)


# =============================================================================
# Model: Layout-Aware Transformer with CRF emissions
# =============================================================================

class LayoutAwareSeg(nn.Module):
    """
    Encodes per-line features and relative layout signals; outputs token-level
    emissions for the CRF.
    """

    def __init__(self):
        super().__init__()
        # Fixed, non-configurable architecture parameters
        d_model = 64
        n_layers = 2
        n_head = 2
        num_rel_buckets = 16
        num_classes = NUM_CLASSES
        dropout = 0.2
        fine_bins = 256
        coarse_bins = 16
        char_embed_dim = 8
        char_vocab_size = 64

        self.d_model = d_model
        self.num_rel_buckets = num_rel_buckets
        self.num_classes = num_classes

        # Character embeddings for categorical features (first/last char categories)
        self.char_emb = nn.Embedding(char_vocab_size, char_embed_dim)

        # Geometry encoder for bounding boxes (x1, y1, x2, y2)
        bbox_dim = d_model // 2
        self.bbox_embed = MultiResBBoxEmbedding(
            d_model=bbox_dim,
            fine_bins=fine_bins,
            coarse_bins=coarse_bins,
        )

        # Additional numeric/categorical line features
        # Numeric: width, height, area, char_area_ratio, uppercase_ratio,
        #          some_font_to_prev_ratio, deltaXToPrev, deltaYToPrev,
        #          has_caption_label, bold_italic_ratio  -> 10 dims
        numeric_dim = 10
        char_features_dim = 2 * char_embed_dim
        total_feature_dim = numeric_dim + char_features_dim

        feature_dim = d_model - bbox_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(total_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        # Transformer layers with relative position bias
        n_rel = (2 * num_rel_buckets) ** 2
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'self_attn': RelPosMultiheadAttention(d_model, n_head, n_rel),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model),
            })
            self.layers.append(layer)

        # Token-level classifier -> emissions for CRF
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, line_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            line_features: (B, T, 16)
                [x1, y1, x2, y2,
                 width, height, area, char_area_ratio, uppercase_ratio,
                 first_char_cat, last_char_cat,
                 some_font_to_prev_ratio, deltaXToPrev, deltaYToPrev,
                 has_caption_label, bold_italic_ratio]
            mask:   (B, T) bool padding mask, True where padded
        Returns:
            emissions: (B, T, C)
        """
        # Split features
        bboxes = line_features[..., :4]

        # width, height, area, char_area_ratio, uppercase_ratio
        numeric_features_primary = line_features[..., 4:9]   # (B, T, 5)
        first_char_cat = line_features[..., 9].long()        # (B, T)
        last_char_cat = line_features[..., 10].long()        # (B, T)

        some_font_to_prev_ratio = line_features[..., 11:12]  # (B, T, 1)
        deltaXToPrev = line_features[..., 12:13]             # (B, T, 1)
        deltaYToPrev = line_features[..., 13:14]             # (B, T, 1)
        has_caption_label = line_features[..., 14:15]        # (B, T, 1)
        bold_italic_ratio = line_features[..., 15:16]        # (B, T, 1)

        rel_ids = _pairwise_rel_pos(bboxes, self.num_rel_buckets)  # (B, T, T)

        # Geometry embedding
        bbox_emb = self.bbox_embed(bboxes)  # (B, T, bbox_dim)

        # Categorical embeddings
        first_char_emb = self.char_emb(first_char_cat)  # (B, T, char_dim)
        last_char_emb = self.char_emb(last_char_cat)    # (B, T, char_dim)

        # Combine and project non-geometry features
        combined_features = torch.cat([
            numeric_features_primary,  # (B, T, 5)
            some_font_to_prev_ratio,   # (B, T, 1)
            deltaXToPrev,              # (B, T, 1)
            deltaYToPrev,              # (B, T, 1)
            has_caption_label,         # (B, T, 1)
            bold_italic_ratio,         # (B, T, 1)
            first_char_emb,            # (B, T, char_dim)
            last_char_emb,             # (B, T, char_dim)
        ], dim=-1)  # -> (B, T, 10 + 2*char_dim)

        feat_emb = self.feature_proj(combined_features)  # (B, T, feature_dim)

        # Concatenate geometry + features
        x = torch.cat([bbox_emb, feat_emb], dim=-1)  # (B, T, d_model)

        # Transformer encoder stack
        for layer in self.layers:
            attn_out = layer['self_attn'](layer['norm1'](x), rel_ids, key_padding_mask=mask)
            x = x + attn_out
            ffn_out = layer['ffn'](layer['norm2'](x))
            x = x + ffn_out

        # Emissions for CRF
        return self.classifier(x)


# =============================================================================
# Linear-chain CRF (no third-party dependency)
# =============================================================================

class LinearCRF(nn.Module):
    """
    Linear-chain CRF with start/end transitions.

    - emissions: FloatTensor (B, T, C)
    - tags: LongTensor (B, T)
    - mask: BoolTensor (B, T) — True for valid positions, False for padding
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))  # from i -> j
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Negative log-likelihood loss.
        """
        log_Z = self._compute_log_partition(emissions, mask)          # (B,)
        path_score = self._compute_path_score(emissions, tags, mask)  # (B,)
        nll = log_Z - path_score                                      # (B,)
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        """
        Viterbi decoding.
        Returns:
            best_paths: List[Tensor(T_i)] of tag indices per sequence
        """
        return self._viterbi_decode(emissions, mask)

    def _compute_path_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, T):
            valid = mask[:, t]
            emit_t = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * valid
        last_indices = mask.long().sum(dim=1) - 1  # (B,)
        last_tags = tags.gather(1, last_indices.unsqueeze(1)).squeeze(1)  # (B,)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]  # (B, C)
        for t in range(1, T):
            alpha_expanded = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, C, C)
            next_alpha = torch.logsumexp(alpha_expanded, dim=1)                  # (B, C)
            mask_t = mask[:, t].unsqueeze(1)  # (B,1)
            alpha = torch.where(mask_t, next_alpha + emissions[:, t], alpha)
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        B, T, C = emissions.shape
        backpointers = []
        score = self.start_transitions + emissions[:, 0]  # (B, C)
        backpointers.append(torch.zeros(B, C, dtype=torch.long, device=emissions.device))

        for t in range(1, T):
            score_exp = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, C, C)
            best_score, best_bp = torch.max(score_exp, dim=1)               # (B, C), (B, C)
            best_score = best_score + emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, best_score, score)
            backpointers.append(torch.where(mask_t, best_bp, backpointers[-1]))

        score = score + self.end_transitions
        _, best_last_tags = torch.max(score, dim=1)  # (B,)

        best_paths = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            bp_seq = [best_last_tags[b].item()]
            for t in range(seq_len - 1, 0, -1):
                bp = backpointers[t][b, bp_seq[-1]].item()
                bp_seq.append(bp)
            bp_seq.reverse()
            best_paths.append(torch.tensor(bp_seq, device=emissions.device, dtype=torch.long))
        return best_paths


# =============================================================================
# Transition Priors and Structured Constraints
# =============================================================================

# Define structural roles for labels:
# - 0..4: "start" (B-types) for five semantic types
# - 5: singleton "unit" (U-type): always single line, no continuation
# - 6..10: "continuation" (I-types) for the five types above (paired 0->6, 1->7, 2->8, 3->9, 4->10)
START_CLASSES: set = set(range(0, 6))     # 0..5 are starts
CONT_CLASSES: set = set(range(6, 11))     # 6..10 are continuations
SINGLETON_CLASSES: set = {5}              # class 5 is singleton (U-type)

# Pairs linking starts to their continuations (exclude singleton 5)
PAIR_SETS: List[set] = [
    {0, 6},
    {1, 7},
    {2, 8},
    {3, 9},
    {4, 10},
]
PAIR_UNION: set = set().union(*PAIR_SETS) if PAIR_SETS else set()

# Continuations tend to persist (longer segments)
STAY_CLASSES: set = set(CONT_CLASSES)


def apply_transition_bias_init(
        crf: LinearCRF,
        init_strength: float = 1.8,
        cross_pair_penalty: float = 5.0,
):
    """
    Initialize CRF transitions to reflect the intended structure:

    - Cannot start a sequence in a continuation (6..10).
    - Start s ∈ {0..4} can move to its own continuation cont(s).
    - Continuations have strong self-loops (long runs).
    - Cross-pair continuation jumps (I-a -> I-b) are strongly discouraged.
    - Entering a continuation from unrelated states is discouraged.
    - Singleton (5) is a unit label: discourage self-loop and transitions to continuations;
      allow moving to starts next (a new block can begin after a singleton).
    """
    with torch.no_grad():
        crf.reset_parameters()
        trans = crf.transitions
        trans[:] = 0.0

        # Disallow starting in continuations; neutral for starts (incl. singleton)
        if CONT_CLASSES:
            cont_idx = torch.tensor(sorted(CONT_CLASSES), dtype=torch.long)
            crf.start_transitions[cont_idx] = -4.0
        if START_CLASSES:
            start_idx = torch.tensor(sorted(START_CLASSES), dtype=torch.long)
            crf.start_transitions[start_idx] = 0.0

        # Encourage self-loops for continuations; mild self-loops for non-singleton starts
        for s in sorted(START_CLASSES - SINGLETON_CLASSES):
            trans[s, s] = 0.2
        for c in sorted(CONT_CLASSES):
            trans[c, c] = init_strength + 0.6

        # Singleton: discourage self-loop to prefer length-1 segments
        for u in sorted(SINGLETON_CLASSES):
            trans[u, u] = -0.8

        # Pair maps
        start_of = {max(pair): min(pair) for pair in PAIR_SETS}  # cont -> start
        cont_of = {min(pair): max(pair) for pair in PAIR_SETS}  # start -> cont

        # Encourage start -> its continuation; discourage continuation -> its start
        for s, c in cont_of.items():
            trans[s, c] = init_strength
        for c, s in start_of.items():
            trans[c, s] = -init_strength

        # Discourage entering a continuation from unrelated sources
        all_labels = set(range(NUM_CLASSES))
        for c, s in start_of.items():
            illegal_sources = all_labels - {c, s}
            for i in illegal_sources:
                trans[i, c] = trans[i, c] - init_strength

        # Strongly discourage cont -> cont across different pairs
        for ci in sorted(CONT_CLASSES):
            for cj in sorted(CONT_CLASSES):
                if ci != cj:
                    trans[ci, cj] = trans[ci, cj] - cross_pair_penalty

        # Singleton behavior: prefer moving into starts; discourage to any continuation
        for u in sorted(SINGLETON_CLASSES):
            for s in sorted(START_CLASSES):
                trans[u, s] = max(trans[u, s], 0.3)
            for c in sorted(CONT_CLASSES):
                trans[u, c] = -max(init_strength, 3.0)

        # Keep end transitions neutral
        crf.end_transitions[:] = 0.0


def transition_length_prior_loss(
        crf: LinearCRF,
        margin: float = 1.0,
        self_weight: float = 1.0,
        exit_weight: float = 0.5,
        enter_weight: float = 0.5,
        cross_pair_weight: float = 0.25,
):
    """
    Regularizer for CRF transitions with soft constraints.

    Encourages:
      - strong self-loops for labels in PAIR_UNION and for continuations (STAY_CLASSES)
    Discourages:
      - exits from paired labels into non-paired (except to starts)
      - enters from non-paired into paired (allow singleton -> start; penalize singleton -> cont)
      - cross-pair transitions within each PAIR_SET
    """
    if not PAIR_SETS and not PAIR_UNION and not STAY_CLASSES:
        return torch.tensor(0.0, device=crf.transitions.device)

    trans = crf.transitions
    device = trans.device

    all_labels = set(range(crf.num_tags))
    non_pair = all_labels - PAIR_UNION  # includes singleton 5 and any other non-paired

    loss = torch.tensor(0.0, device=device)

    # Encourage self for paired labels (starts/conts in PAIR_UNION)
    for i in PAIR_UNION:
        loss = loss + self_weight * F.relu(margin - trans[i, i]) ** 2

    # Discourage exits from paired labels into non-paired, except to starts
    for i in PAIR_UNION:
        for j in non_pair:
            if j in START_CLASSES:
                continue
            loss = loss + exit_weight * F.relu(trans[i, j] + margin) ** 2

    # Discourage enters from non-paired into paired,
    # but allow singleton -> starts; penalize singleton -> continuations
    for i in non_pair:
        for j in PAIR_UNION:
            if i in SINGLETON_CLASSES and j in START_CLASSES:
                continue
            loss = loss + enter_weight * F.relu(trans[i, j] + margin) ** 2

    # Cross-pair penalties (i != j within a pair)
    for pair in PAIR_SETS:
        p = list(pair)
        for i in p:
            for j in p:
                if i != j:
                    loss = loss + cross_pair_weight * F.relu(trans[i, j] + margin) ** 2

    # Continuations should have strong self-loops
    for i in STAY_CLASSES:
        loss = loss + 0.5 * self_weight * F.relu(margin - trans[i, i]) ** 2

    denom = (
            len(PAIR_UNION)
            + len(PAIR_UNION) * len(non_pair)
            + len(non_pair) * len(PAIR_UNION)
            + sum((len(p) * (len(p) - 1)) for p in PAIR_SETS)
            + len(STAY_CLASSES)
    )
    if denom > 0:
        loss = loss / denom

    return loss


# =============================================================================
# Data I/O and Preparation
# =============================================================================

def convert_types_to_classes(type_ids):
    """Clamp to valid class range [0, NUM_CLASSES-1]."""
    return [int(max(0, min(NUM_CLASSES - 1, int(t)))) for t in type_ids]


def read_json_records(data_dir: Path) -> list:
    """Load .json records from a directory. Accepts list or single-record JSON files."""
    records = []
    for fp in sorted(data_dir.glob("*.json")):
        try:
            with open(fp) as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    records.extend(obj)
                else:
                    records.append(obj)
        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}")
    return records


def describe_and_split(records: list, split_ratio: float = SPLIT_RATIO, seed: int = GLOBAL_SEED) -> Tuple[list, list, Dict[int, int]]:
    """Shuffle, split, and print dataset summary."""
    random.Random(seed).shuffle(records)
    split = int(split_ratio * len(records))
    train, dev = records[:split], records[split:]

    print(f"Total records: {len(records)}")
    print(f"Train: {len(train)}, Dev: {len(dev)}")

    type_counts: Dict[int, int] = {}
    avg_seq_len = 0.0

    for r in train:
        if 'lines' not in r or 'types' not in r:
            print(f"Warning: Record missing 'lines' or 'types': {list(r.keys())}")
            continue
        types = r["types"]
        for type_id in types:
            type_counts[int(type_id)] = type_counts.get(int(type_id), 0) + 1
        avg_seq_len += len(types)

    print(f"\nType distribution:")
    for type_id in sorted(type_counts.keys()):
        print(f"  Class {type_id}: {type_counts[type_id]:,}")

    if len(train) > 0:
        print(f"Average sequence length: {avg_seq_len / len(train):.1f}")
    else:
        print("No training data!")

    return train, dev, type_counts


def calculate_class_weights(type_counts: Dict[int, int], beta: float = 0.999):
    """
    Effective-number class weights (Cui et al.), normalized to mean 1.0.
    """
    class_counts: Dict[int, int] = {}
    for type_id, count in type_counts.items():
        tid = int(type_id)
        if 0 <= tid < NUM_CLASSES:
            class_counts[tid] = int(count)

    num_classes = NUM_CLASSES
    weights = []
    for class_idx in range(num_classes):
        n = max(1, int(class_counts.get(class_idx, 1)))
        eff_num = (1.0 - beta ** n) / (1.0 - beta)
        w = 1.0 / eff_num
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights * (num_classes / weights.sum())

    print(f"\nClass weights (effective-number, beta={beta}, normalized):")
    for class_idx, weight in enumerate(weights.tolist()):
        print(f"  Class {class_idx}: {weight:.3f}")

    return weights


def contains_any_type(types, wanted):
    st = set(int(t) for t in types)
    return any(t in st for t in wanted)


def compute_rare_types(type_counts: Dict[int, int], k: int = 5) -> set:
    """Pick k least frequent classes from observed training distribution."""
    items = [(int(t), int(c)) for t, c in type_counts.items() if 0 <= int(t) < NUM_CLASSES]
    if not items:
        return set()
    items.sort(key=lambda x: x[1])
    return set(t for t, _ in items[:k])


def oversample_rare_sequences(records, rare_types: set, factor=3):
    """Replicate samples that contain any of the rare types."""
    if not rare_types:
        return records[:]
    boosted = []
    for r in records:
        if 'types' not in r:
            continue
        if contains_any_type(r['types'], rare_types):
            boosted.extend([r] * factor)
        else:
            boosted.append(r)
    random.shuffle(boosted)
    return boosted


def batchify(records, bs=DEFAULT_BATCH_SIZE):
    """Yield batches (lists) of records."""
    for i in range(0, len(records), bs):
        yield records[i: i + bs]


def collate(chunk, device):
    """
    Turn a list of variable-length samples into padded tensors.

    Returns:
        x: [B, T, 16] float32 (all line features)
        y: [B, T] long (class labels with PAD_LABEL where no data)
        pad_mask: [B, T] bool (True where padded)
    """
    B = len(chunk)
    lengths = []
    for r in chunk:
        if 'lines' in r and 'types' in r:
            lengths.append(min(len(r['lines']), len(r['types']), MAX_TOKENS))
        else:
            lengths.append(0)
    max_len = max([0] + lengths)

    x_pad = torch.zeros(B, max_len, NUM_LINE_FEATURES, dtype=torch.float32)
    y_pad = torch.full((B, max_len), PAD_LABEL, dtype=torch.long)

    for i, r in enumerate(chunk):
        if 'lines' not in r or 'types' not in r:
            continue
        lines = r["lines"]
        types = r["types"]
        n = min(len(lines), len(types), MAX_TOKENS)
        if n > 0:
            x_pad[i, :n] = torch.tensor(lines[:n], dtype=torch.float32)
            y_pad[i, :n] = torch.tensor(convert_types_to_classes(types[:n]), dtype=torch.long)

    pad_mask = y_pad.eq(PAD_LABEL)
    return x_pad.to(device), y_pad.to(device), pad_mask.to(device)


def flat_masked(pred, gold, ignore=PAD_LABEL):
    """Return 1-D tensors (pred, gold) after removing positions == ignore."""
    mask = gold.ne(ignore)
    return pred[mask], gold[mask]


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, crf, dataset, device, num_classes=NUM_CLASSES, topn_confusions: int = 5):
    """
    Evaluate model on dataset using CRF Viterbi decoding.

    Returns:
        avg_loss, overall_acc, macro_precision, macro_recall, macro_f1,
        per_class_precision, per_class_recall, per_class_f1,
        (block_start_precision, block_start_recall, block_start_f1),
        top_confusions (list of (gold, pred, count, frac_of_errors)),
        singleton_pct (float 0..1),
        avg_seg_len_per_class (list of floats, len=num_classes)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    per_class_correct = torch.zeros(num_classes, device=device)
    per_class_total = torch.zeros(num_classes, device=device)
    per_class_pred = torch.zeros(num_classes, device=device)

    # Confusion (CPU tensor to aggregate across batches)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # Segment stats
    total_segments = 0
    singleton_segments = 0
    seg_len_sum = [0.0 for _ in range(num_classes)]
    seg_len_cnt = [0 for _ in range(num_classes)]

    # Block-start (binary) metrics where classes 0..5 are "start"
    bs_tp = 0
    bs_fp = 0
    bs_fn = 0

    for batch_idx, chunk in enumerate(batchify(dataset, bs=DEFAULT_BATCH_SIZE)):
        x, y, pad_mask = collate(chunk, device)
        valid_mask = ~pad_mask  # True where valid
        y_valid = y.clone()
        y_valid[pad_mask] = 0

        emissions = model(x, mask=pad_mask)
        loss = crf(emissions, y_valid, valid_mask, reduction="mean")
        total_loss += loss.item()
        n_batches += 1

        paths = crf.decode(emissions, valid_mask)

        Bsz, Tsz = y.shape
        pred_full = torch.full((Bsz, Tsz), PAD_LABEL, dtype=torch.long, device=device)
        for i, seq in enumerate(paths):
            pred_full[i, :seq.numel()] = seq

        pred, gold = flat_masked(pred_full, y)

        # Per-class statistics
        for class_idx in range(num_classes):
            mask_gold = (gold == class_idx)
            mask_pred = (pred == class_idx)
            per_class_total[class_idx] += mask_gold.sum()
            per_class_pred[class_idx] += mask_pred.sum()
            per_class_correct[class_idx] += (mask_gold & mask_pred).sum()

        # Confusion accumulation (CPU)
        pred_cpu = pred.detach().to('cpu')
        gold_cpu = gold.detach().to('cpu')
        idx = gold_cpu * num_classes + pred_cpu
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.view(num_classes, num_classes)

        # Segment stats from decoded sequences
        pred_full_cpu = pred_full.detach().to('cpu')
        for i in range(Bsz):
            row = pred_full_cpu[i]
            valid_len = int((row != PAD_LABEL).sum().item())
            if valid_len == 0:
                continue
            current_label = row[0].item()
            current_len = 1
            for t in range(1, valid_len):
                lab = row[t].item()
                if lab == current_label:
                    current_len += 1
                else:
                    # finalize previous segment
                    total_segments += 1
                    if current_len == 1:
                        singleton_segments += 1
                    if 0 <= current_label < num_classes:
                        seg_len_sum[current_label] += current_len
                        seg_len_cnt[current_label] += 1
                    # start new segment
                    current_label = lab
                    current_len = 1
            # finalize last segment
            total_segments += 1
            if current_len == 1:
                singleton_segments += 1
            if 0 <= current_label < num_classes:
                seg_len_sum[current_label] += current_len
                seg_len_cnt[current_label] += 1

        # Block-start binary metrics
        gold_is_start = (gold <= 5)
        pred_is_start = (pred <= 5)
        bs_tp += int((gold_is_start & pred_is_start).sum().item())
        bs_fp += int((~gold_is_start & pred_is_start).sum().item())
        bs_fn += int((gold_is_start & ~pred_is_start).sum().item())

        # Explicitly drop batch-scoped tensors
        del emissions, loss, x, y, pad_mask, valid_mask, y_valid, paths, pred_full, pred, gold

    total_correct = float(per_class_correct.sum().item())
    total_samples = float(per_class_total.sum().item())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for class_idx in range(num_classes):
        pc = float(per_class_correct[class_idx].item())
        pp = float(per_class_pred[class_idx].item())
        pt = float(per_class_total[class_idx].item())
        prec = pc / pp if pp > 0 else 0.0
        rec = pc / pt if pt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)

    macro_precision = sum(per_class_precision) / num_classes
    macro_recall = sum(per_class_recall) / num_classes
    macro_f1 = sum(per_class_f1) / num_classes

    bs_precision = bs_tp / (bs_tp + bs_fp) if (bs_tp + bs_fp) > 0 else 0.0
    bs_recall = bs_tp / (bs_tp + bs_fn) if (bs_tp + bs_fn) > 0 else 0.0
    bs_f1 = 2 * bs_precision * bs_recall / (bs_precision + bs_recall) if (bs_precision + bs_recall) > 0 else 0.0

    # Confusion top-N (exclude diagonal)
    conf_errors = conf.clone()
    diag = torch.eye(num_classes, dtype=torch.bool)
    conf_errors[diag] = 0
    total_errors = conf_errors.sum().item()
    flat_vals = conf_errors.view(-1)
    if topn_confusions > 0 and total_errors > 0:
        topk_vals, topk_idx = torch.topk(flat_vals, k=min(topn_confusions, flat_vals.numel()))
        top_confusions = []
        for v, idx_flat in zip(topk_vals.tolist(), topk_idx.tolist()):
            if v <= 0:
                break
            g = idx_flat // num_classes
            p = idx_flat % num_classes
            frac = v / total_errors if total_errors > 0 else 0.0
            top_confusions.append((int(g), int(p), int(v), float(frac)))
    else:
        top_confusions = []

    # Segment metrics
    singleton_pct = (singleton_segments / total_segments) if total_segments > 0 else 0.0
    avg_seg_len_per_class = [
        (seg_len_sum[c] / seg_len_cnt[c]) if seg_len_cnt[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    avg_loss = total_loss / n_batches if n_batches else 0.0

    return (
        avg_loss,
        overall_acc,
        macro_precision,
        macro_recall,
        macro_f1,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        (bs_precision, bs_recall, bs_f1),
        top_confusions,
        singleton_pct,
        avg_seg_len_per_class,
    )


# =============================================================================
# Checkpointing
# =============================================================================

@dataclass
class TrainConfig:
    """Immutable training configuration used throughout the run."""
    data_dir: Path
    output_base: str
    resume: Optional[Path]
    batch_size: int
    split_ratio: float
    seed: int
    lr: float
    weight_decay: float
    prior_lambda: float
    prior_margin: float
    patience: int
    min_lr: float
    oversample_factor: int


def save_checkpoint(path: Path,
                    model: LayoutAwareSeg,
                    crf: LinearCRF,
                    opt: optim.Optimizer,
                    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
                    epoch: int,
                    metrics: Dict[str, float]):
    """Serialize model, CRF, optimizer, scheduler, and metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "crf_state": crf.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "timestamp": time.time(),
    }, str(path))
    print(f"[checkpoint] Saved: {path}")


def load_checkpoint(path: Path,
                    model: LayoutAwareSeg,
                    crf: LinearCRF,
                    opt: Optional[optim.Optimizer] = None,
                    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None) -> Dict[str, Any]:
    """Restore model/CRF (and optionally optimizer/scheduler) from checkpoint."""
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    crf.load_state_dict(ckpt["crf_state"])
    if opt is not None and "optimizer_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"Warning: failed to load optimizer state: {e}")
    if scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            print(f"Warning: failed to load scheduler state: {e}")
    print(f"[checkpoint] Loaded: {path}")
    return ckpt


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    CLI:
      --data   : path to directory with .json samples
      --output : base filename; saves <base>.best.pt / .last.pt / .onnx / .crf.json
      --resume : optional checkpoint to resume/continue training
    """
    p = argparse.ArgumentParser(description="Layout-Aware Transformer + CRF training")
    p.add_argument("--data", type=str, required=True,
                   help="Path to a directory with .json training samples")
    p.add_argument("--output", type=str, required=True,
                   help="Output base file name (without extension)")
    p.add_argument("--resume", type=str, default=None,
                   help="Optional path to a .pt checkpoint to resume/continue training")
    return p


# =============================================================================
# Main
# =============================================================================

def select_device() -> torch.device:
    """
    Select training device.

    Supports only:
      - Apple Metal (MPS)
      - CUDA

    Raises:
        RuntimeError if neither MPS nor CUDA is available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("No supported accelerator found. This script requires MPS or CUDA.")


def main():
    args = build_arg_parser().parse_args()

    # Fixed, deterministic seeding
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # Consolidate config
    cfg = TrainConfig(
        data_dir=Path(args.data),
        output_base=args.output,
        resume=Path(args.resume) if args.resume else None,
        batch_size=DEFAULT_BATCH_SIZE,
        split_ratio=SPLIT_RATIO,
        seed=GLOBAL_SEED,
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        prior_lambda=HPARAMS["prior_lambda"],
        prior_margin=HPARAMS["prior_margin"],
        patience=HPARAMS["patience"],
        min_lr=HPARAMS["min_lr"],
        oversample_factor=HPARAMS["oversample_factor"],
    )

    # Data checks
    if not cfg.data_dir.exists():
        print(f"ERROR: data directory not found: {cfg.data_dir}")
        sys.exit(2)

    # Load + split
    all_records = read_json_records(cfg.data_dir)
    train, dev, type_counts = describe_and_split(all_records, split_ratio=cfg.split_ratio, seed=cfg.seed)

    # Oversample rare classes
    rare_types = compute_rare_types(type_counts, k=5)
    if cfg.oversample_factor > 1:
        train = oversample_rare_sequences(train, rare_types=rare_types, factor=cfg.oversample_factor)
        if rare_types:
            print(f"Oversampling sequences containing rare classes: {sorted(rare_types)} x{cfg.oversample_factor}")

    # Device selection: MPS or CUDA only
    try:
        device = select_device()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    print("Training on:", device)

    # Model + CRF (non-configurable model)
    model = LayoutAwareSeg().to(device)

    crf = LinearCRF(num_tags=NUM_CLASSES).to(device)
    apply_transition_bias_init(crf)

    # Optimizer + scheduler
    opt = optim.AdamW(list(model.parameters()) + list(crf.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg.patience, factor=0.5, min_lr=cfg.min_lr)

    PRIOR_LAMBDA = cfg.prior_lambda
    PRIOR_MARGIN = cfg.prior_margin

    # Print effective-number class weights (for insight; CRF loss doesn't use them directly)
    _ = calculate_class_weights(type_counts, beta=0.9999).to(device)

    # Resume if requested (always full resume in this streamlined script)
    start_epoch = 1
    best_f1 = -1.0
    best_ckpt_path = Path(f"{cfg.output_base}.best.pt")
    last_ckpt_path = Path(f"{cfg.output_base}.last.pt")

    if cfg.resume and cfg.resume.exists():
        ckpt = load_checkpoint(cfg.resume, model, crf, opt, scheduler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metrics = ckpt.get("metrics", {})
        best_f1 = float(best_metrics.get("macro_f1", best_f1))
    elif cfg.resume:
        print(f"Warning: --resume path does not exist: {cfg.resume}")

    # Ctrl-C handling (graceful exit)
    stop_requested = False
    train_start_time = time.time()

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[signal] Ctrl-C received. Finishing current step and saving best model...")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Training loop (continues until interrupted)
    epoch = start_epoch
    try:
        while True:
            model.train()
            crf.train()
            running_loss = 0.0
            tokens_seen = 0
            t0 = time.time()

            # Loss breakdown accumulators
            sum_nll = 0.0
            sum_prior = 0.0
            step_count = 0

            for step, chunk in enumerate(batchify(train, bs=cfg.batch_size)):
                x, y, pad_mask = collate(chunk, device)
                valid_mask = ~pad_mask
                y_valid = y.clone()
                y_valid[pad_mask] = 0

                opt.zero_grad()

                emissions = model(x, mask=pad_mask)
                nll = crf(emissions, y_valid, valid_mask, reduction="mean")
                prior = transition_length_prior_loss(
                    crf,
                    margin=PRIOR_MARGIN,
                    self_weight=1.0,
                    exit_weight=0.5,
                    enter_weight=0.5,
                    cross_pair_weight=0.25,
                )
                loss = nll + PRIOR_LAMBDA * prior

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(crf.parameters()), max_norm=1.0)
                opt.step()

                tokens_seen += valid_mask.sum().item()
                running_loss += loss.item() * max(1, valid_mask.sum().item())

                # accumulate loss components
                sum_nll += float(nll.item())
                sum_prior += float(prior.item())
                step_count += 1

                if step == 0:
                    lr = opt.param_groups[0]['lr']
                    print(f"[epoch {epoch}] LR = {lr:.3e} | prior={prior.item():.4f}")

                # Explicitly drop per-step tensors
                del emissions, nll, prior, loss, x, y, pad_mask, valid_mask, y_valid

                if stop_requested:
                    break

            train_loss = running_loss / max(1, tokens_seen)

            # Validation
            (
                val_loss,
                acc,
                prec,
                rec,
                f1,
                pc_prec,
                pc_rec,
                pc_f1,
                (bs_prec, bs_rec, bs_f1),
                top_confusions,
                singleton_pct,
                avg_seg_len_per_class,
            ) = evaluate(model, crf, dev, device, num_classes=NUM_CLASSES, topn_confusions=50)

            print(
                f"Epoch {epoch:2d} | "
                f"train loss {train_loss:5.4f} | "
                f"dev loss {val_loss:5.4f} | "
                f"acc {acc * 100:4.1f}% | "
                f"P {prec * 100:4.1f}% R {rec * 100:4.1f}% F1 {f1 * 100:4.1f}%"
            )

            # Loss breakdown summary
            if step_count > 0:
                avg_nll = sum_nll / step_count
                avg_prior = sum_prior / step_count
                avg_prior_term = PRIOR_LAMBDA * avg_prior
                total_obj = avg_nll + avg_prior_term
                prior_share = (avg_prior_term / total_obj) if total_obj > 0 else 0.0
                lr = opt.param_groups[0]['lr']
                print(
                    f"[loss] nll {avg_nll:.4f} | prior {avg_prior:.4f} "
                    f"(λ·prior {avg_prior_term:.4f}, share {prior_share*100:4.1f}%) | lr {lr:.3e}"
                )

            # Per-class metrics summary
            print("Per-class metrics:")
            for cls_idx in range(NUM_CLASSES):
                p = pc_prec[cls_idx] * 100
                r = pc_rec[cls_idx] * 100
                f_c = pc_f1[cls_idx] * 100
                print(f"  Class {cls_idx:>2}: P {p:5.1f}% | R {r:5.1f}% | F1 {f_c:5.1f}%")

            # Rare-classes macro F1 (if any rare types identified)
            if rare_types:
                rare_f1 = sum(pc_f1[c] for c in sorted(rare_types) if 0 <= c < NUM_CLASSES)
                denom = sum(1 for c in rare_types if 0 <= c < NUM_CLASSES)
                rare_macro_f1 = (rare_f1 / denom) if denom > 0 else 0.0
                print(f"Rare-classes macro F1 (classes {sorted(rare_types)}): {rare_macro_f1 * 100:5.1f}%")

            # Confusion summary (top-N)
            if top_confusions:
                print("Top confusions (gold -> pred):")
                for g, p, cnt, frac in top_confusions:
                    print(f"  {g} -> {p}: {cnt} ({frac*100:4.1f}% of errors)")

            # CRF transition diagnostics
            with torch.no_grad():
                trans = crf.transitions.detach().cpu()
                mean_t = float(trans.mean().item())
                std_t = float(trans.std(unbiased=False).item())
                diag_mean = float(trans.diag().mean().item())
                flat = trans.view(-1)
                topk_vals, topk_idx = torch.topk(flat, k=min(5, flat.numel()))
                topk_list = []
                for v, idx_flat in zip(topk_vals.tolist(), topk_idx.tolist()):
                    i = idx_flat // NUM_CLASSES
                    j = idx_flat % NUM_CLASSES
                    topk_list.append((int(i), int(j), float(v)))
                print(f"CRF transitions: mean {mean_t:.4f} | std {std_t:.4f} | mean self {diag_mean:.4f}")
                print("  Top transitions (i->j, weight): " + ", ".join([f"{i}->{j}:{w:.2f}" for i,j,w in topk_list]))

            # Segment statistics from decoded paths
            print(f"Segments: singleton {singleton_pct * 100:5.1f}%")
            print("  Avg segment length per class: " + ", ".join([f"{i}:{l:.2f}" for i, l in enumerate(avg_seg_len_per_class)]))

            # Block-start (0..5) -> binary metrics
            print(f"Block-start (0..5) -> P {bs_prec * 100:5.1f}% | R {bs_rec * 100:5.1f}% | F1 {bs_f1 * 100:5.1f}%")

            scheduler.step(val_loss)

            metrics = {
                "macro_f1": float(f1),
                "macro_precision": float(prec),
                "macro_recall": float(rec),
                "val_loss": float(val_loss),
                "acc": float(acc),
                "epoch_time_sec": float(time.time() - t0),
            }
            save_checkpoint(last_ckpt_path, model, crf, opt, scheduler, epoch, metrics)

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(best_ckpt_path, model, crf, opt, scheduler, epoch, metrics)

            total_elapsed = time.time() - train_start_time
            h = int(total_elapsed // 3600)
            m = int((total_elapsed % 3600) // 60)
            s = int(total_elapsed % 60)
            print(f"[time] Total elapsed since start: {h:02d}:{m:02d}:{s:02d}")

            if stop_requested:
                print("[train] Stop requested. Exiting training loop.")
                break

            epoch += 1

    except KeyboardInterrupt:
        print("\n[keyboard] Interrupted by user. Proceeding to save the best model...")

    # -----------------------------------------------------------------------
    # Export best model to ONNX + CRF params JSON
    # -----------------------------------------------------------------------
    try:
        if best_ckpt_path.exists():
            _ = load_checkpoint(best_ckpt_path, model, crf)  # no opt/scheduler needed
        else:
            print(f"Warning: best checkpoint not found at {best_ckpt_path}, exporting current model.")

        model_cpu = model.to("cpu").eval()

        B, T = 1, MAX_TOKENS
        dummy_features = torch.zeros(B, T, NUM_LINE_FEATURES, dtype=torch.float32)
        dummy_mask = torch.zeros(B, T, dtype=torch.bool)

        with torch.no_grad():
            _ = model_cpu(dummy_features, dummy_mask)

        onnx_path = f"{cfg.output_base}.onnx"
        torch.onnx.export(
            model_cpu,
            (dummy_features, dummy_mask),
            onnx_path,
            input_names=["line_features", "pad_mask"],
            output_names=["emissions"],
            dynamic_axes={
                "line_features": {0: "B", 1: "T"},
                "pad_mask": {0: "B", 1: "T"},
                "emissions": {0: "B", 1: "T"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"Exported ONNX model (emissions only) to '{onnx_path}'")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    crf_path = f"{cfg.output_base}.crf.json"
    try:
        state = {
            "num_tags": int(crf.num_tags),
            "transitions": crf.transitions.detach().cpu().tolist(),
            "start_transitions": crf.start_transitions.detach().cpu().tolist(),
            "end_transitions": crf.end_transitions.detach().cpu().tolist(),
            "class_to_type": {int(i): int(i) for i in range(NUM_CLASSES)},
        }
        with open(crf_path, "w") as f:
            json.dump(state, f)
        print(f"Wrote CRF params to '{crf_path}'")
    except Exception as e:
        print(f"Failed to write CRF params: {e}")


if __name__ == "__main__":
    main()