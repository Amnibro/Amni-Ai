"""
Amni-A1 Semantic Column Architecture
======================================
The key innovation that eliminates the layer depth problem.

Standard transformer:  48 layers sequential → O(48 * layer_cost)
Amni-A1 columns:       8 columns × 6 layers parallel → O(6 * layer_cost)

Each Semantic Column handles one knowledge domain:
  Column 0: General knowledge (always active)
  Column 1: Coding & programming
  Column 2: Mathematics & logic
  Column 3: Science & engineering
  Column 4: Language & linguistics
  Column 5: Social & human behavior
  Column 6: Creative & artistic
  Column 7: Spatial & physical world

For any given query, the Reffelt nonce router activates 1-3 columns.
The rest are COMPLETELY SKIPPED — zero compute, zero memory access.

Within each column, blocks process sequentially (6 deep).
Across columns, processing is PARALLEL (GPU batches all active columns).

Every 2 blocks, active columns exchange information via a lightweight
CrossColumnMerge (weighted average, O(n_active × hidden)).

Result: A 48-block model with the latency of a 6-block model.
Adding more columns (= more knowledge domains) adds WIDTH, not DEPTH.
A 400B model with 64 columns runs at the same speed as a 14B with 8.

Scaling table:
  14B:    8 columns × 6 depth × hidden=5120  (2-3 active → 6L effective)
  70B:   16 columns × 6 depth × hidden=8192  (2-3 active → 6L effective)
  400B:  32 columns × 8 depth × hidden=12288 (2-4 active → 8L effective)
  1T+:   64 columns × 8 depth × hidden=16384 (2-4 active → 8L effective)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from amni.a1.core import (
    A1Block, HybridLinear, a1_rms_norm, get_silu_lut,
    DEVICE, WDTYPE, _is_gpu
)


# ===================================================================
#  SEMANTIC COLUMN — A stack of blocks for one knowledge domain
# ===================================================================

class SemanticColumn:
    """
    A vertical stack of A1Blocks that handles one semantic domain.

    Each column is an independent mini-transformer:
    - Has its own blocks (attention + MLP)
    - Processes input independently of other columns
    - Tagged with a Reffelt nonce range for routing

    Columns are the unit of activation/deactivation.
    If a column is inactive, ALL of its blocks are skipped.
    """

    def __init__(self, name: str, domain: str, n_blocks: int,
                 hidden: int, n_heads: int, n_kv_heads: int, inter: int,
                 nonce_range: Tuple[int, int] = (0, 10000),
                 sparsity_threshold: float = 0.01):
        self.name = name
        self.domain = domain
        self.nonce_range = nonce_range
        self.active = True
        self.activation_weight = 1.0  # how strongly this column contributes

        self.blocks = [
            A1Block(f"{name}.block.{i}", hidden, n_heads, n_kv_heads, inter,
                    sparsity_threshold)
            for i in range(n_blocks)
        ]

        # Assign nonces to blocks within this column
        nonce_step = (nonce_range[1] - nonce_range[0]) // max(n_blocks, 1)
        for i, blk in enumerate(self.blocks):
            blk.nonce = nonce_range[0] + i * nonce_step

    def reset_kv(self):
        for b in self.blocks:
            b.reset_kv()

    def forward(self, x: torch.Tensor, use_cache: bool = True,
                sparse_mask: Optional[torch.Tensor] = None,
                merge_callback=None) -> torch.Tensor:
        """
        Forward through all blocks in this column.
        merge_callback: called every 2 blocks for cross-column exchange.
        """
        if not self.active:
            return x  # SKIP entire column

        for i, block in enumerate(self.blocks):
            x = block.forward(x, use_cache, sparse_mask)

            # Cross-column merge every 2 blocks
            if merge_callback is not None and (i + 1) % 2 == 0:
                x = merge_callback(self.name, x, i)

            # Update sparsity mask from current activations
            sparse_mask = x.abs().max(dim=0).values.max(dim=0).values > 0.01

        return x

    def load_synthetic_weights(self, hidden: int, inter: int,
                                n_kv_heads: int, n_heads: int):
        for blk in self.blocks:
            blk.load_synthetic_weights(hidden, inter, n_kv_heads, n_heads)

    @property
    def n_params(self) -> int:
        """Estimate parameter count."""
        if not self.blocks:
            return 0
        # Per block: 4 attn projections + 3 MLP projections
        blk = self.blocks[0]
        attn_params = (blk.attn.q_proj.in_f * blk.attn.q_proj.out_f +
                       blk.attn.k_proj.in_f * blk.attn.k_proj.out_f +
                       blk.attn.v_proj.in_f * blk.attn.v_proj.out_f +
                       blk.attn.o_proj.in_f * blk.attn.o_proj.out_f)
        mlp_params = (blk.mlp.gate.in_f * blk.mlp.gate.out_f +
                      blk.mlp.up.in_f * blk.mlp.up.out_f +
                      blk.mlp.down.in_f * blk.mlp.down.out_f)
        return len(self.blocks) * (attn_params + mlp_params)


# ===================================================================
#  COLUMN ROUTER — Reffelt nonce-based column activation
# ===================================================================

class ColumnRouter:
    """
    Routes input to relevant Semantic Columns based on Reffelt nonces.

    The router examines the input prompt's semantic fingerprint (nonces)
    and determines which columns should be active and their weights.

    Routing is O(1) per column — just integer distance comparison.
    No learned routing parameters, no gating networks, no MoE overhead.
    The semantic structure IS the routing mechanism.

    Always-active columns (e.g., General) have nonce=0 and are never skipped.
    """

    def __init__(self, columns: List[SemanticColumn],
                 nonce_threshold: int = 15000):
        self.columns = columns
        self.nonce_threshold = nonce_threshold
        self.active_nonces: List[int] = []

    def set_context(self, nonces: List[int]):
        """
        Set the active semantic context from input prompt.
        This determines which columns fire for the entire generation.
        Called once per prompt, not per token.
        """
        self.active_nonces = nonces
        self._update_column_activation()

    def _update_column_activation(self):
        """Activate/deactivate columns based on nonce proximity."""
        for col in self.columns:
            if col.nonce_range[0] == 0:  # General column — always active
                col.active = True
                col.activation_weight = 1.0
                continue

            # Find minimum distance from any active nonce to this column's range
            col_center = (col.nonce_range[0] + col.nonce_range[1]) // 2

            if not self.active_nonces:
                col.active = True  # no context = all active
                col.activation_weight = 1.0
                continue

            min_dist = min(abs(col_center - n) for n in self.active_nonces)

            if min_dist <= self.nonce_threshold:
                col.active = True
                # Weight inversely proportional to distance
                col.activation_weight = max(0.1, 1.0 - (min_dist / self.nonce_threshold))
            else:
                col.active = False
                col.activation_weight = 0.0

    def get_active_columns(self) -> List[SemanticColumn]:
        return [c for c in self.columns if c.active]

    def get_routing_info(self) -> Dict:
        return {
            "total_columns": len(self.columns),
            "active_columns": sum(1 for c in self.columns if c.active),
            "skipped_columns": sum(1 for c in self.columns if not c.active),
            "column_weights": {c.name: c.activation_weight
                               for c in self.columns if c.active},
        }


# ===================================================================
#  COLUMN MERGE — Combines outputs from parallel columns
# ===================================================================

class ColumnMerge:
    """
    Merges outputs from multiple active Semantic Columns.

    Merge strategy: Weighted average based on column activation weights.
    The General column always contributes with weight 1.0.
    Domain columns contribute proportionally to their nonce proximity.

    This is a lightweight operation: O(n_active_columns * hidden).
    With 2-3 active columns, it's negligible compared to block compute.

    For more sophisticated merging, a small learned projection could be added,
    but the simple weighted average works well for semantic separation.
    """

    def __init__(self, hidden: int):
        self.hidden = hidden

    def merge(self, column_outputs: List[Tuple[str, torch.Tensor, float]]
              ) -> torch.Tensor:
        """
        Merge outputs from active columns.

        column_outputs: list of (column_name, output_tensor, weight)
        Returns: merged tensor of shape (B, S, hidden)
        """
        if len(column_outputs) == 1:
            return column_outputs[0][1]  # single column, no merge needed

        # Weighted average
        total_weight = sum(w for _, _, w in column_outputs)
        if total_weight == 0:
            total_weight = 1.0

        merged = torch.zeros_like(column_outputs[0][1])
        for name, output, weight in column_outputs:
            merged += output * (weight / total_weight)

        return merged


# ===================================================================
#  CROSS-COLUMN EXCHANGE — Periodic information sharing
# ===================================================================

class CrossColumnExchange:
    """
    Enables active columns to share information at checkpoint layers.

    Called every 2 blocks within each column. Active columns briefly
    exchange a summary of their current state via lightweight averaging.

    This prevents columns from diverging too far and allows cross-domain
    reasoning (e.g., "explain coding using a cooking metaphor" needs
    both the Coding and Creative columns to share context).

    The exchange is OPTIONAL and can be disabled for maximum speed.
    When disabled, columns are fully independent (fastest, but less
    capable at cross-domain tasks).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._pending_exchanges: Dict[int, List[Tuple[str, torch.Tensor]]] = {}

    def submit(self, column_name: str, x: torch.Tensor,
               checkpoint_idx: int) -> torch.Tensor:
        """
        Submit a column's output for exchange at a checkpoint.
        Returns the column's output (possibly blended with others).

        In practice, we buffer all active columns' outputs at each checkpoint,
        then blend them. Since columns process in parallel, the exchange
        happens between parallel batches.
        """
        if not self.enabled:
            return x

        key = checkpoint_idx
        if key not in self._pending_exchanges:
            self._pending_exchanges[key] = []

        self._pending_exchanges[key].append((column_name, x))
        return x  # Return unmodified; blending happens in flush()

    def flush(self, checkpoint_idx: int) -> Dict[str, torch.Tensor]:
        """
        Blend all submitted outputs at a checkpoint.
        Returns dict of column_name → blended_output.

        Blend formula: 0.9 * own_output + 0.1 * mean(other_outputs)
        This preserves column specialization while allowing cross-pollination.
        """
        key = checkpoint_idx
        if key not in self._pending_exchanges:
            return {}

        entries = self._pending_exchanges.pop(key)
        if len(entries) <= 1:
            return {name: x for name, x in entries}

        # Compute mean of all columns
        all_outputs = torch.stack([x for _, x in entries])
        mean_output = all_outputs.mean(dim=0)

        # Blend: 90% own + 10% mean
        blended = {}
        for name, x in entries:
            blended[name] = 0.9 * x + 0.1 * mean_output

        return blended

    def reset(self):
        self._pending_exchanges.clear()


# ===================================================================
#  DEFAULT DOMAIN CONFIGURATION
# ===================================================================

# Standard 8-domain configuration for Amni-A1
DEFAULT_DOMAINS = [
    {"name": "general",  "domain": "General Knowledge",      "nonce_range": (0, 0)},       # always active
    {"name": "coding",   "domain": "Coding & Programming",   "nonce_range": (10000, 25000)},
    {"name": "math",     "domain": "Mathematics & Logic",    "nonce_range": (25000, 40000)},
    {"name": "science",  "domain": "Science & Engineering",  "nonce_range": (40000, 55000)},
    {"name": "language", "domain": "Language & Linguistics",  "nonce_range": (55000, 70000)},
    {"name": "social",   "domain": "Social & Human Behavior","nonce_range": (70000, 85000)},
    {"name": "creative", "domain": "Creative & Artistic",    "nonce_range": (85000, 100000)},
    {"name": "spatial",  "domain": "Spatial & Physical World","nonce_range": (100000, 115000)},
]

# Scale configurations
A1_CONFIGS = {
    "a1-small": {
        "n_columns": 8, "blocks_per_column": 4, "hidden": 2048,
        "n_heads": 16, "n_kv_heads": 4, "inter": 5504,
        "total_blocks": 32, "est_params": "~3B",
    },
    "a1-medium": {
        "n_columns": 8, "blocks_per_column": 6, "hidden": 5120,
        "n_heads": 40, "n_kv_heads": 8, "inter": 13824,
        "total_blocks": 48, "est_params": "~14B",
    },
    "a1-large": {
        "n_columns": 16, "blocks_per_column": 6, "hidden": 8192,
        "n_heads": 64, "n_kv_heads": 8, "inter": 22016,
        "total_blocks": 96, "est_params": "~70B",
    },
    "a1-huge": {
        "n_columns": 32, "blocks_per_column": 8, "hidden": 12288,
        "n_heads": 96, "n_kv_heads": 8, "inter": 32768,
        "total_blocks": 256, "est_params": "~400B",
    },
    "a1-titan": {
        "n_columns": 64, "blocks_per_column": 8, "hidden": 16384,
        "n_heads": 128, "n_kv_heads": 16, "inter": 43008,
        "total_blocks": 512, "est_params": "~1T",
    },
}
