import numpy as np
from amni.storage.catalog import TextureCatalog

# Semantic nonce domains for Qwen layer types
# Based on functional roles in LLMs (empirically motivated):
# - Layers 0-11  (early):    syntax / token-level  → nonce 1000
# - Layers 12-23 (middle):   factual / entity       → nonce 5000
# - Layers 24-35 (upper-mid):reasoning / abstract   → nonce 9000
# - Layers 36-47 (late):     output / style         → nonce 13000
# MLP neurons 0..inter/2 carry "concept" specialization
# MLP neurons inter/2..inter carry "relation" specialization
DOMAIN_NONCES = {
    "syntax":    1000,
    "factual":   5000,
    "reasoning": 9000,
    "output":   13000,
}
NEURON_NONCES = {
    "concept":   2000,
    "relation":  8000,
}

def layer_domain(layer_idx: int, n_layers: int) -> int:
    ratio = layer_idx / max(n_layers - 1, 1)
    return (DOMAIN_NONCES["syntax"]    if ratio < 0.25 else
            DOMAIN_NONCES["factual"]   if ratio < 0.50 else
            DOMAIN_NONCES["reasoning"] if ratio < 0.75 else
            DOMAIN_NONCES["output"])

def tag_mlp_pages(catalog: TextureCatalog, layer_idx: int, proj: str,
                  n_layers: int, inter_size: int, page_size: int):
    """
    Tags each page of a gate/up projection with a nonce.
    First half of neurons (concept): domain nonce.
    Second half (relation): relation nonce.
    """
    key  = f"layers.{layer_idx}.mlp.{proj}"
    meta = catalog.get(key)
    if not meta or "weights" not in meta:
        return
    wm      = meta["weights"]
    n_pages = wm["num_pages"]
    shape   = wm["shape"]  # (inter, hidden)
    rows    = shape[0]
    split_page = max(1, n_pages // 2)
    domain_n   = layer_domain(layer_idx, n_layers)
    nonces = [domain_n if p < split_page else NEURON_NONCES["relation"]
              for p in range(n_pages)]
    wm["page_nonces"] = nonces

def tag_attn_pages(catalog: TextureCatalog, layer_idx: int,
                   proj: str, n_layers: int):
    """Tags attention projection pages with layer domain nonce."""
    key  = f"layers.{layer_idx}.self_attn.{proj}"
    meta = catalog.get(key)
    if not meta or "weights" not in meta:
        return
    n_pages  = meta["weights"]["num_pages"]
    nonce    = layer_domain(layer_idx, n_layers)
    meta["weights"]["page_nonces"] = [nonce] * n_pages

def apply_atlas_tags(catalog: TextureCatalog, n_layers: int,
                     inter_size: int, page_size: int):
    """
    Walk the catalog and write page_nonces into every layer entry.
    This is a post-conversion step — no re-encoding needed.
    """
    print(f"  Tagging {n_layers} layers with atlas nonces ...")
    tagged = 0
    for i in range(n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tag_mlp_pages(catalog, i, proj, n_layers, inter_size, page_size)
            tagged += 1
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tag_attn_pages(catalog, i, proj, n_layers)
            tagged += 1
    print(f"  Tagged {tagged} weight tensors with semantic nonces.")
    return catalog

def domain_nonce_for_prompt(prompt: str) -> list:
    """
    Simple keyword-based domain router.
    Returns list of active nonces to load from the texture manager.
    A real system would use the Atlas word lookup.
    """
    p = prompt.lower()
    nonces = []
    if any(w in p for w in ("code", "function", "python", "debug", "algorithm", "program", "class")):
        nonces.append(DOMAIN_NONCES["reasoning"])
        nonces.append(NEURON_NONCES["concept"])
    if any(w in p for w in ("who", "when", "where", "what", "history", "born", "capital", "country")):
        nonces.append(DOMAIN_NONCES["factual"])
        nonces.append(NEURON_NONCES["concept"])
    if any(w in p for w in ("write", "story", "poem", "explain", "describe", "summarize")):
        nonces.append(DOMAIN_NONCES["output"])
        nonces.append(NEURON_NONCES["relation"])
    # Always include the full early-layer syntax domain
    nonces.append(DOMAIN_NONCES["syntax"])
    return list(set(nonces)) if nonces else list(DOMAIN_NONCES.values())
