"""Shared real-prompt family for Lane A FPA train + eval.

A1/A2 data discipline: the same English family is used for train and eval.
v0 distill trained on random token ids (plumbing only). Do not silently mix
random-id batches into A2.

Family ``real_prompt_en_v1`` is 128 local sentences (no download). The first
48 are the v0 eval core so a 32-prompt slice stays bit-identical to that pack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

FAMILY_ID = "real_prompt_en_v1"
PROMPT_CAP = 128

# First 48 = v0 real-prompt eval core (do not reorder).
FAMILY_V1_CORE: Tuple[str, ...] = (
    "The river flooded the valley after three days of rain.",
    "Please explain how a bicycle stays upright while moving.",
    "Paris is the capital of France and a major cultural center.",
    "What is seventeen times twenty-three?",
    "She left the keys on the kitchen table this morning.",
    "A compass points roughly toward magnetic north.",
    "Write a short poem about winter light on snow.",
    "The library closes at six on weekdays and four on Sunday.",
    "Why does ice float on water?",
    "He promised to call when the train reached Ottawa.",
    "Photosynthesis converts light into chemical energy in plants.",
    "Name three planets closer to the Sun than Jupiter.",
    "The bread needs twenty minutes more in the oven.",
    "If a car starts from rest and accelerates at three meters per second squared for four seconds, what is its speed?",
    "They argued about whether the map was upside down.",
    "Translate the word apple into Spanish.",
    "A week has seven days and a common year has three hundred sixty-five.",
    "The cat slept in the sunbeam until noon.",
    "Describe the water cycle in two sentences.",
    "Mercury is the closest planet to the Sun.",
    "I need a recipe for tomato soup without cream.",
    "Sound travels slower than light.",
    "The museum exhibit opens next Tuesday.",
    "How do you reverse a list in Python?",
    "The mountain trail is icy after sunset.",
    "Gold is denser than aluminum.",
    "Please summarize the plot of a typical fable.",
    "The battery lasted only four hours under load.",
    "What city is the capital of Japan?",
    "She planted beans along the fence in May.",
    "A triangle has three sides and three angles.",
    "The radio played a song I had not heard in years.",
    "Explain gravity to a ten-year-old.",
    "The harbor was quiet except for gulls.",
    "Two plus two equals four.",
    "He packed a raincoat because the forecast said storms.",
    "Iron rusts when exposed to water and air.",
    "Where does the Nile empty?",
    "The clock on the tower was five minutes fast.",
    "List the primary colors of light.",
    "They crossed the bridge before the fog arrived.",
    "A sonnet usually has fourteen lines.",
    "The server returned a 404 for the missing page.",
    "Why is the sky blue at noon?",
    "She measured the window before buying curtains.",
    "The Pacific is the largest ocean on Earth.",
    "Boil the water before adding the pasta.",
    "Can you spell the word necessary?",
)

# Extension to 128 unique English sentences (documented larger mix).
FAMILY_V1_EXT: Tuple[str, ...] = (
    "Copper is a better electrical conductor than iron.",
    "The train to Montreal leaves from platform three.",
    "How many continents are usually listed in school atlases?",
    "She folded the letter and slid it into the envelope.",
    "A hectare is ten thousand square meters.",
    "Please list three uses for baking soda.",
    "The oak tree dropped acorns across the path.",
    "What does HTTP status 503 mean?",
    "He tightened the jar lid until it would not turn.",
    "Venus is wrapped in a thick atmosphere of carbon dioxide.",
    "Write two sentences about a lighthouse in fog.",
    "The recipe calls for a pinch of salt and a bay leaf.",
    "Why do we see lightning before we hear thunder?",
    "Saturday market stalls sold apples, cheese, and honey.",
    "A prime number has exactly two distinct positive divisors.",
    "The river bends south before it reaches the delta.",
    "How do you open a file for writing in Python?",
    "She counted the stairs so she would not trip in the dark.",
    "Steel is an alloy of iron and carbon.",
    "The concert was postponed because of the ice storm.",
    "Name the largest desert on Earth.",
    "He wound the clock and set the alarm for dawn.",
    "A kilometer is one thousand meters.",
    "Please explain what a hash function is for.",
    "The garden needed water after the dry week.",
    "Who wrote the play Romeo and Juliet?",
    "They mapped the cave with a laser scanner.",
    "Water freezes at zero degrees Celsius at standard pressure.",
    "The attic smelled of pine and old paper.",
    "How many bits are in a byte?",
    "She reserved a table for four at seven.",
    "Jupiter has more mass than the rest of the planets combined.",
    "A haiku is a short poem, often in three lines.",
    "The ferry crossed the strait in twenty minutes.",
    "What is the square root of eighty-one?",
    "He labeled the boxes kitchen, books, and winter clothes.",
    "Nitrogen makes up most of Earth's dry atmosphere.",
    "The trail marker was a faded blue blaze on the birch.",
    "Please convert 32 Fahrenheit to Celsius.",
    "A compiler turns source code into machine code.",
    "The kettle clicked off as soon as the water boiled.",
    "Where is the Great Barrier Reef?",
    "She sketched the floor plan before moving any furniture.",
    "An octahedron has eight faces.",
    "The newsstand still sold printed maps of the city.",
    "Why do leaves change color in autumn?",
    "He checked the tire pressure before the long drive.",
    "Sodium chloride is common table salt.",
    "The choir rehearsed the last movement twice.",
    "What year did the first human walk on the Moon?",
    "They stored the grain in a dry silo.",
    "A linked list stores elements with pointers, not a contiguous block.",
    "The creek was low after August.",
    "Please name two noble gases.",
    "She practiced the scale until the notes were even.",
    "Mount Kilimanjaro is in Tanzania.",
    "A boolean value is either true or false.",
    "The envelope had no return address.",
    "How does a lever multiply force?",
    "He replaced the washer and the drip stopped.",
    "The Andes run along the west of South America.",
    "Write a sentence that uses a semicolon correctly.",
    "The pond froze hard enough to walk on.",
    "What is the chemical symbol for gold?",
    "She sorted the screws by length and thread.",
    "A stack is last-in, first-out.",
    "The orchard rows were planted north to south.",
    "Please explain osmosis without using the word membrane more than once.",
    "He missed the last bus and walked home.",
    "Australia is both a country and a continent.",
    "A JSON object is a collection of key-value pairs.",
    "The lantern burned a clean blue flame.",
    "How many degrees are in a right angle?",
    "She translated the notice into French and German.",
    "The Sahara is a hot desert; Antarctica is a cold one.",
    "A mutex prevents two threads from writing the same data at once.",
    "The mailbox flag was already up.",
    "What causes tides on Earth?",
    "He measured twice and cut the board once.",
    "The Danube flows through several European capitals.",
)

FAMILY_V1: Tuple[str, ...] = FAMILY_V1_CORE + FAMILY_V1_EXT
DEFAULT_PROMPTS: Tuple[str, ...] = FAMILY_V1_CORE  # v0 32-prompt slice source

if len(FAMILY_V1) != PROMPT_CAP:
    raise RuntimeError(f"real_prompt_en_v1 must have {PROMPT_CAP} sentences, got {len(FAMILY_V1)}")


def data_mix_id(n_prompts: int, seq_len: int, *, source: str = FAMILY_ID) -> str:
    return f"{source}:{int(n_prompts)}x{int(seq_len)}"


def prompt_fingerprint(prompts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()[:12]


def load_prompts(path: str = "", limit: int = 32, *, family: Sequence[str] = FAMILY_V1) -> List[str]:
    texts: List[str] = []
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"prompts file not found: {path}")
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(raw)
            if isinstance(data, list):
                texts = [str(x).strip() for x in data]
            elif isinstance(data, dict) and "prompts" in data:
                texts = [str(x).strip() for x in data["prompts"]]
            else:
                raise ValueError("JSON prompts must be a list or {prompts: [...]}")
        else:
            texts = [ln.strip() for ln in raw.splitlines()]
    else:
        texts = list(family)
    texts = [t for t in texts if t and not t.startswith("#")]
    if not texts:
        raise ValueError("no prompts after filtering")
    n = min(len(texts), max(1, min(int(limit), PROMPT_CAP)))
    return texts[:n]


class WordTokenizer:
    """Tiny fallback when transformers/teacher tokenizer is absent (synthetic)."""

    def __init__(self, vocab_size: int = 128, pad_id: int = 0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_id
        self.eos_token_id = 1

    def __call__(self, texts: Sequence[str], padding=True, truncation=True, max_length=64, return_tensors="pt"):
        rows = []
        for t in texts:
            ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in t[:max_length]]
            if not ids:
                ids = [2]
            if truncation:
                ids = ids[:max_length]
            rows.append(ids)
        if padding:
            m = max(len(r) for r in rows)
            m = min(m, max_length)
            rows = [r + [self.pad_token_id] * (m - len(r)) for r in rows]
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


# v0 name
_WordTokenizer = WordTokenizer


def load_tokenizer(teacher_path: str, synthetic: bool, vocab: int):
    if synthetic:
        return WordTokenizer(vocab_size=vocab)
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers required for real-prompt tokenize") from e
    tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_prompts(tok: Any, prompts: Sequence[str], seq_len: int, device: torch.device) -> torch.Tensor:
    enc = tok(list(prompts), padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return ids.to(device)


def batched(ids: torch.Tensor, batch: int) -> List[torch.Tensor]:
    return [ids[i : i + batch] for i in range(0, ids.shape[0], batch)]


def cycle_batch(ids: torch.Tensor, batch: int, step: int) -> torch.Tensor:
    """Deterministic wrap around the same prompt-id table (train == eval family)."""
    n = int(ids.shape[0])
    if n <= 0:
        raise ValueError("empty prompt id table")
    start = (int(step) * int(batch)) % n
    idx = [(start + i) % n for i in range(int(batch))]
    return ids[idx]


def source_tag(path: str = "") -> str:
    if path:
        return f"real_prompt_file:{Path(path).name}"
    return FAMILY_ID
