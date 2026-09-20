"""Loading and splitting AgentHarm's behaviours for Harm Gate work.

WHY THERE IS A SPLIT AT ALL
---------------------------
The Harm Gate's rubric was hand-tuned against six harmful cases we wrote
ourselves, using the vocabulary the rubric looks for. Measured against
AgentHarm it caught 2 of 176. That is what overfitting a detector to its own
test set looks like, and the lesson is not "tune it against 352 cases
instead" - it is "tune against data you do not then report on".

So `test_public` is split in half, deterministically:

    dev       tune here, look at these prompts as much as you like
    heldout   never inspected while tuning; the number that gets reported

**The split is by `id_original`, not by row.** AgentHarm ships four augmented
variants of each base behaviour, and they are near-paraphrases. Splitting by
row would put variants of the same behaviour on both sides, so the rubric
would be tuned on one phrasing and "evaluated" on a reworded copy of it - a
leak that would make the held-out number meaningless in exactly the flattering
direction.

The 8-behaviour `validation` split AgentHarm provides is also loaded, and is
dev data by their intent.

Everything here is pure file reading. No network (see
`scripts/fetch_agentharm.py`), no LLM calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import settings

FILES = {
    ("harmful", "test"): "harmful_behaviors_test_public.json",
    ("benign", "test"): "benign_behaviors_test_public.json",
    ("harmful", "validation"): "harmful_behaviors_validation.json",
    ("benign", "validation"): "benign_behaviors_validation.json",
}


def cache_dir() -> Path:
    return settings.PROJECT_ROOT / "external" / "agentharm"


def load(kind: str = "harmful", split: str = "test") -> list[dict]:
    """Behaviours for one file, or [] when it has not been fetched."""
    path = cache_dir() / FILES[(kind, split)]
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["behaviors"]


def is_dev(behaviour: dict) -> bool:
    """Which half of `test_public` a behaviour falls in.

    Hashed on `id_original` so every augmentation of one base behaviour lands
    on the same side. Stable across machines and runs - `hash()` is salted per
    process and would silently reshuffle the split between the tuning run and
    the reporting run.
    """
    key = str(behaviour.get("id_original") or behaviour.get("id", ""))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def split_behaviours(behaviours: list[dict]) -> tuple[list[dict], list[dict]]:
    """(dev, heldout), grouped so no base behaviour straddles the boundary."""
    dev = [b for b in behaviours if is_dev(b)]
    heldout = [b for b in behaviours if not is_dev(b)]
    return dev, heldout


def dev_set(kind: str = "harmful") -> list[dict]:
    """Everything it is legitimate to tune against: our dev half + theirs."""
    return split_behaviours(load(kind, "test"))[0] + load(kind, "validation")


def heldout_set(kind: str = "harmful") -> list[dict]:
    """The half that reported numbers come from. Do not tune against this."""
    return split_behaviours(load(kind, "test"))[1]


def is_fetched() -> bool:
    return (cache_dir() / FILES[("harmful", "test")]).exists()


__all__ = [
    "FILES", "cache_dir", "dev_set", "heldout_set", "is_dev", "is_fetched",
    "load", "split_behaviours",
]
