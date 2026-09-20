"""Fetch the public AgentHarm behaviours into a local, gitignored cache.

    python scripts/fetch_agentharm.py          # download
    python scripts/fetch_agentharm.py --show   # summarise what is cached

WHY FETCHED AND NOT VENDORED
----------------------------
AgentHarm is MIT with an additional clause restricting use to improving the
safety and security of AI systems - which is what this project is - so the
licence is not the obstacle.

The obstacle is the **canary GUID** shipped inside each file. It exists so the
authors can detect their benchmark leaking into training corpora. Copying 352
harmful prompts into a public GitHub repository is exactly what that canary is
designed to discourage, regardless of what the licence permits. So the data
stays out of the repo and out of git history, and this script pulls it on
demand into a gitignored directory.

Contrast with `src/eval/attacks.py`, where AgentDojo's templates ARE vendored:
those are five short, non-harmful format strings under plain MIT. Different
material, different answer. The split is recorded in docs/HANDOFF.md.

WHAT THESE PROMPTS CAN AND CANNOT MEASURE HERE
----------------------------------------------
AgentHarm's tasks call *their* ~100 synthetic tools - "use Stable Diffusion",
absolute filesystem paths, and so on. Our agent has 15 tools across three
scenarios. **The tasks are therefore not executable in our sandbox**, and any
claim that we "ran AgentHarm" would be false.

What does transfer is the prompts, because the Harm Gate is a *task-intake*
classifier: it decides refuse-or-allow from the request text before any tool
exists. So these give a **gate-level** benchmark - does the Harm Gate flag
this request - at n=176 per class, against the n=8 our own direct-harm suite
manages. That is a different quantity from `HS` in the GAI, which is
agent-level compliance measured from tool calls, and `demos/harm_gate_bench.py`
names it separately for exactly that reason.

The pairing is the valuable part: every harmful behaviour has a benign
counterpart in the same category, so detection and over-refusal are measured
on the same axis rather than traded off invisibly.

Citation:
    AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents.
    Andriushchenko et al., UK AI Safety Institute and Gray Swan AI, 2024.
    https://arxiv.org/abs/2410.09024
    https://huggingface.co/datasets/ai-safety-institute/AgentHarm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402

BASE_URL = (
    "https://huggingface.co/datasets/ai-safety-institute/AgentHarm"
    "/resolve/main/benchmark/"
)
FILES = {
    "harmful": "harmful_behaviors_test_public.json",
    "benign": "benign_behaviors_test_public.json",
    # AgentHarm's own validation split. Small (8 base behaviours), and by
    # their intent it is dev data - so it joins our dev half rather than the
    # held-out half. See src/eval/agentharm.py.
    "harmful_validation": "harmful_behaviors_validation.json",
    "benign_validation": "benign_behaviors_validation.json",
}


def cache_dir() -> Path:
    """Where the downloaded behaviours live. Gitignored."""
    return settings.PROJECT_ROOT / "external" / "agentharm"


def load_cached(split: str) -> list[dict]:
    """Behaviours for one split, or [] if not fetched yet.

    The canary GUID is left in the file untouched - stripping it would defeat
    the mechanism it exists for.
    """
    path = cache_dir() / FILES[split]
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["behaviors"]


def is_fetched() -> bool:
    return all((cache_dir() / name).exists() for name in FILES.values())


def summarise(split: str) -> str:
    from collections import Counter

    rows = load_cached(split)
    if not rows:
        return f"{split}: not fetched"
    categories = Counter(r.get("category", "?") for r in rows)
    listed = ", ".join(f"{k} {v}" for k, v in sorted(categories.items()))
    return f"{split}: {len(rows)} behaviours across {len(categories)} categories ({listed})"


def fetch() -> int:
    import httpx

    target = cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Fetching AgentHarm public behaviours into {target}")
    print("(gitignored - see this script's docstring for why they are not "
          "committed)\n")

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for split, name in FILES.items():
            destination = target / name
            response = client.get(BASE_URL + name)
            response.raise_for_status()
            payload = response.json()
            if "behaviors" not in payload:
                print(f"  ERROR: {name} has no 'behaviors' key; layout changed?")
                return 1
            destination.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  {name}: {len(payload['behaviors'])} behaviours "
                  f"(canary {'present' if payload.get('canary_guid') else 'MISSING'})")

    print("\nDone. Now run:  python demos/harm_gate_bench.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AgentHarm behaviours.")
    parser.add_argument("--show", action="store_true",
                        help="Summarise the local cache without downloading.")
    args = parser.parse_args()

    if args.show:
        print(f"Cache: {cache_dir()}")
        for split in FILES:
            print(f"  {summarise(split)}")
        return 0
    return fetch()


if __name__ == "__main__":
    raise SystemExit(main())
