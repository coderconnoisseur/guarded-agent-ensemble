"""One-command AgentDojo run for a machine Claude cannot see (docs/PROF_MACHINE.md).

Two phases, because the models depend on hardware nobody has measured yet:

    python scripts/prof_run.py probe
        Seconds, no downloads. Writes prof_probe_<host>.json: GPU, VRAM, RAM,
        disk, Python, Ollama, Groq key present, and a suggested model list.
        Bring that file back; the models and the exact `run` command are
        chosen from it.

    python scripts/prof_run.py run --models qwen2.5:7b [qwen2.5:14b ...]
        Hours, unattended. Per model: ollama pull, a private Ollama server
        (its own port, so an already-running Ollama is left alone), then
        Condition A and Condition B (defense revision 1) on AgentDojo banking,
        held-out split. Ends by writing prof_results_<host>_<stamp>.zip with
        every result, log, the probe, and the exact settings used.

Safe to re-run: finished runs are skipped, so an interrupted job resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 11435  # private server; 11434 may already be someone's Ollama
CONTEXT = 8192  # what every banking number so far used (spike doc 9.2)
HOST = socket.gethostname()

# q4_K_M download size (GB) and KV cache at 8k context, f16, conservative
# (GB). Ollama's library sizes; KV = 2 * layers * kv_heads * 128 * 2B * 8192.
CANDIDATES = {
    "qwen2.5:7b": (4.7, 0.47),
    "qwen2.5:14b": (9.0, 1.61),
    "qwen2.5:32b": (20.0, 2.15),
}
OVERHEAD_GB = 0.6  # compute buffers + display, measured ~0.8 on the laptop

# Measured on the 4 GB laptop, qwen3:4b-instruct, banking (spike 9.2, 10):
# seconds per run with 26/37 layers on GPU. The scale for every estimate.
REF_SEC_PER_RUN = {"A": 38.6, "B": 92.0}
RUNS_PER_CONDITION = 80  # held-out: 8 user tasks x (1 clean + 9 attacked)


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001 - probing; absence is the answer
        return ""


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        *[(f"x{i}", ctypes.c_ulonglong) for i in range(5)]]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.total / 1e9, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def gpus() -> list[dict]:
    out = sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
              "--format=csv,noheader,nounits"])
    found = []
    for line in out.splitlines():
        name, total, free, driver = [x.strip() for x in line.split(",")]
        found.append({"name": name, "vram_total_gb": round(int(total) / 1024, 1),
                      "vram_free_gb": round(int(free) / 1024, 1), "driver": driver})
    return found


def suggest(vram_free: float, ram: float | None) -> list[dict]:
    out = []
    for model, (size, kv) in CANDIDATES.items():
        need = size + kv + OVERHEAD_GB
        if need <= vram_free:
            fit, slow = "fits fully on GPU", 1.0
        elif ram and need <= vram_free + ram * 0.5:
            share = vram_free / need
            fit, slow = f"partial offload (~{share:.0%} on GPU) - slow", 3.0
        else:
            fit, slow = "does not fit", None
        hours = None
        if slow:
            # Very rough: per-token cost scales with size; full-GPU 7B on a
            # mid card is taken as ~as fast as the laptop's partly-offloaded 4B.
            scale = slow * size / 4.7
            hours = round(RUNS_PER_CONDITION * sum(REF_SEC_PER_RUN.values()) * scale / 3600, 1)
        out.append({"model": model, "needs_gb": round(need, 1), "fit": fit, "est_hours": hours})
    return out


def probe(_: argparse.Namespace) -> Path:
    gpu = gpus()
    ram = ram_gb()
    # Where Ollama will store the models, not the home drive (which on the
    # laptop this was written on had 1.6 GB free while models lived on D:).
    models_dir = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    anchor = next((p for p in [models_dir, *models_dir.parents] if p.exists()), Path.home())
    free_disk = round(shutil.disk_usage(anchor).free / 1e9, 1)
    env = (ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else ""
    report = {
        "host": HOST, "when": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(), "python": sys.version.split()[0],
        "cpu": platform.processor(), "cores": os.cpu_count(), "ram_gb": ram,
        "gpus": gpu, "ollama_models_dir": str(models_dir), "disk_free_gb_models": free_disk,
        "ollama": sh(["ollama", "--version"]) or None,
        "git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]) or None,
        "agentdojo_installed": bool(sh([sys.executable, "-c", "import agentdojo; print(1)"], 60)),
        "groq_key_in_env_file": "GROQ_API_KEY=" in env and "GROQ_API_KEY=\n" not in env,
        "suggestions": suggest(gpu[0]["vram_free_gb"] if gpu else 0.0, ram),
    }
    problems = []
    if not report["ollama"]:
        problems.append("Ollama not found - install from https://ollama.com")
    if not report["agentdojo_installed"]:
        problems.append("pip install -r requirements.txt -r requirements-agentdojo.txt")
    if not report["groq_key_in_env_file"]:
        problems.append("no GROQ_API_KEY in .env - Condition B's judge and guard models need it")
    biggest = max((CANDIDATES[s["model"]][0] for s in report["suggestions"] if s["est_hours"]), default=0)
    if free_disk < biggest + 2:
        problems.append(f"only {free_disk} GB free where Ollama stores models ({models_dir})")
    if sys.version_info < (3, 11):
        problems.append("Python 3.11+ required")
    report["problems"] = problems
    path = ROOT / f"prof_probe_{HOST}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWROTE {path}\nBring this file back. Problems to fix first: {problems or 'none'}")
    return path


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # continuous | system


def server_up(timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/version", timeout=2)
            return True
        except OSError:
            time.sleep(2)
    return False


def run(args: argparse.Namespace) -> None:
    keep_awake()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    logdir = ROOT / "results" / "prof_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = (logdir / f"run_{stamp}.log").open("a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    probe_path = probe(args)
    manifest = {"host": HOST, "started": stamp, "models": {}, "context": CONTEXT,
                "defense_revision_B": 1, "split": "heldout", "port": PORT}
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{PORT}", OLLAMA_CONTEXT_LENGTH=str(CONTEXT),
               OLLAMA_BASE_URL=f"http://127.0.0.1:{PORT}/v1", USE_LOCAL_BACKBONE="1",
               PYTHONIOENCODING="utf-8")
    server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log)
    try:
        if not server_up(180):
            say("FATAL: private Ollama server did not start"); return
        for model in args.models:
            entry = manifest["models"].setdefault(model, {})
            say(f"== {model}: pulling")
            pulled = subprocess.run(["ollama", "pull", model], env=env, stdout=log, stderr=log)
            if pulled.returncode:
                entry["error"] = "pull failed"; say(f"   pull FAILED, skipping {model}"); continue
            for cond, extra in (("A", []), ("B", ["--defense-revision", "1"])):
                started = time.time()
                say(f"   Condition {cond}: running (resumes if interrupted)")
                rc = subprocess.run(
                    [sys.executable, "-u", "demos/agentdojo_run.py", "--suite", "banking",
                     "--condition", cond, "--model", model, "--tasks", "heldout", *extra],
                    cwd=ROOT, env=env, stdout=log, stderr=log).returncode
                entry[cond] = {"exit": rc, "hours": round((time.time() - started) / 3600, 2)}
                say(f"   Condition {cond}: exit {rc} after {entry[cond]['hours']} h")
    finally:
        server.terminate()
        manifest["finished"] = datetime.now().strftime("%Y%m%d-%H%M")
        bundle(manifest, probe_path, log.name, stamp)
        log.close()


def bundle(manifest: dict, probe_path: Path, log_path: str, stamp: str) -> None:
    manifest["git_commit"] = sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    # A commit hash alone lies if the code was edited after checkout.
    manifest["git_dirty"] = bool(sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    out = ROOT / f"prof_results_{HOST}_{stamp}.zip"
    slugs = [m.replace("/", "-").replace(":", "-") for m in manifest["models"]]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.write(probe_path, probe_path.name)
        z.write(log_path, Path(log_path).name)
        for f in (ROOT / "results" / "agentdojo").glob("*.json"):
            if any(s in f.name for s in slugs):
                z.write(f, f"agentdojo/{f.name}")
        budget = ROOT / "src" / "llm" / ".budget.json"
        if budget.exists():
            z.write(budget, "budget.json")
    print(f"\nWROTE {out}\nThat zip is the whole result - bring it back.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="describe this machine; no downloads")
    r = sub.add_parser("run", help="pull models and run A + B on banking held-out")
    r.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    {"probe": probe, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
