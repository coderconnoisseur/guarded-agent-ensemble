# Running on the professor's machine

Two steps, a few days apart. Claude cannot see that machine, so the first
step only describes it; the models and the exact command are chosen from that.

## Once: set up (≈15 min, needs internet)

1. Install **Python 3.11+**, **git**, and **Ollama** (https://ollama.com).
2. ```
   git clone https://github.com/coderconnoisseur/guarded-agent-ensemble
   cd guarded-agent-ensemble
   pip install -r requirements.txt -r requirements-agentdojo.txt
   ```
3. Copy your `.env` (it holds `GROQ_API_KEY`) into that folder. Condition B's
   misalignment judge, Harm Gate classifier and Firewall guard run on Groq.

## Step 1: probe (seconds, downloads nothing)

```
python scripts/prof_run.py probe
```

Writes `prof_probe_<machine>.json`. **Bring it back** and paste it to Claude:
it lists the GPU, VRAM, RAM, disk and any problem to fix, plus a first guess
at which models fit. Claude replies with the exact `run` command.

## Step 2: run (hours, unattended)

The command will look like:

```
python scripts/prof_run.py run --models qwen2.5:7b
```

Per model it downloads the model, starts its **own** Ollama on port 11435
(an Ollama already running is left alone), and runs Condition A and Condition
B (defense revision 1) on AgentDojo banking, held-out split - 160 runs. The
machine is kept awake while it runs (Windows).

When it ends it prints the path of **`prof_results_<machine>_<time>.zip`**.
That zip is the whole result: every run, the logs, the probe and the exact
settings. **Bring it back.**

If it is interrupted (power, reboot, closed window), run the same command
again: finished runs are skipped and it carries on.
