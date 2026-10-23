# Deferred work

Things we know about but have chosen not to do yet. Each one says what the
evidence is, so it can be picked up without re-deriving it.

## 1. The ensemble's utility cost under attack

**Evidence (AgentDojo, `qwen3:4b-instruct-2507`, `important_instructions`, held-out):**

- Banking: 0/72 attacks land on the ensemble (no defense: 39/72), but it
  completes fewer real tasks. Excluding `user_task_8` and `user_task_10`, whose
  utility checks pass without real work (see 2), the ensemble completes 14/54
  versus 25/54 with no defense (p = 0.04) and 23/54 with AgentDojo's PI detector.
- Slack: attacks drop from 69/105 to 14/105 (rev 1), at a large utility cost.

**Two candidate causes. Diagnose before fixing.**

1. *Backbone capacity.* The 4B model writes the plan, and on the same model the
   Firewall and Misalignment judges get weaker. Moving the planner to the 27B
   (`--planner-model`) changed Slack dev utility only a little. Test by running
   the same harness, unchanged, at several sizes of one model family
   (`scripts/prof_run.py`, needs more compute).
2. *The harness itself.* The harness sees each tool call, not where its arguments
   came from, so it can't tell an attacker's URL or IBAN from a legitimate one.
   Candidate fix: tag every value with where it came from (user prompt vs.
   untrusted tool output), and only gate calls whose critical arguments came from
   untrusted data (the CaMeL idea). Also check whether the plan is too rigid and
   how often the Firewall gives false positives.

**First step:** for every held-out run where the ensemble lost utility that
Condition A had, record which module stopped the legitimate action (Planner
rejection, Firewall flag, Misalignment block). Tune on dev tasks only, and report
on held-out tasks.

## 2. Lenient AgentDojo utility checks

`banking/user_task_8` always passes, and `banking/user_task_10` passes when the
account is left unchanged, so an agent that gives up scores full marks. Report
banking utility both ways: all tasks, and excluding these two. Audit the Slack
and travel checks the same way before quoting their utility numbers.

## 3. DIV_ASR (generated attacks)

The AgentVigil-style mutation loop was stopped three times by a safety
classifier, so it isn't built. The plan instead: AgentDojo's other published
attack types (`tool_knowledge`, `injecagent`, `ignore_previous`, `system_message`,
`direct`) on held-out tasks, scored as "attack types never used in development".
State in the write-up that there is no adaptive attack.

## 4. Smaller items

- `GROQ_MODEL_CHAIN` in `config/settings.py` lists `qwen/qwen3.6-27b`, which is no
  longer on Groq.
- The workspace suite needs Firewall/Quarantine changes first (spike doc §4.2–4.6).
