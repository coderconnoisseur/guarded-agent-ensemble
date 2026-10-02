# Related-work check — what is already published

**Searched:** 2026-10-03 · ~6 targeted queries, arXiv + industry. **Not
exhaustive** — this is a triage pass to decide whether the paper plan survives,
not a literature review. A real review is still needed before writing.

**Verdict in one line: all three claims we proposed as contributions are
substantially covered in existing work.** The plan needs recalibrating. Details
and what survives are below.

---

## Claim 1 — "defense value scales inversely with backbone capability"

**Status: largely anticipated.**

[Securing AI Agents Against Prompt Injection Attacks (arXiv 2511.15759)](https://arxiv.org/html/2511.15759v1)
reports directly:

> "Models with higher baseline vulnerability show larger absolute improvements
> from defenses, but the relative reduction remains similar across models."

That is our finding, stated first and measured over more models.
[GuardianAgentBench (arXiv 2607.20982)](https://arxiv.org/pdf/2607.20982) adds
the opposite framing — that guardrails matter *more* for capable models because
capability amplifies consequences — and
[Safety, or Just Capability? (arXiv 2607.28685)](https://arxiv.org/html/2607.28685)
audits the capability/safety confound in agent-safety benchmarks directly.

**What survives.** 2511.15759 reports Attack Success Rate and task retention
**separately**. It therefore cannot state that a defense is *net harmful* —
there is no single number for it to go negative on. Our GAI can, and does:
−0.097 on the 27B arm. "There exists a backbone capability above which this
ensemble is a net loss" is a sharper claim than "relative reduction is similar",
and it needs a composite index to make.

**The risk.** Our 27B showed no benefit partly because *our attacks were too
weak for it* (0/30 against AgentDojo templates). A reviewer will say the finding
is about our suite, not about capability. **AgentDojo would settle this**: if
the 27B is genuinely attackable there and *still* shows no composite gain, the
claim holds. If it is not attackable, we have measured our own ceiling.

---

## Claim 2 — "defenses that borrow the backbone degrade with it"

**Status: published, in both of its halves.**

The *adversarial* half is well covered —
[Lakera, "Stop Letting Models Grade Their Own Homework"](https://www.lakera.ai/blog/stop-letting-models-grade-their-own-homework-why-llm-as-a-judge-fails-at-prompt-injection-defense)
and [NHI Mgmt Group](https://nhimg.org/community/agentic-ai-and-nhis/prompt-injection-defense-why-llm-judges-break-under-attack/)
both argue that a self-judging defense shares the adversarial weakness of the
model it protects, collapsing the trust boundary.

The *capability* half — which is actually ours — is also already stated. From
the judge-reliability literature: judge performance is sensitive to the
capability of the underlying model, smaller judges detect worse, and the choice
of auditor model is a design decision. See
[Time to REFLECT (arXiv 2605.19196)](https://arxiv.org/pdf/2605.19196) and
[Preemptive Detection and Correction of Misaligned Actions (InferAct, arXiv 2407.11843)](https://arxiv.org/abs/2407.11843).

**What survives.** Only the measurement, not the claim: `MF1` 0.31 → 0.81 from
swapping *nothing but the judge model*, inside a fixed system, with the
contrast against the Harm Gate (dedicated model, zero utility cost) in the same
run. That is a clean within-system quantification of a known effect. It is a
result, not a contribution.

---

## Claim 3 — "the ensemble beats any single module"

**Status: published.** 2511.15759 runs the layered ablation and concludes "no
single mechanism achieves acceptable protection independently, validating our
multi-layered approach", with per-layer numbers (content filtering ~42%,
guardrails 62–67%, response verification ~60%, full framework 88.1%).

Ours is the same shape on a different module set. CLAUDE.md §9.1 calls this the
strongest claim the project can support — it is still *true and worth
reporting*, but it is a replication, not a finding.

---

## Claim 4 — the methods / silent-failure angle

**Status: active field, our angle adjacent but not clearly novel.**

- [Position: LLM-Safety Evaluations Lack Robustness (arXiv 2503.02574)](https://arxiv.org/html/2503.02574) — poor documentation, train/test overlap.
- [From Confident Closing to Silent Failure (arXiv 2606.09863)](https://arxiv.org/html/2606.09863) — agents asserting success the environment contradicts.
- [Guardrails as Scapegoats (arXiv 2607.19449)](https://arxiv.org/html/2607.19449v1) — silent infrastructure failures unaudited; agents treat empty payloads as real data in 56.6% of cases.
- [EvalSafetyGap (arXiv 2606.30219)](https://arxiv.org/pdf/2606.30219) — reported ASR reductions not comparable across attacks, models, judges, budgets.

**What survives.** Those papers study agents failing silently. Ours is about
**the defense harness** failing silently — a module that fails open while
recording that it ran, a `--model` flag accepted and ignored, a metric tuned and
reported on the same six cases, a latency number measuring the rate limiter.
Each produced a plausible number and each made results look *better*. That
specific catalogue, with the guards built for it, is adjacent to "Guardrails as
Scapegoats" rather than covered by it. Thin for a paper on its own; good as a
section.

---

## Composite metrics

Partly anticipated. A "Helpfulness Safety" score exists — the harmonic mean of
benign score and (1 − harm score) — and
[StepGuard (arXiv 2608.24777)](https://arxiv.org/pdf/2608.24777) shows the
security/utility tension explicitly (malicious 22.8 → 3.4, completion
70.9 → 52.8). Our GAI is richer (7 terms, weight-sensitivity analysis,
renormalisation when a term is unbuilt) but "a composite that trades safety
against utility" is not new.

---

## What this means for the paper

**Not a novel-contribution paper as currently scoped.** Everything proposed as
a contribution has a prior claim. Two honest routes remain:

1. **A replication-and-negative-result paper.** "We implemented an ensemble of
   four published defenses on small open models and found it net-harmful above
   a capability threshold, with a composite index that makes the harm visible."
   Negative and replication results are genuinely useful and some venues want
   them. This is the honest framing of what we have.
2. **Go after the sharper version of Claim 1.** It needs: AgentDojo (so the
   strong backbone is genuinely attackable and the numbers are comparable to
   published ones), **3–4 backbone capability points** rather than two, and the
   composite. If the net-harm threshold reproduces on a real benchmark across a
   capability curve, that is a claim 2511.15759 does not make.

Route 2 is the better paper and it is exactly the work already queued
(HANDOFF §12.4, §12.5). Route 1 is available now.

**Do not** claim novelty for the ensemble composition, for "don't let the model
judge itself", or for "layered beats single" without citing the above.

---

## Caveat

Six queries, one pass. Several of these were found by title and snippet rather
than read in full — 2511.15759 is the only one fetched and read directly. Before
writing, each paper cited here should be read properly, and the search widened
(ACL/EMNLP/NeurIPS proceedings, not just arXiv).
