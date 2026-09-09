"""System prompt templates for the ReAct loop.

Implements the prompted-JSON tool-calling protocol from CLAUDE.md 5.3. The
protocol is text-only on purpose: free-tier models report native `tools`
support inconsistently, and a small model that advertises it does not reliably
honour it. A parsed text protocol works on any chat-capable model, which is
what makes the fallback chain in config/settings.py actually usable.
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are a task-completing assistant that works by calling tools.

You have access to exactly these tools:

{tool_catalogue}

RESPONSE FORMAT - follow it exactly.

To call a tool, reply with exactly two lines:

Thought: <one sentence on why this tool call is the right next step>
Action: {{"tool": "<tool_name>", "args": {{"<arg>": "<value>"}}}}

When you have everything you need and the task is done, reply with:

Thought: <one sentence on why the task is now complete>
Final: <your answer to the user>

RULES
- Every reply must contain exactly one Action line or one Final line, never both.
- The Action value must be a single line of valid JSON with exactly the keys
  "tool" and "args". "args" is an object, and it is {{}} if the tool takes none.
- Use only the tool names listed above, spelled exactly as shown.
- After each Action you will receive an "Observation:" with the tool's result.
  Read it before deciding your next step.
- Do not invent observations, and do not write "Observation:" yourself.
- Do not wrap your reply in code fences or markdown.
- If an observation shows the task is already satisfied, answer with Final
  rather than calling another tool.
"""

# Fed back when the model's reply cannot be parsed. Restates the format and
# quotes the specific problem, since 5.3 allows one or two repair attempts
# before the step is abandoned.
PARSE_REPAIR_TEMPLATE = """\
Your last reply could not be parsed: {problem}

Reply again using exactly one of these two shapes, with nothing else around it:

Thought: <reasoning>
Action: {{"tool": "<tool_name>", "args": {{...}}}}

or:

Thought: <reasoning>
Final: <answer>
"""

OBSERVATION_TEMPLATE = "Observation: {observation}"


def build_system_prompt(tool_catalogue: str) -> str:
    """Render the system prompt for a given registry's tool catalogue."""
    return SYSTEM_PROMPT_TEMPLATE.format(tool_catalogue=tool_catalogue)


def build_repair_prompt(problem: str) -> str:
    """Render the corrective message for an unparseable reply."""
    return PARSE_REPAIR_TEMPLATE.format(problem=problem)


def build_observation(observation: str) -> str:
    """Render a tool result as the next user-role turn."""
    return OBSERVATION_TEMPLATE.format(observation=observation)
