"""Run our pipelines inside AgentDojo's environments and scoring harness.

AgentDojo scores a run on three things only - the final assistant message, the
environment after the run, and the tool calls on assistant messages
(docs/AGENTDOJO_SPIKE.md 2). So the bridge is one base `ToolRegistry` whose
handlers call AgentDojo's `runtime.run_function(env, ...)`. Condition A and B
wrap it exactly as they wrap our own tools; `loop.py` is untouched.

Requires the `agentdojo` package (requirements-agentdojo.txt), deliberately
outside requirements.txt.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatUserMessage,
    text_content_block_from_string,
)

from src.agent.loop import AgentResult
from src.tools.registry import ToolParam, ToolRegistry, ToolSpec

# (critical, read_only, returns_untrusted) for every tool in AgentDojo v1.2.2.
#
# Labelled 2026-10-04 from each tool's description and return type, BEFORE any
# run, by the rule our own tools already follow (src/tools/*.py):
#   critical  - changes state with an outside effect (send, delete, write,
#               transfer, update, book, cancel, share, invite);
#   untrusted - returns content a third party wrote (mail, files, web pages,
#               messages, transactions, vendor listings and reviews);
#   trusted   - the user's own records (balance, profile, contacts, drafts).
# These tags are what every defense keys off, which makes them a tuning
# surface: change one after seeing results and the result is tuned.
TAGS: dict[str, tuple[bool, bool, bool]] = {
    # workspace
    "add_calendar_event_participants": (True, False, False),
    "append_to_file": (True, False, True),  # returns the whole file back
    "cancel_calendar_event": (True, False, False),
    "create_calendar_event": (True, False, False),
    "create_file": (True, False, False),
    "delete_email": (True, False, False),
    "delete_file": (True, False, False),
    "get_current_day": (False, True, False),
    "get_day_calendar_events": (False, True, True),
    "get_draft_emails": (False, True, False),
    "get_file_by_id": (False, True, True),
    "get_received_emails": (False, True, True),
    "get_sent_emails": (False, True, False),
    "get_unread_emails": (False, True, True),
    "list_files": (False, True, True),  # unlike ours, returns file contents
    "reschedule_calendar_event": (True, False, False),
    "search_calendar_events": (False, True, True),
    "search_contacts_by_email": (False, True, False),
    "search_contacts_by_name": (False, True, False),
    "search_emails": (False, True, True),
    "search_files": (False, True, True),
    "search_files_by_filename": (False, True, True),
    "send_email": (True, False, False),
    "share_file": (True, False, False),
    # travel
    "check_restaurant_opening_hours": (False, True, True),
    "get_all_car_rental_companies_in_city": (False, True, True),
    "get_all_hotels_in_city": (False, True, True),
    "get_all_restaurants_in_city": (False, True, True),
    "get_car_fuel_options": (False, True, True),
    "get_car_price_per_day": (False, True, True),
    "get_car_rental_address": (False, True, True),
    "get_car_types_available": (False, True, True),
    "get_contact_information_for_restaurants": (False, True, True),
    "get_cuisine_type_for_restaurants": (False, True, True),
    "get_dietary_restrictions_for_all_restaurants": (False, True, True),
    "get_flight_information": (False, True, True),
    "get_hotels_address": (False, True, True),
    "get_hotels_prices": (False, True, True),
    "get_price_for_restaurants": (False, True, True),
    "get_rating_reviews_for_car_rental": (False, True, True),
    "get_rating_reviews_for_hotels": (False, True, True),
    "get_rating_reviews_for_restaurants": (False, True, True),
    "get_restaurants_address": (False, True, True),
    "get_user_information": (False, True, False),
    "reserve_car_rental": (True, False, False),
    "reserve_hotel": (True, False, False),
    "reserve_restaurant": (True, False, False),
    # banking
    "get_balance": (False, True, False),
    "get_iban": (False, True, False),
    "get_most_recent_transactions": (False, True, True),
    "get_scheduled_transactions": (False, True, True),
    "get_user_info": (False, True, False),
    "read_file": (False, True, True),
    "schedule_transaction": (True, False, False),
    "send_money": (True, False, False),
    "update_password": (True, False, False),
    "update_scheduled_transaction": (True, False, False),
    "update_user_info": (True, False, False),
    # slack
    "add_user_to_channel": (True, False, False),
    "get_channels": (False, True, False),
    "get_users_in_channel": (False, True, False),
    "get_webpage": (False, True, True),
    "invite_user_to_slack": (True, False, False),
    "post_webpage": (True, False, False),
    "read_channel_messages": (False, True, True),
    "read_inbox": (False, True, True),
    "remove_user_from_slack": (True, False, False),
    "send_channel_message": (True, False, False),
    "send_direct_message": (True, False, False),
}


def _param_type(schema: dict[str, Any]) -> str:
    if "type" in schema:
        return str(schema["type"])
    options = [s.get("type", "any") for s in schema.get("anyOf", [])]
    return " | ".join(options) or "any"


def build_registry(
    runtime: FunctionsRuntime,
    env: Env,
    executed: list[FunctionCall],
    outputs: list[str],
) -> ToolRegistry:
    """Our registry over AgentDojo's tools, bound to one task environment.

    Every call that reaches a handler - i.e. got past every defense wrapper -
    is appended to `executed`, which becomes the trace AgentDojo grades. A
    call a defense blocked never gets here, so it can never be scored as an
    attack that succeeded (HANDOFF 1, the `arg_contains` bug).
    """
    registry = ToolRegistry()
    for fn in runtime.functions.values():
        try:
            critical, read_only, untrusted = TAGS[fn.name]
        except KeyError:
            # An untagged tool would run with every defense blind to it.
            raise KeyError(f"AgentDojo tool {fn.name!r} has no tags in TAGS") from None
        schema = fn.parameters.model_json_schema()
        required = set(schema.get("required", []))
        params = [
            ToolParam(name, _param_type(p), p.get("description", ""), name in required)
            for name, p in schema.get("properties", {}).items()
        ]

        def handler(_name: str = fn.name, **kwargs: Any) -> str:
            executed.append(FunctionCall(function=_name, args=kwargs))
            result, error = runtime.run_function(env, _name, kwargs)
            if error:
                raise RuntimeError(error)
            text = tool_result_to_str(result)
            outputs.append(text)
            return text

        registry.register(ToolSpec(
            name=fn.name, description=fn.description, params=params,
            handler=handler, critical=critical, read_only=read_only,
            returns_untrusted=untrusted,
        ))
    return registry


Runner = Callable[[ToolRegistry], Any]  # registry -> object with .run(task)


class GuardedPipeline(BasePipelineElement):
    """One AgentDojo pipeline element that runs a whole Condition A/B task.

    `make_runner` builds a fresh ConditionA/ConditionB around each task's
    registry, so no state crosses tasks. The name must contain "local":
    `important_instructions` refuses a pipeline whose name holds no model
    string AgentDojo knows, and "local" renders as "Local model".
    """

    def __init__(self, make_runner: Runner, label: str) -> None:
        self.make_runner = make_runner
        self.name = f"local-{label}"
        self.last_result: AgentResult | None = None
        self.last_outputs: list[str] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        executed: list[FunctionCall] = []
        outputs: list[str] = []
        registry = build_registry(runtime, env, executed, outputs)
        result = self.make_runner(registry).run(query)
        self.last_result, self.last_outputs = result, outputs

        out: list[ChatMessage] = [
            ChatUserMessage(role="user", content=[text_content_block_from_string(query)])
        ]
        for call in executed:
            out.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=[call]))
        out.append(ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(result.final_answer or "")],
            tool_calls=None,
        ))
        return query, runtime, env, [*messages, *out], extra_args
