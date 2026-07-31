"""
Project: PICU Federated Learning Fairness Pipeline
Layer 3 — Agentic Fairness System

Provides:
- subgroup and hospital fairness tools,
- validated mitigation proposal tools,
- a LangGraph agent,
- human approval for mitigation proposals,
- a console entry point for Layer 3.

Approved corrections are returned to the caller. This module does not retrain
the federated model itself.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from config import AGENT_CFG, AgentConfig


# ============================================================
# Shared pipeline context
# ============================================================

_ctx: dict[str, Any] = {}


def set_context(
    fl_results: dict[str, Any],
    clients: dict[str, Any],
    fl_config: Any,
    harm_threshold: float = 20.0,
    hospital_gap_threshold: float = 3.0,
) -> None:
    """
    Inject the current federated-learning state used by the tools.

    The optional thresholds preserve backward compatibility with existing
    calls from app.py and the benchmark runner.
    """
    if not isinstance(fl_results, dict):
        raise TypeError("fl_results must be a dictionary.")

    if not isinstance(clients, dict):
        raise TypeError("clients must be a dictionary.")

    _ctx.clear()
    _ctx.update(
        {
            "fl_results": fl_results,
            "clients": clients,
            "fl_config": fl_config,
            "harm_threshold": float(harm_threshold),
            "hospital_gap_threshold": float(hospital_gap_threshold),
        }
    )


def _get_results() -> dict[str, Any] | None:
    results = _ctx.get("fl_results")
    return results if isinstance(results, dict) else None


# ============================================================
# Correction-call validation
# ============================================================

def is_valid_correction_call(tool_call: dict[str, Any]) -> bool:
    """
    Validate a correction call before routing it to human approval.

    Invalid correction calls are sent to ToolNode so the tool can return a
    structured validation error to the LLM.
    """
    name = str(tool_call.get("name", ""))
    args = tool_call.get("args", {}) or {}

    if not isinstance(args, dict):
        return False

    if name == "run_fedprox":
        try:
            mu = float(args.get("mu", 0.05))
        except (TypeError, ValueError):
            return False

        return 0.001 <= mu <= 1.0

    if name == "reweight_data":
        hospital = str(args.get("hospital", ""))

        try:
            new_weight = float(args.get("new_weight"))
        except (TypeError, ValueError):
            return False

        return (
            hospital in {"sainte_justine", "sickkids"}
            and 0.0 < new_weight <= 1.0
        )

    return False


# ============================================================
# Tools
# ============================================================

@tool
def evaluate_fairness(
    hospital: Literal["sainte_justine", "sickkids", "all"] = "all",
) -> str:
    """
    Return subgroup MAE broken down by age group and gender.

    Args:
        hospital: "sainte_justine", "sickkids", or "all".

    Returns:
        JSON containing subgroup metrics, the worst returned subgroup, and all
        returned subgroups above the configured harm threshold.
    """
    results = _get_results()

    if results is None:
        return json.dumps(
            {"error": "No FL results available. Run Layer 2 first."},
            indent=2,
        )

    subgroup_metrics = results.get("subgroup_metrics", {})

    if not isinstance(subgroup_metrics, dict):
        return json.dumps(
            {"error": "subgroup_metrics is missing or malformed."},
            indent=2,
        )

    targets = (
        list(subgroup_metrics.keys())
        if hospital == "all"
        else [hospital]
    )

    report: dict[str, Any] = {}
    all_returned_maes: dict[str, float] = {}

    for site in targets:
        metrics = subgroup_metrics.get(site)

        if not isinstance(metrics, dict):
            report[site] = {"error": "hospital not found"}
            continue

        age_mae = {
            key.removeprefix("age_"): round(float(value), 3)
            for key, value in metrics.items()
            if key.startswith("age_")
        }

        gender_mae = {
            key.removeprefix("gender_"): round(float(value), 3)
            for key, value in metrics.items()
            if key.startswith("gender_")
        }

        report[site] = {
            "age_group_mae": age_mae,
            "gender_mae": gender_mae,
        }

        all_returned_maes.update(
            {
                f"{site}/{group}": mae
                for group, mae in age_mae.items()
            }
        )

        all_returned_maes.update(
            {
                f"{site}/gender/{gender}": mae
                for gender, mae in gender_mae.items()
            }
        )

    if all_returned_maes:
        threshold = float(_ctx.get("harm_threshold", 20.0))
        worst_name = max(
            all_returned_maes,
            key=all_returned_maes.get,
        )
        worst_mae = all_returned_maes[worst_name]

        report["worst_subgroup"] = {
            "subgroup": worst_name,
            "mae": round(worst_mae, 3),
            "harm_threshold": threshold,
            "flag": worst_mae > threshold,
        }

        report["subgroups_above_threshold"] = [
            {
                "subgroup": subgroup_name,
                "mae": round(mae, 3),
            }
            for subgroup_name, mae in sorted(
                all_returned_maes.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if mae > threshold
        ]

    return json.dumps(report, indent=2)


@tool
def compare_hospitals() -> str:
    """
    Compare overall MAE, RMSE, and R-squared across hospitals.

    Uses hospital-level metrics from Layer 2. It does not estimate overall
    performance by averaging subgroup metrics.
    """
    results = _get_results()

    if results is None:
        return json.dumps(
            {"error": "No FL results available. Run Layer 2 first."},
            indent=2,
        )

    final_metrics = results.get("final_metrics", {})

    if not isinstance(final_metrics, dict):
        return json.dumps(
            {"error": "final_metrics is missing or malformed."},
            indent=2,
        )

    table: dict[str, Any] = {}

    for hospital, metrics in final_metrics.items():
        if not isinstance(metrics, dict):
            continue

        table[hospital] = {
            "mae": round(
                float(metrics.get(f"{hospital}_mae", 0.0)),
                3,
            ),
            "rmse": round(
                float(metrics.get(f"{hospital}_rmse", 0.0)),
                3,
            ),
            "r2": round(
                float(metrics.get(f"{hospital}_r2", 0.0)),
                3,
            ),
        }

    if len(table) >= 2:
        hospital_names = list(table.keys())
        first_hospital = hospital_names[0]
        second_hospital = hospital_names[1]

        mae_gap = abs(
            table[first_hospital]["mae"]
            - table[second_hospital]["mae"]
        )

        threshold = float(
            _ctx.get("hospital_gap_threshold", 3.0)
        )

        table["summary"] = {
            "hospitals_compared": [
                first_hospital,
                second_hospital,
            ],
            "mae_gap": round(mae_gap, 3),
            "fairness_threshold": threshold,
            "fairness_flag": mae_gap > threshold,
            "strategy_used": results.get(
                "strategy",
                "unknown",
            ),
        }

    return json.dumps(table, indent=2)


@tool
def run_fedprox(mu: float = 0.05) -> str:
    """
    Propose switching the FL strategy to FedProx.

    Args:
        mu: Proximal penalty strength between 0.001 and 1.0.

    A valid proposal requires human approval. This tool proposes a correction;
    it does not retrain the model and does not guarantee improvement.
    """
    try:
        mu = float(mu)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": "mu must be numeric",
                "received_value": str(mu),
                "requires_human_approval": False,
            },
            indent=2,
        )

    if not 0.001 <= mu <= 1.0:
        return json.dumps(
            {
                "error": (
                    "mu must be between 0.001 and 1.0"
                ),
                "received_value": mu,
                "requires_human_approval": False,
            },
            indent=2,
        )

    current_strategy = getattr(
        _ctx.get("fl_config"),
        "strategy",
        "fedavg",
    )

    return json.dumps(
        {
            "correction_type": "fedprox",
            "description": (
                f"Switch FL strategy from "
                f"{current_strategy} to fedprox"
            ),
            "new_strategy": "fedprox",
            "fed_prox_mu": mu,
            "rationale": (
                "FedProx may reduce client drift. Its effect must "
                "be verified after retraining and reevaluation."
            ),
            "requires_human_approval": True,
        },
        indent=2,
    )


@tool
def reweight_data(
    hospital: Literal["sainte_justine", "sickkids"],
    new_weight: float,
) -> str:
    """
    Propose changing a hospital's aggregation importance weight.

    Args:
        hospital: "sainte_justine" or "sickkids".
        new_weight: New aggregation weight in the interval (0, 1].

    A valid proposal requires human approval. Its effect must be verified after
    retraining.
    """
    hospital = str(hospital)

    if hospital not in {
        "sainte_justine",
        "sickkids",
    }:
        return json.dumps(
            {
                "error": (
                    "hospital must be 'sainte_justine' "
                    "or 'sickkids'"
                ),
                "received_value": hospital,
                "requires_human_approval": False,
            },
            indent=2,
        )

    try:
        new_weight = float(new_weight)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "error": "new_weight must be numeric",
                "received_value": str(new_weight),
                "requires_human_approval": False,
            },
            indent=2,
        )

    if not 0.0 < new_weight <= 1.0:
        return json.dumps(
            {
                "error": (
                    "new_weight must be greater than 0 "
                    "and no more than 1"
                ),
                "received_value": new_weight,
                "requires_human_approval": False,
            },
            indent=2,
        )

    current_weights = getattr(
        _ctx.get("fl_config"),
        "client_weights",
        {},
    )

    if not isinstance(current_weights, dict):
        current_weights = {}

    try:
        old_weight = float(
            current_weights.get(hospital, 1.0)
        )
    except (TypeError, ValueError):
        old_weight = 1.0

    return json.dumps(
        {
            "correction_type": "reweight",
            "description": (
                f"Change {hospital} aggregation weight "
                f"from {old_weight:.2f} to "
                f"{new_weight:.2f}"
            ),
            "hospital": hospital,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "rationale": (
                "Changing the client's aggregation influence may "
                "reduce propagation of site-specific bias. The "
                "result must be verified after retraining."
            ),
            "requires_human_approval": True,
        },
        indent=2,
    )


@tool
def generate_report() -> str:
    """
    Generate a structured fairness audit report for the current FL run.

    Use only when the clinician explicitly requests a report, audit report,
    formal report, or structured summary.
    """
    results = _get_results()

    if results is None:
        return (
            "ERROR: No FL results available. "
            "Run Layer 2 first."
        )

    final_metrics = results.get(
        "final_metrics",
        {},
    )
    subgroup_metrics = results.get(
        "subgroup_metrics",
        {},
    )
    round_mae = results.get(
        "round_mae",
        [],
    )

    harm_threshold = float(
        _ctx.get("harm_threshold", 20.0)
    )
    hospital_gap_threshold = float(
        _ctx.get("hospital_gap_threshold", 3.0)
    )

    lines = [
        "FAIRNESS AUDIT REPORT",
        "=====================",
        (
            "FL strategy: "
            f"{str(results.get('strategy', 'unknown')).upper()}"
        ),
        (
            "Rounds: "
            f"{len(round_mae) if isinstance(round_mae, list) else 0}"
        ),
        (
            "Subgroup harm threshold: "
            f"{harm_threshold:.3f} bpm"
        ),
        (
            "Hospital MAE-gap threshold: "
            f"{hospital_gap_threshold:.3f} bpm"
        ),
        "",
        "Hospital metrics",
        "----------------",
    ]

    if isinstance(final_metrics, dict):
        for hospital, metrics in final_metrics.items():
            if not isinstance(metrics, dict):
                continue

            mae = float(
                metrics.get(f"{hospital}_mae", 0.0)
            )
            rmse = float(
                metrics.get(f"{hospital}_rmse", 0.0)
            )
            r2 = float(
                metrics.get(f"{hospital}_r2", 0.0)
            )

            lines.append(
                f"{hospital}: "
                f"MAE={mae:.3f}, "
                f"RMSE={rmse:.3f}, "
                f"R²={r2:.3f}"
            )

    lines.extend(
        [
            "",
            "Top 5 age-group harms",
            "---------------------",
        ]
    )

    age_harms: dict[str, float] = {}

    if isinstance(subgroup_metrics, dict):
        age_harms = {
            (
                f"{hospital}/"
                f"{key.removeprefix('age_')}"
            ): float(value)
            for hospital, metrics
            in subgroup_metrics.items()
            if isinstance(metrics, dict)
            for key, value in metrics.items()
            if key.startswith("age_")
        }

    for label, mae in sorted(
        age_harms.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]:
        flag = (
            "FLAG"
            if mae > harm_threshold
            else "OK"
        )
        lines.append(
            f"{label}: MAE={mae:.3f} [{flag}]"
        )

    lines.extend(
        [
            "",
            "Gender fairness",
            "---------------",
        ]
    )

    if isinstance(subgroup_metrics, dict):
        for hospital, metrics in subgroup_metrics.items():
            if not isinstance(metrics, dict):
                continue

            male_mae = metrics.get("gender_M")
            female_mae = metrics.get("gender_F")

            if (
                male_mae is None
                or female_mae is None
            ):
                continue

            gap = abs(
                float(male_mae)
                - float(female_mae)
            )

            lines.append(
                f"{hospital}: "
                f"M={float(male_mae):.3f}, "
                f"F={float(female_mae):.3f}, "
                f"gap={gap:.3f}"
            )

    return "\n".join(lines)


TOOLS = [
    evaluate_fairness,
    compare_hospitals,
    run_fedprox,
    reweight_data,
    generate_report,
]

CORRECTION_TOOLS = {
    "run_fedprox",
    "reweight_data",
}


# ============================================================
# Agent state and prompt
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_correction: Optional[dict[str, Any]]
    approved: bool
    correction_log: list[dict[str, Any]]


SYSTEM_PROMPT = """
You are a clinical AI fairness analyst for a pediatric federated learning system.

Your role is to help clinicians inspect model fairness and propose carefully
governed mitigations.You may propose a retraining configuration, but you do not apply 
    the configuration or retrain the model directly.

Tool-use rules:
- If the user is NOT requesting fairness information,
DO NOT call any tool.
- Do not call tools for greetings, thanks, farewells, capability questions,
  requests to wait, or other casual conversation.
- Call at most one tool in each assistant turn.
- Never repeat the same tool call for the same purpose.
- After receiving a tool result, decide whether another tool is truly needed.
- Use evaluate_fairness for age-group, gender, subgroup, harm-threshold,
  worst-subgroup, or subgroup-count questions.
- Use compare_hospitals for hospital-level MAE, RMSE, R-squared, site
  comparison, or hospital MAE-gap questions.
- Never calculate overall hospital metrics by averaging subgroup metrics.
- Use generate_report only when the clinician explicitly requests a report,
  audit report, formal report, or structured report.

Mitigation and governance rules:
- Before proposing a mitigation, inspect current fairness evidence with the
  appropriate analysis tool, unless that evidence already appears in a tool
  result earlier in the same conversation.
- When a mitigation is requested, propose exactly one mitigation.
- Valid mitigation calls require human approval before execution.
- Do not say a correction was executed merely because it was proposed or
  approved.
- Do not claim that FedProx or reweighting will improve fairness. State that
  the result must be verified after retraining and reevaluation.
- Do not claim clinical safety or deployment readiness from these fairness
  metrics alone.
- Do not invent tools, subgroup-specific reweighting capabilities, metrics,
  or evidence that are not present in tool outputs.

Answer rules:
- Base numeric and factual conclusions on tool outputs.
- Answer the exact question asked.
- Be clear and concise.
"""


# ============================================================
# Graph nodes and routing
# ============================================================

def make_agent_node(llm_with_tools: Any):
    """Create the LangGraph LLM node."""

    def agent_node(
        state: AgentState,
    ) -> dict[str, list[AIMessage]]:
        messages = state["messages"]

        request_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *[
                message
                for message in messages
                if not isinstance(
                    message,
                    SystemMessage,
                )
            ],
        ]

        response = llm_with_tools.invoke(
            request_messages
        )

        return {
            "messages": [response],
        }

    return agent_node


def should_continue(
    state: AgentState,
) -> Literal["tools", "approval", "end"]:
    """
    Route the latest assistant message.

    - no tool call -> END
    - valid correction call -> approval
    - analysis or invalid correction call -> ToolNode
    """
    last_message = state["messages"][-1]

    if not isinstance(
        last_message,
        AIMessage,
    ):
        return "end"

    tool_calls = (
        getattr(
            last_message,
            "tool_calls",
            None,
        )
        or []
    )

    if not tool_calls:
        return "end"

    correction_calls = [
        current_call
        for current_call in tool_calls
        if current_call.get("name")
        in CORRECTION_TOOLS
    ]

    if (
        correction_calls
        and any(
            is_valid_correction_call(
                current_call
            )
            for current_call
            in correction_calls
        )
    ):
        return "approval"

    return "tools"


def human_approval_node(
    state: AgentState,
) -> dict[str, Any]:
    """
    Pause on the first valid correction call.

    This node does not execute or apply the correction.
    """
    last_message = state["messages"][-1]

    if not isinstance(
        last_message,
        AIMessage,
    ):
        return {
            "messages": [],
        }

    all_calls = (
        getattr(
            last_message,
            "tool_calls",
            None,
        )
        or []
    )

    valid_corrections = [
        current_call
        for current_call in all_calls
        if (
            current_call.get("name")
            in CORRECTION_TOOLS
            and is_valid_correction_call(
                current_call
            )
        )
    ]

    if not valid_corrections:
        return {
            "messages": [],
        }

    selected_call = valid_corrections[0]

    human_response = interrupt(
        {
            "prompt": (
                "The agent proposes an FL correction.\n"
                f"Tool: {selected_call['name']}\n"
                "Arguments: "
                f"{json.dumps(selected_call.get('args', {}), indent=2)}\n\n"
                "Approve this proposal? Type yes or no."
            ),
            "tool_name": selected_call["name"],
            "args": selected_call.get(
                "args",
                {},
            ),
        }
    )

    approved = (
        str(human_response)
        .strip()
        .lower()
        in {
            "yes",
            "y",
            "approve",
            "approved",
            "ok",
        }
    )

    selected_outcome = (
        (
            "APPROVED — the proposal may be "
            "applied by the pipeline before "
            "retraining. No retraining has "
            "occurred yet."
        )
        if approved
        else (
            "REJECTED — the clinician declined "
            "this proposal. Do not claim that "
            "the correction was applied."
        )
    )

    tool_messages: list[ToolMessage] = [
        ToolMessage(
            content=selected_outcome,
            tool_call_id=selected_call["id"],
            name=selected_call["name"],
        )
    ]

    # Satisfy the tool-call protocol if a model ignored the
    # one-tool-per-turn instruction.
    for other_call in all_calls:
        if (
            other_call.get("id")
            == selected_call.get("id")
        ):
            continue

        tool_messages.append(
            ToolMessage(
                content=(
                    "IGNORED — only one tool call "
                    "is processed per assistant "
                    "turn. Call this tool again "
                    "separately if it is still needed."
                ),
                tool_call_id=other_call["id"],
                name=other_call.get("name"),
            )
        )

    log_entry = {
        "tool": selected_call["name"],
        "args": selected_call.get(
            "args",
            {},
        ),
        "approved": approved,
    }

    pending_correction = (
        {
            "tool": selected_call["name"],
            "args": selected_call.get(
                "args",
                {},
            ),
        }
        if approved
        else None
    )

    return {
        "messages": tool_messages,
        "pending_correction": (
            pending_correction
        ),
        "approved": approved,
        "correction_log": [
            *state.get(
                "correction_log",
                [],
            ),
            log_entry,
        ],
    }


# ============================================================
# Graph construction
# ============================================================

def build_agent_graph(
    cfg: AgentConfig = AGENT_CFG,
):
    """Compile and return the fairness-agent graph."""
    llm = ChatOpenAI(
        model=cfg.vllm_model,
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        temperature=cfg.temperature,
        max_retries=2,
        timeout=120,
    )

    llm_with_tools = llm.bind_tools(
        TOOLS
    )

    graph = StateGraph(AgentState)

    graph.add_node(
        "agent",
        make_agent_node(llm_with_tools),
    )
    graph.add_node(
        "tools",
        ToolNode(TOOLS),
    )
    graph.add_node(
        "approval",
        human_approval_node,
    )

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "approval": "approval",
            "end": END,
        },
    )

    graph.add_edge(
        "tools",
        "agent",
    )
    graph.add_edge(
        "approval",
        "agent",
    )

    return graph.compile(
        checkpointer=MemorySaver()
    )


# ============================================================
# Console helpers
# ============================================================

def _message_to_text(
    message: AIMessage,
) -> str:
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            (
                block.get("text", "")
                if isinstance(block, dict)
                else str(block)
            )
            for block in content
        )

    return str(content or "")


def _print_step(
    step: dict[str, Any],
) -> None:
    messages = step.get(
        "messages",
        [],
    )

    if not messages:
        return

    last_message = messages[-1]

    if isinstance(
        last_message,
        AIMessage,
    ):
        text = _message_to_text(
            last_message
        ).strip()

        if text:
            suffix = (
                "..."
                if len(text) > 400
                else ""
            )
            print(
                f"\n[Agent] "
                f"{text[:400]}"
                f"{suffix}"
            )

        tool_calls = (
            getattr(
                last_message,
                "tool_calls",
                None,
            )
            or []
        )

        if tool_calls:
            print(
                "\n[Tool calls] "
                + json.dumps(
                    tool_calls,
                    indent=2,
                )
            )

    elif isinstance(
        last_message,
        ToolMessage,
    ):
        content = str(
            last_message.content
        )
        suffix = (
            "..."
            if len(content) > 300
            else ""
        )

        print(
            f"\n[Tool result] "
            f"{content[:300]}"
            f"{suffix}"
        )


def _check_interrupt(
    app: Any,
    graph_config: dict[str, Any],
) -> Optional[dict[str, Any]]:
    snapshot = app.get_state(
        graph_config
    )

    for task in (
        getattr(
            snapshot,
            "tasks",
            (),
        )
        or ()
    ):
        interrupts = getattr(
            task,
            "interrupts",
            None,
        )

        if interrupts:
            return interrupts[0].value

    return None


# ============================================================
# Public entry point
# ============================================================

def run_layer3(
    fl_results: dict[str, Any],
    clients: dict[str, Any],
    fl_config: Any,
    initial_query: str = (
        "Analyse the fairness of the current "
        "federated learning model. Check all "
        "hospitals and age groups. Identify "
        "the most harmed subgroup and propose "
        "one concrete correction."
    ),
    cfg: AgentConfig = AGENT_CFG,
) -> dict[str, Any]:
    """
    Run the console-based Layer 3 workflow.

    Approved proposals are returned to the caller. This function does not
    modify FL configuration or rerun Layer 2.
    """
    set_context(
        fl_results=fl_results,
        clients=clients,
        fl_config=fl_config,
        harm_threshold=(
            cfg.harm_mae_threshold
        ),
        hospital_gap_threshold=(
            cfg.hospital_gap_threshold
        ),
    )

    app = build_agent_graph(cfg)

    graph_config = {
        "configurable": {
            "thread_id": (
                "fl-fairness-"
                f"{uuid.uuid4().hex}"
            ),
        },
        "recursion_limit": max(
            20,
            cfg.max_iterations * 4,
        ),
    }

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=initial_query
            )
        ],
        "pending_correction": None,
        "approved": False,
        "correction_log": [],
    }

    print("\n" + "─" * 56)
    print(
        "LAYER 3 — "
        "Agentic Fairness Analysis"
    )
    print("─" * 56)
    print(f"Model: {cfg.vllm_model}")
    print(f"Query: {initial_query}")

    final_state: dict[str, Any] = {}

    for step in app.stream(
        initial_state,
        config=graph_config,
        stream_mode="values",
    ):
        final_state = step
        _print_step(step)

    approval_count = 0

    while (
        approval_count
        < cfg.max_iterations
    ):
        interrupt_payload = (
            _check_interrupt(
                app,
                graph_config,
            )
        )

        if interrupt_payload is None:
            break

        print("\n" + "─" * 56)
        print("HUMAN APPROVAL REQUIRED")
        print("─" * 56)
        print(
            interrupt_payload.get(
                "prompt",
                str(interrupt_payload),
            )
        )

        while True:
            decision = input(
                "\nDecision (yes/no): "
            ).strip().lower()

            if decision in {
                "yes",
                "y",
                "approve",
                "approved",
                "ok",
                "no",
                "n",
                "reject",
                "rejected",
            }:
                break

            print(
                "Please enter yes or no."
            )

        for step in app.stream(
            Command(resume=decision),
            config=graph_config,
            stream_mode="values",
        ):
            final_state = step
            _print_step(step)

        approval_count += 1

    correction_log = final_state.get(
        "correction_log",
        [],
    )

    approved_corrections = [
        entry
        for entry in correction_log
        if entry.get("approved")
    ]

    return {
        "correction_log": correction_log,
        "approved_corrections": (
            approved_corrections
        ),
        "final_state": final_state,
    }
