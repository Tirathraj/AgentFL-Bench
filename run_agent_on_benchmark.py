from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from config import AGENT_CFG, FL_CFG
from layer3_agent import (
    build_agent_graph,
    set_context,
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {path} "
                    "must contain a JSON object."
                )

            rows.append(row)

    return rows


def load_frozen_context(
    path: Path,
) -> tuple[dict[str, Any], Any]:
    """
    Load the frozen FL results and reconstruct the FLConfig
    expected by the Layer 3 tools.
    """
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if "frozen_fl_results" not in payload:
        raise KeyError(
            "The context file must contain "
            "'frozen_fl_results'. Use the full "
            "benchmark_context_full.json file."
        )

    fl_results = payload["frozen_fl_results"]

    if not isinstance(fl_results, dict):
        raise TypeError(
            "'frozen_fl_results' must be a JSON object."
        )

    fl_cfg = copy.deepcopy(FL_CFG)

    cfg_data = payload.get(
        "federated_learning_configuration",
        {},
    )

    field_names = [
        "strategy",
        "n_rounds",
        "local_epochs",
        "learning_rate",
        "fed_prox_mu",
        "client_weights",
        "personalized_finetune_epochs",
        "test_size",
        "random_state",
    ]

    for field_name in field_names:
        if (
            field_name in cfg_data
            and hasattr(fl_cfg, field_name)
        ):
            setattr(
                fl_cfg,
                field_name,
                copy.deepcopy(cfg_data[field_name]),
            )

    return fl_results, fl_cfg


def normalize_tool_call(
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a LangChain tool call to the benchmark format.
    """
    arguments = tool_call.get(
        "args",
        tool_call.get("arguments", {}),
    )

    if not isinstance(arguments, dict):
        arguments = {}

    return {
        "name": str(tool_call.get("name", "")),
        "arguments": arguments,
    }


def message_text(message: AIMessage) -> str:
    """
    Convert AIMessage content to plain text.
    """
    content = message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
            else:
                text = str(block)

            if text:
                parts.append(str(text))

        return " ".join(parts).strip()

    if content:
        return str(content).strip()

    return ""


def extract_run_result(
    app: Any,
    graph_config: dict[str, Any],
    invoke_result: Any,
) -> dict[str, Any]:
    """
    Extract ordered tool calls, tool outputs, final answer,
    and approval-interrupt status.
    """
    snapshot = app.get_state(graph_config)

    snapshot_values = (
        getattr(snapshot, "values", None)
        or {}
    )

    messages = snapshot_values.get(
        "messages",
        [],
    )

    # Fallback for LangGraph versions that return state
    # directly from invoke().
    if (
        not messages
        and isinstance(invoke_result, dict)
    ):
        messages = invoke_result.get(
            "messages",
            [],
        )

    ordered_calls: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, Any]] = []

    tool_name_by_id: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    seen_tool_messages: set[tuple[str, str]] = set()

    final_answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            calls = (
                getattr(message, "tool_calls", None)
                or []
            )

            for current_call in calls:
                call_id = str(
                    current_call.get("id", "")
                )

                # Checkpointed state can occasionally expose
                # a message more than once.
                if call_id and call_id in seen_call_ids:
                    continue

                if call_id:
                    seen_call_ids.add(call_id)

                normalized_call = normalize_tool_call(
                    current_call
                )

                ordered_calls.append(
                    normalized_call
                )

                if call_id:
                    tool_name_by_id[call_id] = (
                        normalized_call["name"]
                    )

            # Only no-tool assistant messages qualify as
            # the final natural-language response.
            text = message_text(message)

            if text and not calls:
                final_answer = text

        elif isinstance(message, ToolMessage):
            tool_call_id = str(
                getattr(
                    message,
                    "tool_call_id",
                    "",
                )
                or ""
            )

            content = str(message.content)

            message_key = (
                tool_call_id,
                content,
            )

            if message_key in seen_tool_messages:
                continue

            seen_tool_messages.add(message_key)

            tool_name = tool_name_by_id.get(
                tool_call_id
            )

            if not tool_name:
                tool_name = getattr(
                    message,
                    "name",
                    None,
                )

            tool_outputs.append(
                {
                    "name": tool_name,
                    "content": content,
                }
            )

    approval_requested = False

    for task in (
        getattr(snapshot, "tasks", ())
        or ()
    ):
        interrupts = getattr(
            task,
            "interrupts",
            None,
        )

        if interrupts:
            approval_requested = True
            break

    if not approval_requested:
        next_nodes = (
            getattr(snapshot, "next", ())
            or ()
        )

        approval_requested = (
            "approval" in next_nodes
        )

    return {
        "tool_calls": ordered_calls,
        "tool_outputs": tool_outputs,
        "final_answer": final_answer,
        "approval_requested": approval_requested,
    }


def safe_slug(value: str) -> str:
    value = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        value,
    )

    return value.strip("-") or "model"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PICU fairness LangGraph agent "
            "on the gold benchmark."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Path to benchmark.jsonl",
    )

    parser.add_argument(
        "--context",
        required=True,
        type=Path,
        help=(
            "Path to benchmark_context_full.json "
            "containing frozen_fl_results."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model",
        default=AGENT_CFG.vllm_model,
        help=(
            "Exact model ID returned by the "
            "vLLM /v1/models endpoint."
        ),
    )

    parser.add_argument(
        "--run-id",
        default="1",
    )

    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    benchmark_cases = load_jsonl(
        args.benchmark
    )

    frozen_results, frozen_fl_cfg = (
        load_frozen_context(args.context)
    )

    # Current tools do not use client objects.
    set_context(
        fl_results=frozen_results,
        clients={},
        fl_config=frozen_fl_cfg,
    )

    agent_cfg = copy.deepcopy(AGENT_CFG)
    agent_cfg.vllm_model = args.model
    agent_cfg.temperature = 0.0

    model_slug = safe_slug(args.model)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, case in enumerate(
            benchmark_cases,
            start=1,
        ):
            case_id = str(case["id"])
            question = str(case["question"])

            # New graph and new MemorySaver for every case.
            app = build_agent_graph(agent_cfg)

            graph_config = {
                "configurable": {
                    "thread_id": (
                        f"benchmark-{model_slug}-"
                        f"{args.run_id}-{case_id}"
                    )
                },
                "recursion_limit": (
                    args.recursion_limit
                ),
            }

            initial_state = {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ],
                "pending_correction": None,
                "approved": False,
                "correction_log": [],
            }

            started = time.perf_counter()
            error: str | None = None

            try:
                invoke_result = app.invoke(
                    initial_state,
                    config=graph_config,
                )

                extracted = extract_run_result(
                    app=app,
                    graph_config=graph_config,
                    invoke_result=invoke_result,
                )

            except Exception as exc:
                extracted = {
                    "tool_calls": [],
                    "tool_outputs": [],
                    "final_answer": "",
                    "approval_requested": False,
                }

                error = (
                    f"{type(exc).__name__}: {exc}"
                )

            elapsed = (
                time.perf_counter() - started
            )

            row = {
                "id": case_id,
                "model": args.model,
                "run_id": str(args.run_id),
                "question": question,
                "tool_calls": (
                    extracted["tool_calls"]
                ),
                "tool_outputs": (
                    extracted["tool_outputs"]
                ),
                "final_answer": (
                    extracted["final_answer"]
                ),
                "approval_requested": (
                    extracted[
                        "approval_requested"
                    ]
                ),
                "error": error,
                "latency_seconds": round(
                    elapsed,
                    6,
                ),
            }

            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            status = error or "ok"

            print(
                f"[{index:03d}/"
                f"{len(benchmark_cases):03d}] "
                f"{case_id}: {status}"
            )

    print(
        f"\nPredictions written to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()