"""
    Project: PICU Federated Learning Fairness Pipeline
    Author: Alexander Mazur
    Date: 2026_05_27

    ───────
Streamlit UI for the PICU Federated Learning Fairness Pipeline.

Six-step workflow:
  1. Doctor asks a question
  2. Agent analyzes FL results
  3. Agent detects unfairness across hospitals / subgroups
  4. Agent suggests corrections (FedProx, reweighting)
  5. Doctor approves → model retrains automatically
  6. Updated results and audit report are displayed

Run:
    streamlit run app.py

Requirements:
    pip install -r requirements.txt
    ollama serve
    ollama pull llama3.2:3b
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))

from config import SIM_CFG, FL_CFG, AGENT_CFG, AGE_GROUPS, FEATURES, FLConfig
from layer1_simulation import run_layer1, HospitalData
from layer2_federated import FLClient, FedAvgServer, _metrics
from layer3_agent import set_context, TOOLS, CORRECTION_TOOLS, SYSTEM_PROMPT

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI


# ══════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "PICU FL Fairness Dashboard",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────

st.markdown("""
<style>
/* Metric cards */
.kpi-card{background:var(--background-color,#f8f9fa);border-radius:10px;
  padding:16px 20px;border-left:4px solid #4e8df5;margin:6px 0}
.kpi-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;
  color:#6b7280;margin-bottom:4px}
.kpi-value{font-size:1.6rem;font-weight:700;line-height:1}

/* Step badges */
.step-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
  border-radius:20px;font-size:.78rem;font-weight:600;margin:2px 0}
.step-done{background:#d1fae5;color:#065f46}
.step-active{background:#dbeafe;color:#1e40af}
.step-pending{background:#f3f4f6;color:#6b7280}

/* Correction card */
.correction-card{background:#fffbeb;border:2px solid #f59e0b;border-radius:12px;
  padding:16px;margin:12px 0}
.correction-title{font-weight:700;color:#92400e;font-size:1rem;margin-bottom:8px}

/* Chat tool result */
.tool-result{background:#f0fdf4;border-left:3px solid #22c55e;border-radius:0 8px 8px 0;
  padding:10px 14px;margin:4px 0;font-size:.82rem}

/* Header strip */
.header-strip{display:flex;align-items:center;gap:12px;padding:8px 0 20px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Session-state defaults
# ══════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    # Pipeline data
    "hospitals":           None,   # Layer 1 output
    "clients":             None,   # Layer 2 FLClient objects
    "server":              None,   # Layer 2 FedAvgServer
    "fl_results":          None,   # Layer 2 metrics dict
    "fl_results_initial":  None,   # Snapshot before any correction
    "fl_results_latest":   None,   # Latest results after correction
    "round_maes":          [],     # Live training MAE history

    # FL config (mutable copy)
    "fl_cfg":              copy.deepcopy(FL_CFG),
    "importance_weights":  None,

    # Agent state
    "lang_messages":       [],     # LangChain message objects
    "chat_messages":       [],     # Display list [{role, content, name?}]
    "pending_correction":  None,   # Tool call dict awaiting approval
    "correction_log":      [],     # Full history of proposed corrections
    "agent_iter":          0,      # Guard against infinite loops
    "agent_state":         "idle", # idle|thinking|awaiting_approval|done|error

    # Pipeline phases
    "phase":               "idle", # idle|data|training|trained|analyzing|retrained
    "correction_round":    0,

    # Sidebar model field
    "agent_model":         AGENT_CFG.vllm_model,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = copy.deepcopy(_v) if isinstance(_v, (dict, list)) else _v

st.session_state.agent_model = AGENT_CFG.vllm_model
# ══════════════════════════════════════════════════════════════════
#  LLM helper
# ══════════════════════════════════════════════════════════════════
def synchronize_agent_model() -> None:
    configured_model = AGENT_CFG.vllm_model

    if st.session_state.get("agent_model") != configured_model:
        st.session_state.agent_model = configured_model
        get_base_llm.clear()
        get_tool_llm.clear()
@st.cache_resource(show_spinner=False)
def get_base_llm(model: str):
    """LLM without tools, used for greetings and general conversation."""
    return ChatOpenAI(
        model=model,
        base_url=AGENT_CFG.vllm_base_url,
        api_key=AGENT_CFG.vllm_api_key,
        temperature=AGENT_CFG.temperature,
        max_retries=2,
        timeout=120,
    )


@st.cache_resource(show_spinner=False)
def get_tool_llm(model: str):
    """LLM with fairness tools, used for model-analysis requests."""
    llm = ChatOpenAI(
        model=model,
        base_url=AGENT_CFG.vllm_base_url,
        api_key=AGENT_CFG.vllm_api_key,
        temperature=AGENT_CFG.temperature,
        max_retries=2,
        timeout=120,
    )
    return llm.bind_tools(TOOLS)


def ollama_online() -> bool:
    try:
        import httpx
        r = httpx.get(f"{AGENT_CFG.ollama_base_url}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  Tool execution
# ══════════════════════════════════════════════════════════════════

_TOOL_MAP = {t.name: t for t in TOOLS}


def call_tool(name: str, args: dict) -> str:
    if name not in _TOOL_MAP:
        return f"Unknown tool: {name}"
    try:
        return str(_TOOL_MAP[name].invoke(args))
    except Exception as e:
        return f"Tool error ({name}): {e}"


# ══════════════════════════════════════════════════════════════════
#  Layer 1 — data simulation
# ══════════════════════════════════════════════════════════════════

def do_layer1() -> None:
    with st.spinner("Loading Sainte-Justine data and simulating SickKids..."):
        hospitals = run_layer1(SIM_CFG)
    st.session_state.hospitals = hospitals
    st.session_state.phase     = "data"
    st.toast("✅ Data ready", icon="📊")


# ══════════════════════════════════════════════════════════════════
#  Layer 2 — FL training with live progress
# ══════════════════════════════════════════════════════════════════

def do_layer2(
    importance_weights: dict | None = None,
    label: str = "",
) -> None:
    """Run FL training, updating the UI live per round."""
    hospitals = st.session_state.hospitals
    cfg       = st.session_state.fl_cfg

    if importance_weights is None:
        importance_weights = copy.deepcopy(cfg.client_weights)

    clients = {name: FLClient(hd, cfg) for name, hd in hospitals.items()}
    server  = FedAvgServer(n_features=len(FEATURES))

    client_sizes   = {n: c.hospital.n_train for n, c in clients.items()}
    proximal_mu    = cfg.fed_prox_mu if cfg.strategy == "fedprox" else 0.0
    global_weights = server.global_model.get_weights()

    # ── Live progress widgets ──────────────────────────────────
    heading = f"**Training** ({cfg.strategy.upper()}{label})"
    st.markdown(heading)
    progress    = st.progress(0, text="Round 1 …")
    col_chart, col_kpi = st.columns([3, 1])
    chart_slot  = col_chart.empty()
    kpi_slot    = col_kpi.empty()
    round_maes: list[float] = []

    for rnd in range(1, cfg.n_rounds + 1):

        # ── One FL round ──────────────────────────────────────
        local_w = {
            name: client.train(global_weights=global_weights, proximal_mu=proximal_mu)
            for name, client in clients.items()
        }
        global_weights = server.aggregate(local_w, client_sizes, importance_weights)
        for c in clients.values():
            c.model.set_weights(global_weights)

        mae = float(np.mean([
            mean_absolute_error(c.y_test, c.model.predict(c.X_test))
            for c in clients.values()
        ]))
        round_maes.append(mae)

        # ── Live MAE chart ────────────────────────────────────
        fig = go.Figure(go.Scatter(
            x=list(range(1, rnd + 1)), y=round_maes,
            mode="lines+markers",
            line=dict(color="#4e8df5", width=2),
            marker=dict(size=5, color="#4e8df5"),
            fill="tozeroy", fillcolor="rgba(78,141,245,0.08)",
        ))
        fig.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0),
            xaxis_title="Round", yaxis_title="MAE (bpm)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        chart_slot.plotly_chart(fig, use_container_width=True, key=f"live_mae_rnd_{rnd}")

        with kpi_slot.container():
            st.metric("Round",   f"{rnd}/{cfg.n_rounds}")
            st.metric("Avg MAE", f"{mae:.2f} bpm")

        progress.progress(rnd / cfg.n_rounds, text=f"Round {rnd}/{cfg.n_rounds}")

    # ── Personalized fine-tune ─────────────────────────────────
    if cfg.strategy == "personalized":
        with st.spinner("Fine-tuning local models …"):
            for c in clients.values():
                c.fine_tune(global_weights)

    # ── Evaluate ──────────────────────────────────────────────
    final_metrics    = {}
    subgroup_metrics = {}
    for name, client in clients.items():
        final_metrics[name]    = client.evaluate()
        subgroup_metrics[name] = client.evaluate_subgroups()

    fl_results = {
        "round_mae":        round_maes,
        "final_metrics":    final_metrics,
        "subgroup_metrics": subgroup_metrics,
        "global_weights":   global_weights,
        "strategy":         cfg.strategy,
    }

    # ── Persist to session state ───────────────────────────────
    st.session_state.clients     = clients
    st.session_state.server      = server
    st.session_state.fl_results  = fl_results
    st.session_state.round_maes  = round_maes

    if st.session_state.fl_results_initial is None:
        st.session_state.fl_results_initial = fl_results

    st.session_state.fl_results_latest = fl_results
    st.session_state.phase = "trained"

    set_context(fl_results, clients, st.session_state.fl_cfg)
    progress.empty()


# ══════════════════════════════════════════════════════════════════
#  Agent — one reasoning step
# ══════════════════════════════════════════════════════════════════

MAX_AGENT_ITERS = 12

def is_general_conversation(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())

    exact_messages = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "how are you",
        "nice to meet you",
        "please wait a moment",
        "i am just testing the chat",
        "bye",
        "bye for now",
        "who are you",
        "what can you do",
        "what can you help me with",
        "tell me briefly what this assistant does",
        "can you explain your role",
    }

    return normalized.rstrip(".?!") in {
        item.rstrip(".?!")
        for item in exact_messages
    }
def agent_step() -> None:
    if st.session_state.agent_iter >= MAX_AGENT_ITERS:
        st.session_state.agent_state = "done"
        return

    st.session_state.agent_iter += 1

    # Keep only conversation messages in session state.
    history = [
        message
        for message in st.session_state.lang_messages
        if not isinstance(message, SystemMessage)
    ]

    # Always send the current system prompt.
    request_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
    ]

    # Retrieve the most recent clinician message.
    last_user_text = ""
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            last_user_text = str(message.content)
            break

    # Decide whether tools should be available for this turn.
    if is_general_conversation(last_user_text):
        llm = get_base_llm(st.session_state.agent_model)
    else:
        llm = get_tool_llm(st.session_state.agent_model)

    try:
        response = llm.invoke(request_messages)

    except Exception as e:
        import traceback
        traceback.print_exc()

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ LLM error: {type(e).__name__}: {e}",
            }
        )
        st.session_state.agent_state = "error"
        return

    # Save the response, but do not permanently save the system prompt.
    st.session_state.lang_messages = history
    st.session_state.lang_messages.append(response)

    msgs = st.session_state.lang_messages

    raw = response.content or ""
    if isinstance(raw, list):
        raw = " ".join(
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in raw
        )

    if raw.strip():
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": raw.strip(),
            }
        )

    tool_calls = getattr(response, "tool_calls", None) or []

    if not tool_calls:
        st.session_state.agent_state = "done"
        return

    analysis_calls = [
        tc for tc in tool_calls
        if tc["name"] not in CORRECTION_TOOLS
    ]

    correction_calls = [
        tc for tc in tool_calls
        if tc["name"] in CORRECTION_TOOLS
    ]

    for tc in analysis_calls:
        result = call_tool(tc["name"], tc["args"])

        msgs.append(
            ToolMessage(
                content=result,
                tool_call_id=tc["id"],
            )
        )

        st.session_state.chat_messages.append(
            {
                "role": "tool",
                "name": tc["name"],
                "content": result,
            }
        )

    if correction_calls:
        st.session_state.pending_correction = correction_calls[0]
        st.session_state.agent_state = "awaiting_approval"
        return

    st.session_state.agent_state = "thinking"


def start_agent(question: str) -> None:
    """Kick off a new agent turn with a doctor's question."""
    set_context(
        st.session_state.fl_results,
        st.session_state.clients,
        st.session_state.fl_cfg,
    )
    st.session_state.lang_messages.append(HumanMessage(content=question))
    st.session_state.chat_messages.append({"role": "user", "content": question})
    st.session_state.agent_state  = "thinking"
    st.session_state.agent_iter   = 0
    st.session_state.phase        = "analyzing"


def handle_approval(approved: bool) -> None:
    """Inject approval decision as a ToolMessage and resume the agent."""
    tc = st.session_state.pending_correction
    if tc is None:
        return
    verdict = (
        "✅ APPROVED — correction will be applied before retraining."
        if approved else
        "❌ REJECTED — please suggest a different correction."
    )
    st.session_state.lang_messages.append(
        ToolMessage(content=verdict, tool_call_id=tc["id"])
    )
    st.session_state.chat_messages.append(
        {"role": "tool", "name": tc["name"], "content": verdict}
    )
    st.session_state.correction_log.append(
        {"tool": tc["name"], "args": tc["args"], "approved": approved}
    )
    st.session_state.pending_correction = None
    st.session_state.agent_state = "thinking" if approved else "thinking"


def apply_correction_and_retrain(correction_entry: dict) -> None:
    """Apply one approved FL correction, then re-run Layer 2."""
    cfg  = copy.deepcopy(st.session_state.fl_cfg)
    tool_name = correction_entry.get("tool", "")
    args = correction_entry.get("args", {})

    if tool_name == "run_fedprox":
        cfg.strategy = "fedprox"
        cfg.fed_prox_mu = float(args.get("mu", 0.05))

    elif tool_name == "reweight_data":
        hospital = str(args.get("hospital", ""))
        new_weight = float(args.get("new_weight", 1.0))

        if hospital:
            cfg.client_weights[hospital] = new_weight

    st.session_state.fl_cfg = cfg
    st.session_state.correction_round += 1

    label = (
        f", round {st.session_state.correction_round}"
        f", {cfg.strategy.upper()}"
    )
    do_layer2(importance_weights=cfg.client_weights, label=label)
    st.session_state.phase = "retrained"

    # Re-sync context for the agent
    set_context(
        st.session_state.fl_results,
        st.session_state.clients,
        st.session_state.fl_cfg,
    )


# ══════════════════════════════════════════════════════════════════
#  Chart helpers
# ══════════════════════════════════════════════════════════════════

def _subgroup_chart(subgroup_metrics: dict) -> go.Figure:
    rows = []
    for hospital, metrics in subgroup_metrics.items():
        for k, v in metrics.items():
            if k.startswith("age_"):
                rows.append({"Hospital": hospital.replace("_", " ").title(),
                              "Age group": k.replace("age_", ""), "MAE": v})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="MAE", y="Age group", color="Hospital", orientation="h",
        barmode="group", height=360,
        color_discrete_map={
            "Sainte Justine": "#4e8df5",
            "Sickkids":       "#f5934e",
        },
    )
    fig.add_vline(x=20, line_dash="dash", line_color="#ef4444",
                  annotation_text="⚠ 20 bpm threshold",
                  annotation_position="top right")
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _gender_chart(subgroup_metrics: dict) -> go.Figure:
    hospitals = list(subgroup_metrics.keys())
    m_maes = [subgroup_metrics[h].get("gender_M", 0) for h in hospitals]
    f_maes = [subgroup_metrics[h].get("gender_F", 0) for h in hospitals]
    fig = go.Figure(data=[
        go.Bar(name="Male",   x=hospitals, y=m_maes, marker_color="#60a5fa"),
        go.Bar(name="Female", x=hospitals, y=f_maes, marker_color="#f472b6"),
    ])
    fig.update_layout(
        barmode="group", height=220,
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis_title="MAE (bpm)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def _before_after_chart(before: dict, after: dict) -> go.Figure:
    hospitals  = list(before["final_metrics"].keys())
    mae_before = [before["final_metrics"][h][f"{h}_mae"] for h in hospitals]
    mae_after  = [after["final_metrics"][h][f"{h}_mae"]  for h in hospitals]
    labels     = [h.replace("_", " ").title() for h in hospitals]
    fig = go.Figure(data=[
        go.Bar(name="Before correction", x=labels, y=mae_before,
               marker_color="#fca5a5"),
        go.Bar(name="After correction",  x=labels, y=mae_after,
               marker_color="#86efac"),
    ])
    fig.update_layout(
        barmode="group", height=240,
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis_title="MAE (bpm)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏥 PICU FL Fairness")
        st.caption("Pediatric Intensive Care Unit · Federated Learning")
        st.divider()

        # ── FL configuration ───────────────────────────────────
        st.markdown("### ⚙️ Federated Learning")
        strategy = st.selectbox(
            "Aggregation strategy",
            ["fedavg", "fedprox", "personalized"],
            index=["fedavg", "fedprox", "personalized"].index(
                st.session_state.fl_cfg.strategy
            ),
        )
        rounds = st.slider("Communication rounds", 3, 30,
                           st.session_state.fl_cfg.n_rounds)
        epochs = st.slider("Local SGD epochs", 10, 120,
                           st.session_state.fl_cfg.local_epochs, step=10)

        mu = st.session_state.fl_cfg.fed_prox_mu
        if strategy == "fedprox":
            mu = st.number_input(
                "FedProx μ (proximal penalty)",
                min_value=0.001, max_value=1.0,
                value=mu, step=0.01, format="%.3f",
                help="Higher μ = less client drift from the global model",
            )

        # Apply config changes
        cfg = st.session_state.fl_cfg
        cfg.strategy     = strategy
        cfg.n_rounds     = rounds
        cfg.local_epochs = epochs
        if strategy == "fedprox":
            cfg.fed_prox_mu = mu

        st.divider()

        # ── Agent configuration ────────────────────────────────
        st.markdown("### 🤖 Agent (vLLM)")
        model = st.text_input(
            "Model name",
            st.session_state.agent_model,
            help="e.g. gemma3:4b, gemma4:4b, llama3.2:3b",
        )
        st.session_state.agent_model = model

        def vllm_online() -> bool:
            try:
                import httpx
                r = httpx.get(f"{AGENT_CFG.vllm_base_url}/models", timeout=3.0)
                return r.status_code == 200
            except Exception:
                return False

        online = vllm_online()
        if online:
            st.success("🟢 vLLM connected")
        else:
            st.error("🔴 vLLM unreachable")

        st.divider()

        # ── Pipeline status ────────────────────────────────────
        st.markdown("### 📋 Pipeline status")
        phase = st.session_state.phase
        steps = [
            ("1 · Data simulated",   phase in ("data","training","trained","analyzing","retrained")),
            ("2 · FL model trained",  phase in ("trained","analyzing","retrained")),
            ("3 · Analysis run",      phase in ("analyzing","retrained")),
            ("4 · Model retrained",   phase == "retrained"),
        ]
        for label, done in steps:
            icon = "✅" if done else "⏳"
            st.markdown(f"{icon} {label}")

        st.divider()

        # ── Reset ──────────────────────────────────────────────
        if st.button("🔄 Reset everything", use_container_width=True, type="secondary"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
            st.rerun()


# ══════════════════════════════════════════════════════════════════
#  Tab 1 — Data & setup
# ══════════════════════════════════════════════════════════════════

def render_data_tab() -> None:
    st.subheader("Hospital data overview")

    if st.session_state.phase == "idle":
        st.info("Click **Load & Simulate Data** to run Layer 1.")
        if st.button("▶ Load & Simulate Data", type="primary"):
            do_layer1()
            st.rerun()
        return

    hospitals = st.session_state.hospitals

    # ── Per-hospital stats ────────────────────────────────────
    cols = st.columns(len(hospitals))
    icons = {"sainte_justine": "🏥", "sickkids": "🔬"}
    colors = {"sainte_justine": "#4e8df5", "sickkids": "#f5934e"}
    for col, (name, hd) in zip(cols, hospitals.items()):
        with col:
            tag = "Real data" if name == "sainte_justine" else "Simulated (biased)"
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{colors[name]}">
              <div class="kpi-label">{icons.get(name, '🏥')} {name.replace('_',' ').title()}</div>
              <div class="kpi-value">{hd.n_train + hd.n_test:,}</div>
              <div class="kpi-label" style="margin-top:6px">{tag}</div>
              <div style="margin-top:8px;font-size:.82rem;color:#6b7280">
                Train: {hd.n_train:,} &nbsp;·&nbsp; Test: {hd.n_test:,}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Age-group distribution comparison ────────────────────
    st.markdown("**Age-group distribution**")
    dist_rows = []
    for name, hd in hospitals.items():
        meta = pd.concat([hd.meta_train, hd.meta_test])
        for grp, cnt in meta["AgeGroup"].value_counts().items():
            dist_rows.append({
                "Hospital": name.replace("_", " ").title(),
                "Age group": grp,
                "Count": int(cnt),
            })
    if dist_rows:
        df_dist = pd.DataFrame(dist_rows)
        fig = px.bar(
            df_dist, x="Age group", y="Count", color="Hospital",
            barmode="group", height=280,
            color_discrete_map={
                "Sainte Justine": "#4e8df5",
                "Sickkids":       "#f5934e",
            },
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.35),
        )
        st.plotly_chart(fig, use_container_width=True, key="data_age_dist")
    st.markdown("**Bias injections applied to SickKids**")
    bias_cols = st.columns(5)
    scenarios = [
        ("Age shift",     f"+{SIM_CFG.age_shift_ratio*100:.0f}% newborns",  "📈"),
        ("Temp offset",   f"{SIM_CFG.temp_bias:+.1f} °C systematic",         "🌡️"),
        ("Device noise",  f"σ = {SIM_CFG.device_noise_std} bpm",              "📡"),
        ("Physio shift",  f"+{SIM_CFG.physio_hr_shift} bpm (3–9 yr)",         "❤️"),
        ("Data size",     f"{SIM_CFG.sk_data_fraction*100:.0f}% of SJ",       "📉"),
    ]
    for col, (name, value, icon) in zip(bias_cols, scenarios):
        col.metric(f"{icon} {name}", value)


# ══════════════════════════════════════════════════════════════════
#  Tab 2 — FL training
# ══════════════════════════════════════════════════════════════════

def render_training_tab() -> None:
    st.subheader("Federated learning training")

    if st.session_state.phase in ("idle", "data") and st.session_state.hospitals is None:
        st.info("Load data first (Tab 1).")
        return

    if st.session_state.phase in ("idle",):
        st.info("Load data first (Tab 1).")
        return

    if st.session_state.phase == "data":
        cfg = st.session_state.fl_cfg
        st.info(
            f"Strategy: **{cfg.strategy.upper()}** · "
            f"Rounds: **{cfg.n_rounds}** · "
            f"Local epochs: **{cfg.local_epochs}**"
        )
        if st.button("▶ Train Federated Model", type="primary"):
            do_layer2()
            st.rerun()
        return

    # ── Results are available ─────────────────────────────────
    fl = st.session_state.fl_results

    # ── KPI row ───────────────────────────────────────────────
    hospitals_list = list(fl["final_metrics"].keys())
    kpi_cols = st.columns(len(hospitals_list) * 3)
    col_i = 0
    for h in hospitals_list:
        m = fl["final_metrics"][h]
        kpi_cols[col_i].metric(
            f"{h.replace('_',' ').title()} MAE",
            f"{m[f'{h}_mae']:.2f} bpm",
        )
        kpi_cols[col_i + 1].metric(
            "RMSE", f"{m[f'{h}_rmse']:.2f}"
        )
        kpi_cols[col_i + 2].metric(
            "R²", f"{m[f'{h}_r2']:.3f}"
        )
        col_i += 3

    st.divider()

    # ── MAE convergence ───────────────────────────────────────
    if fl["round_mae"]:
        st.markdown("**MAE convergence per round**")
        fig = go.Figure(go.Scatter(
            x=list(range(1, len(fl["round_mae"]) + 1)),
            y=fl["round_mae"],
            mode="lines+markers",
            line=dict(color="#4e8df5", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(78,141,245,0.08)",
        ))
        fig.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0),
            xaxis_title="Round", yaxis_title="Avg MAE (bpm)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"training_convergence_{st.session_state.correction_round}")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**MAE by age group**")
        st.plotly_chart(
            _subgroup_chart(fl["subgroup_metrics"]),
            use_container_width=True,
            key=f"training_subgroup_{st.session_state.correction_round}",
        )
    with chart_col2:
        st.markdown("**MAE by gender**")
        st.plotly_chart(
            _gender_chart(fl["subgroup_metrics"]),
            use_container_width=True,
            key=f"training_gender_{st.session_state.correction_round}",
        )

    # ── Before / after comparison (if correction was applied) ─
    before = st.session_state.fl_results_initial
    latest = st.session_state.fl_results_latest
    if (before is not None and latest is not None
            and st.session_state.correction_round > 0):
        st.divider()
        st.markdown("**Before vs after correction**")
        st.plotly_chart(
            _before_after_chart(before, latest),
            use_container_width=True,
            key=f"training_before_after_{st.session_state.correction_round}",
        )


# ══════════════════════════════════════════════════════════════════
#  Tab 3 — Agent analysis (chat UI)
# ══════════════════════════════════════════════════════════════════

def render_analysis_tab() -> None:
    st.subheader("AI fairness analysis")

    # Guard: need FL results
    if st.session_state.fl_results is None:
        st.info("Train the federated model first (Tab 2).")
        return

    # ── Conversation history ──────────────────────────────────
    for msg in st.session_state.chat_messages:
        role = msg["role"]

        if role == "user":
            with st.chat_message("user", avatar="👨‍⚕️"):
                st.markdown(msg["content"])

        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

        elif role == "tool":
            tool_name = msg.get("name", "tool")
            with st.expander(f"🔧 `{tool_name}` result", expanded=False):
                content = msg["content"]
                # Try to pretty-print JSON
                try:
                    st.json(json.loads(content))
                except Exception:
                    st.code(content, language="text")

    # ── Auto-advance agent when thinking ─────────────────────
    if st.session_state.agent_state == "thinking":
        with st.spinner("Agent reasoning …"):
            agent_step()
        st.rerun()

    # ── Human-in-the-loop approval card ──────────────────────
    if st.session_state.agent_state == "awaiting_approval":
        tc = st.session_state.pending_correction
        if tc:
            args = tc.get("args", {})

            # Default values
            c_type = tc["name"]
            desc = json.dumps(args, indent=2)
            rationale = ""

            # If the model already supplied a rich proposal, use it
            if "correction_type" in args:
                c_type = args["correction_type"]

            if "description" in args:
                desc = args["description"]

            if "rationale" in args:
                rationale = args["rationale"]

            # Otherwise create a readable description from the tool arguments
            elif tc["name"] == "run_fedprox":
                mu = args.get("mu", 0.05)

                c_type = "FedProx"

                desc = (
                    f"Switch the federated learning strategy from "
                    f"{st.session_state.fl_cfg.strategy.upper()} to FedProx "
                    f"using μ = {mu}."
                )

                rationale = (
                    "FedProx adds a proximal penalty that keeps each hospital's "
                    "local model closer to the global model. This reduces client "
                    "drift caused by heterogeneous hospital data and can improve "
                    "fairness across hospitals and patient subgroups."
                )


            elif tc["name"] == "reweight_data":

                hospital = str(args.get("hospital", "selected hospital"))

                raw_weight = args.get("new_weight", 1.0)

                try:

                    weight = float(raw_weight)

                    weight_text = f"{weight:.2f}"

                except (TypeError, ValueError):

                    weight = 1.0

                    weight_text = str(raw_weight)

                c_type = "Hospital Reweighting"

                desc = (

                    f"Reduce the aggregation weight of {hospital} "

                    f"to {weight_text}."

                )

                rationale = (

                    f"Reducing {hospital}'s aggregation influence may limit the "

                    "propagation of its local bias to the global model while still "

                    "allowing the hospital to participate in federated training."

                )

            st.markdown(f"""
            <div class="correction-card">
              <div class="correction-title">⚠️ Step 4 — Agent suggests an improvement</div>
              <b>Correction type:</b> {c_type}<br>
              <b>Action:</b> {desc}<br>
              <b>Rationale:</b> {rationale}
            </div>
            """, unsafe_allow_html=True)

            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("✅ Approve", type="primary", use_container_width=True):
                handle_approval(True)
                st.rerun()
            if c2.button("❌ Reject", use_container_width=True):
                handle_approval(False)
                st.rerun()

    # ── Retrain button (appears after correction approved) ────
    approved = [e for e in st.session_state.correction_log if e["approved"]]
    not_yet_retrained = (
        approved
        and st.session_state.agent_state in ("thinking", "awaiting_approval", "done")
        and st.session_state.phase in ("analyzing",)
    )

    # Also show retrain button when agent is done and there are approved corrections
    if (st.session_state.agent_state == "done"
            and approved
            and st.session_state.phase in ("analyzing",)):
        latest_correction = approved[-1]
        st.success(
            f"✅ Analysis complete. "
            f"**{len(approved)} correction(s)** approved — ready to retrain."
        )
        if st.button(
            "⚙️  Step 5 — Retrain with approved corrections",
            type="primary",
        ):
            with st.spinner("Retraining FL model with correction …"):
                apply_correction_and_retrain(latest_correction)
            st.session_state.agent_state = "idle"
            st.toast("✅ Retraining complete!", icon="🎉")
            st.rerun()

    # ── Chat input (always visible) ───────────────────────────
    if st.session_state.agent_state not in ("thinking", "awaiting_approval"):
        prompt = st.chat_input(
            "Ask the agent… e.g. 'Are infants at risk? How can we improve?'"
        )
        if prompt:
            start_agent(prompt)
            st.rerun()
    else:
        st.chat_input("Waiting for agent …", disabled=True)


# ══════════════════════════════════════════════════════════════════
#  Tab 4 — Report
# ══════════════════════════════════════════════════════════════════

def render_report_tab() -> None:
    st.subheader("Audit report")

    if st.session_state.fl_results is None:
        st.info("Run the pipeline first to generate a report.")
        return

    fl   = st.session_state.fl_results
    sg   = fl["subgroup_metrics"]
    corr = st.session_state.correction_log

    # ── Summary metrics ───────────────────────────────────────
    st.markdown("### 📊 Model performance")
    rep_cols = st.columns(len(fl["final_metrics"]))
    for col, (h, m) in zip(rep_cols, fl["final_metrics"].items()):
        with col:
            st.markdown(f"**{h.replace('_',' ').title()}**")
            st.metric("MAE",  f"{m[f'{h}_mae']:.3f} bpm")
            st.metric("RMSE", f"{m[f'{h}_rmse']:.3f}")
            st.metric("R²",   f"{m[f'{h}_r2']:.3f}")

    st.divider()

    # ── Subgroup harm table ───────────────────────────────────
    st.markdown("### 🔍 Subgroup harm analysis")
    rows = []
    for h, metrics in sg.items():
        for k, v in metrics.items():
            if k.startswith("age_"):
                grp = k.replace("age_", "")
                rows.append({
                    "Hospital":   h.replace("_", " ").title(),
                    "Age group":  grp,
                    "MAE (bpm)":  round(v, 3),
                    "Status":     "⚠️ High" if v > 20 else "✅ OK",
                })
    if rows:
        df = pd.DataFrame(rows).sort_values("MAE (bpm)", ascending=False)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "MAE (bpm)": st.column_config.ProgressColumn(
                    "MAE (bpm)", min_value=0, max_value=40, format="%.2f"
                ),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
            hide_index=True,
        )

    st.divider()

    # ── Correction log ────────────────────────────────────────
    st.markdown("### 🛠️ Correction log")
    if not corr:
        st.info("No corrections proposed yet.")
    else:
        for i, entry in enumerate(corr, 1):
            approved = entry.get("approved", False)
            args     = entry.get("args", {})
            badge    = "✅ Approved" if approved else "❌ Rejected"
            colour   = "#d1fae5" if approved else "#fee2e2"
            st.markdown(
                f"""<div style="background:{colour};border-radius:8px;
                padding:10px 14px;margin:4px 0">
                <b>#{i} {badge}</b> — {entry.get('tool','?')} :
                {args.get('description', json.dumps(args))}
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Before / after ────────────────────────────────────────
    before = st.session_state.fl_results_initial
    latest = st.session_state.fl_results_latest
    if before and latest and st.session_state.correction_round > 0:
        st.markdown("### 📈 Before vs after correction")
        st.plotly_chart(
            _before_after_chart(before, latest),
            use_container_width=True,
            key=f"report_before_after_{st.session_state.correction_round}",
        )
        # Delta metrics
        delta_cols = st.columns(len(fl["final_metrics"]))
        for col, h in zip(delta_cols, fl["final_metrics"]):
            mae_b = before["final_metrics"].get(h, {}).get(f"{h}_mae", 0)
            mae_a = latest["final_metrics"].get(h, {}).get(f"{h}_mae", 0)
            delta = mae_a - mae_b
            col.metric(
                f"{h.replace('_',' ').title()} MAE change",
                f"{mae_a:.3f} bpm",
                delta=f"{delta:+.3f}",
                delta_color="inverse",
            )

    st.divider()

    # ── Raw FL summary (collapsible) ─────────────────────────
    with st.expander("📄 Raw FL results JSON"):
        display = {
            k: v for k, v in fl.items()
            if k != "global_weights"          # skip numpy arrays
        }
        st.json(display)

    # ── Download ──────────────────────────────────────────────
    report_text = _build_text_report(fl, corr)
    st.download_button(
        "⬇️  Download audit report (.txt)",
        data=report_text,
        file_name="fairness_audit_report.txt",
        mime="text/plain",
    )


def _build_text_report(fl_results: dict, corr_log: list) -> str:
    lines = [
        "════════════════════════════════════════",
        "  PICU FL FAIRNESS — AUDIT REPORT",
        "════════════════════════════════════════",
        f"Strategy : {fl_results.get('strategy','?').upper()}",
        f"Rounds   : {len(fl_results.get('round_mae', []))}",
        "",
        "── Hospital metrics ──────────────────",
    ]
    for h, m in fl_results["final_metrics"].items():
        lines.append(
            f"  {h:<22} MAE={m[f'{h}_mae']:.3f}  "
            f"RMSE={m[f'{h}_rmse']:.3f}  R²={m[f'{h}_r2']:.3f}"
        )
    lines += ["", "── Top subgroup harms ────────────────"]
    all_sg = {
        f"{h}/{k.replace('age_', '')}": v
        for h, sg in fl_results["subgroup_metrics"].items()
        for k, v in sg.items() if k.startswith("age_")
    }
    for label, mae in sorted(all_sg.items(), key=lambda x: x[1], reverse=True)[:8]:
        flag = "⚠ " if mae > 20 else "  "
        lines.append(f"  {flag}{label:<42}  MAE={mae:.3f}")

    lines += ["", "── Correction log ────────────────────"]
    if not corr_log:
        lines.append("  No corrections proposed.")
    for i, e in enumerate(corr_log, 1):
        status = "APPROVED" if e.get("approved") else "REJECTED"
        lines.append(f"  #{i} [{status}]  {e.get('tool','?')}  {e.get('args',{})}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    render_sidebar()

    # ── Page header ───────────────────────────────────────────
    st.markdown(
        "## 🏥 PICU Federated Learning — Fairness Dashboard",
        help="Agentic fairness analysis for pediatric heart-rate prediction",
    )

    phase = st.session_state.phase

    # Progress bar across the top
    phase_pct = {
        "idle": 0, "data": 0.25,
        "training": 0.5, "trained": 0.5,
        "analyzing": 0.75, "retrained": 1.0,
    }
    st.progress(
        phase_pct.get(phase, 0),
        text=(
            "⏳ Waiting to start"        if phase == "idle"     else
            "✅ Step 1 — Data loaded"     if phase == "data"     else
            "🔄 Step 2 — Training …"      if phase == "training" else
            "✅ Step 2 — Model trained"   if phase == "trained"  else
            "🔄 Step 3-4 — Analysing …"   if phase == "analyzing" else
            "🎉 Step 5-6 — Complete!"
        ),
    )

    st.divider()

    # ── Quick-start panel (only on first load) ────────────────
    if phase == "idle":
        st.info(
            "👋 Welcome! Start by loading the data, or click **Run full pipeline** "
            "to execute all layers automatically."
        )
        col_a, col_b = st.columns(2)
        if col_a.button("▶ Load & Simulate Data", type="primary", use_container_width=True):
            do_layer1()
            st.rerun()
        if col_b.button("⚡ Run full pipeline (Layers 1+2)", use_container_width=True):
            do_layer1()
            st.rerun()

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1 · Data & Setup",
        "🏥 2 · FL Training",
        "🤖 3-5 · Agent Analysis",
        "📋 6 · Report",
    ])

    with tab1:
        render_data_tab()

    with tab2:
        render_training_tab()

    with tab3:
        render_analysis_tab()

    with tab4:
        render_report_tab()

    # ── Auto-advance: if data is loaded but not trained, nudge ─
    if phase == "data" and st.session_state.fl_results is None:
        st.sidebar.info("Data ready — go to Tab 2 to train the FL model.")


if __name__ == "__main__":
    main()