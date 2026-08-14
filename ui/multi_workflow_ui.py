import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import re
import uuid
import time
import importlib
import sqlite3
from langchain_core.messages import AIMessage
from langgraph.types import Command

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Workflow Terraform Architect",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a1628 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}

/* ── Hero Header ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #7d8590;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* ── Pipeline Stage Cards ── */
.pipeline-wrapper {
    display: flex;
    gap: 8px;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.stage-card {
    flex: 1;
    min-width: 130px;
    padding: 12px 10px;
    border-radius: 12px;
    background: #161b22;
    border: 1px solid #21262d;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stage-card.idle {
    border-color: #21262d;
    color: #484f58;
}
.stage-card.running {
    border-color: #388bfd;
    background: linear-gradient(135deg, #0d2044 0%, #161b22 100%);
    color: #58a6ff;
    box-shadow: 0 0 20px rgba(56,139,253,0.25);
    animation: pulse-blue 1.5s ease-in-out infinite;
}
.stage-card.done {
    border-color: #2ea043;
    background: linear-gradient(135deg, #0a2a14 0%, #161b22 100%);
    color: #3fb950;
}
.stage-card.failed {
    border-color: #da3633;
    background: linear-gradient(135deg, #2a0a0a 0%, #161b22 100%);
    color: #f85149;
}
.stage-card.warning {
    border-color: #d29922;
    background: linear-gradient(135deg, #2a1f0a 0%, #161b22 100%);
    color: #e3b341;
}
@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 20px rgba(56,139,253,0.25); }
    50% { box-shadow: 0 0 30px rgba(56,139,253,0.5); }
}
.stage-icon { font-size: 1.6rem; margin-bottom: 4px; }
.stage-name { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stage-status { font-size: 0.65rem; margin-top: 2px; opacity: 0.8; }

/* ── Log Terminal ── */
.log-terminal {
    background: #010409;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.6;
}
.log-info  { color: #58a6ff; }
.log-ok    { color: #3fb950; }
.log-warn  { color: #e3b341; }
.log-err   { color: #f85149; }
.log-dim   { color: #484f58; }

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.metric-label { font-size: 0.72rem; color: #7d8590; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

/* ── Code tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    background: #161b22; border-radius: 8px 8px 0 0;
    border: 1px solid #21262d; border-bottom: none;
    color: #7d8590; font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important; color: white !important;
    border-color: #1f6feb !important;
}

/* ── Citation pills ── */
.citation-pill {
    display: inline-block;
    background: #1c2b3a;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Workflow Definitions ──────────────────────────────────────────────────────
WORKFLOWS = {
    "Basic": {
        "module": "workflows.agent_workflow",
        "project_name": "Workflow_with_gemini_2.5",
        "description": "Stateless workflow: Architect → Validator ↔ Fixer",
        "stages": [
            ("Architect_Node", "🏗️", "Architect", "Terraform Blueprint"),
            ("Validator_Node", "🔎", "Validator", "Syntax + Security"),
            ("Fixer_Node", "🔧", "Fixer", "Self-Healing"),
        ],
        "has_config": False,
        "uses_sqlite": False
    },
    "RAG": {
        "module": "workflows.agent_workflow_rag",
        "project_name": "Workflow_with_gemini_2.5_RAG",
        "description": "In-memory RAG workflow: Retriever → Architect → Validator ↔ Fixer",
        "stages": [
            ("Retriever_Node", "📚", "Retriever", "ChromaDB Search"),
            ("Architect_Node", "🏗️", "Architect", "Terraform Blueprint"),
            ("Validator_Node", "🔎", "Validator", "Syntax + Security"),
            ("Fixer_Node", "🔧", "Fixer", "Self-Healing"),
        ],
        "has_config": True,
        "uses_sqlite": False
    },
    "Advanced RAG": {
        "module": "workflows.agent_workflow_advanced_rag",
        "project_name": "Workflow_with_gemini_2.5_RAG and Reranking",
        "description": "Persistent Advanced RAG: MultiQuery Retriever → Architect → Validator ↔ Fixer → Cost Estimator",
        "stages": [
            ("Retriever_Node", "🔍", "Retriever", "MultiQuery + Rerank"),
            ("Architect_Node", "🏗️", "Architect", "Terraform Blueprint"),
            ("Validator_Node", "🔎", "Validator", "Syntax + Security"),
            ("Fixer_Node", "🔧", "Fixer", "Self-Healing"),
            ("Cost_Estimator_Node", "💰", "Cost Est.", "Infracost Analysis"),
        ],
        "has_config": True,
        "uses_sqlite": True
    },
    "Secure RAG": {
        "module": "workflows.agent_workflow_secure_rag",
        "project_name": "Workflow_with_gemini_2.5_Secure_RAG",
        "description": "Enterprise Secure RAG: Same as Advanced RAG, but with Checkov Shift-Left Security Scans.",
        "stages": [
            ("Retriever_Node", "🔍", "Retriever", "MultiQuery + Rerank"),
            ("Architect_Node", "🏗️", "Architect", "Secure Blueprint"),
            ("Validator_Node", "🛡️", "Validator", "Checkov + TFLint"),
            ("Fixer_Node", "🔧", "Fixer", "Security Healer"),
            ("Cost_Estimator_Node", "💰", "Cost Est.", "Infracost Analysis"),
        ],
        "has_config": True,
        "uses_sqlite": True
    },
    "HitL RAG": {
        "module": "workflows.agent_workflow_hitl",
        "project_name": "Workflow_with_gemini_2.5_HitL_RAG",
        "description": "Human-in-the-Loop RAG: Pause for review before applying surgical patches via Patcher Node. Supports SRE Upload Mode.",
        "stages": [
            ("Upload_Entry_Node", "📂", "Upload", "SRE Bypass"),
            ("Retriever_Node", "🔍", "Retriever", "MultiQuery + Rerank"),
            ("Architect_Node", "🏗️", "Architect", "Terraform Blueprint"),
            ("Validator_Node", "🔎", "Validator", "Syntax + Security"),
            ("HitL_Node", "⏸️", "HitL", "Human Review"),
            ("Patcher_Node", "🔧", "Patcher", "Surgical Diff"),
        ],
        "has_config": True,
        "uses_sqlite": True
    }
}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = str(uuid.uuid4())
if "selected_workflow" not in st.session_state:
    st.session_state.selected_workflow = "Advanced RAG"

# ── Helper: Load threads ──
@st.cache_data(ttl=5)
def load_threads():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.db")
    threads = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
            threads = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception:
            pass
    return threads

def get_thread_label(t_id, app_module):
    try:
        mod = importlib.import_module(app_module)
        app = mod.app
        state = app.get_state({"configurable": {"thread_id": t_id}})
        if state and state.values:
            req = state.values.get("user_request", "")
            if req:
                return req[:35] + "…" if len(req) > 35 else req
    except Exception:
        pass
    return t_id[:16] + "…"

def render_pipeline(stage_states: dict, stages: list):
    cards_html = '<div class="pipeline-wrapper">'
    for node_id, icon, name, desc in stages:
        status = stage_states.get(node_id, "idle")
        status_labels = {
            "idle": "Waiting…",
            "running": "Running…",
            "done": "Complete ✓",
            "failed": "Failed ✗",
            "warning": "Warning ⚠",
        }
        label = status_labels.get(status, status.capitalize())
        cards_html += f'''
        <div class="stage-card {status}">
            <div class="stage-icon">{icon}</div>
            <div class="stage-name">{name}</div>
            <div class="stage-status">{label}</div>
        </div>'''
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

def render_final_state(state_values: dict, start_time: float):
    if not state_values:
        return

    duration = round(time.time() - start_time, 1) if start_time else 0
    is_valid = state_values.get("is_valid", False)
    retry_count = state_values.get("retry_count", 0)
    citations = state_values.get("citations", [])
    files = state_values.get("terraform_code", {})
    cost = state_values.get("cost_estimate", "")
    messages = state_values.get("messages", [])

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{"✅" if is_valid else "❌"}</div>
        <div class="metric-label">Validation</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{len(files)}</div>
        <div class="metric-label">Files Gen.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{retry_count}</div>
        <div class="metric-label">Fix Attempts</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{len(citations)}</div>
        <div class="metric-label">Sources Used</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{duration}s</div>
        <div class="metric-label">Total Time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if is_valid:
        st.success("✨ **Production Ready** — Code passed all syntax & security validations.")
    else:
        st.error("⚠️ **Validation Failed** — Max retries reached. Review diagnostics below.")
        with st.expander("🔬 Diagnostics & Error Log"):
            st.code(state_values.get("validation_errors", "No errors logged."), language="text")

    tab_labels = ["🧠 Architect Notes"]
    if files: tab_labels.append(f"📂 Terraform Code ({len(files)} files)")
    if citations: tab_labels.append("📚 Citations")
    if cost: tab_labels.append("💰 Cost Analysis")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    with tabs[tab_idx]:
        last_ai = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, "name", "") != "Fixer_Node":
                last_ai = m
                break
        if last_ai:
            text_only = re.sub(r"```[^\n]*\n.*?```", "", last_ai.content, flags=re.DOTALL).strip()
            if text_only:
                st.markdown(text_only)
            else:
                st.info("Architect response contained only code blocks — see the Terraform Code tab.")
        else:
            st.info("No architect commentary available.")
    tab_idx += 1

    if files:
        with tabs[tab_idx]:
            file_tabs = st.tabs([f"📄 {fname}" for fname in files])
            for i, (fname, code) in enumerate(files.items()):
                with file_tabs[i]:
                    col_dl, _ = st.columns([1, 4])
                    with col_dl:
                        st.download_button(
                            f"⬇️ {fname}", data=code,
                            file_name=fname, mime="text/plain",
                            key=f"dl_{fname}_{int(time.time())}"
                        )
                    st.code(code, language="hcl")
        tab_idx += 1

    if citations:
        with tabs[tab_idx]:
            pills = "".join([f'<span class="citation-pill">📎 {c}</span>' for c in citations])
            st.markdown(f'<div style="margin:0.5rem 0">{pills}</div>', unsafe_allow_html=True)
        tab_idx += 1

    if cost:
        with tabs[tab_idx]:
            if any(x in cost.lower() for x in ["not installed", "failed", "error"]):
                st.warning(f"⚠️ {cost}")
            else:
                st.code(cost, language="text")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem">
        <div style="font-size:2.5rem">🏗️</div>
        <div style="font-weight:700; font-size:1.1rem; color:#e6edf3">Terraform Architect</div>
        <div style="font-size:0.75rem; color:#7d8590">Multi-Workflow Runner</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    selected_workflow = st.selectbox(
        "Select Workflow",
        options=list(WORKFLOWS.keys()),
        index=list(WORKFLOWS.keys()).index(st.session_state.selected_workflow)
    )
    if selected_workflow != st.session_state.selected_workflow:
        st.session_state.selected_workflow = selected_workflow
        st.rerun()

    workflow_info = WORKFLOWS[selected_workflow]
    st.info(workflow_info["description"])

    if workflow_info["uses_sqlite"]:
        st.divider()
        st.markdown("**💬 Session History**")
        threads = load_threads()
        if threads:
            options = ["➕ New Session"] + threads
            def fmt(t):
                if t == "➕ New Session": return t
                return f"💬 {get_thread_label(t, workflow_info['module'])}"
            sel = st.selectbox("Load session", options, format_func=fmt, label_visibility="collapsed")
            if sel == "➕ New Session":
                if st.button("Start Fresh", use_container_width=True):
                    st.session_state.active_thread_id = str(uuid.uuid4())
                    st.rerun()
            else:
                if st.button("Load This Session", use_container_width=True):
                    st.session_state.active_thread_id = sel
                    st.rerun()
        else:
            st.caption("No past sessions found.")
        st.caption(f"Thread: `{st.session_state.active_thread_id[:8]}…`")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏗️ Multi-Workflow Agent</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Currently running: <b>{selected_workflow}</b></div>', unsafe_allow_html=True)

workflow_module = workflow_info["module"]
stages = workflow_info["stages"]

try:
    mod = importlib.import_module(workflow_module)
    app = mod.app
except Exception as e:
    st.error(f"Failed to load workflow module `{workflow_module}`: {e}")
    st.stop()

# Show previous history if applicable
if workflow_info["uses_sqlite"]:
    config = {"configurable": {"thread_id": st.session_state.active_thread_id}}
    hist_state = app.get_state(config)
    if hist_state and hist_state.values and hist_state.values.get("user_request"):
        with st.expander("📜 Previous Session Result", expanded=False):
            st.markdown(f"**Request:** {hist_state.values.get('user_request', '')}")
            render_final_state(hist_state.values, 0)
else:
    config = {"configurable": {"thread_id": st.session_state.active_thread_id}} if workflow_info["has_config"] else None

# ── HitL State Management ──
is_paused = False
hitl_state = None
if config and selected_workflow == "HitL RAG":
    current_state = app.get_state(config)
    if current_state and current_state.next and "HitL_Node" in current_state.next:
        is_paused = True
        hitl_state = current_state

# ── SRE Upload Flow ──
if selected_workflow == "HitL RAG" and not is_paused:
    upload_mode = st.toggle("📂 SRE Upload Mode — patch existing infrastructure")
    if upload_mode:
        uploaded_files = st.file_uploader("Upload your .tf files", accept_multiple_files=True, type=["tf"])
        if uploaded_files:
            tf_dict = {f.name: f.getvalue().decode("utf-8") for f in uploaded_files}
            if st.button("🚀 Ingest Files & Start Patching", use_container_width=True):
                with st.spinner("Ingesting files and running security validations..."):
                    app.invoke({"upload_mode": True, "terraform_code": tf_dict}, config=config)
                st.rerun()

st.divider()

if is_paused:
    st.warning("⏸️ **Workflow is paused.** Please review the generated files and either approve them or request a patch.")
    tf_code = hitl_state.values.get("terraform_code", {})
    for fname, code in tf_code.items():
        with st.expander(f"📄 {fname}", expanded=True):
            st.code(code, language="hcl")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Finalize", use_container_width=True):
            with st.spinner("Finalizing..."):
                app.invoke(Command(resume={"hitl_action": "approve", "patch_request": ""}), config=config)
            st.rerun()
    with col2:
        patch_req = st.text_input("Feedback:", placeholder="e.g., 'Change instance type to c5.large'")
        if st.button("💬 Apply Patch", use_container_width=True) and patch_req:
            with st.spinner("Patching..."):
                app.invoke(Command(resume={"hitl_action": "patch", "patch_request": patch_req}), config=config)
            st.rerun()
else:
    # Only show chat input if we are NOT in upload mode
    show_chat = not (selected_workflow == "HitL RAG" and 'upload_mode' in locals() and upload_mode)
    user_prompt = st.chat_input("Describe the AWS infrastructure you want to build…") if show_chat else None
    
    if user_prompt:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            pipeline_placeholder = st.empty()
            log_placeholder = st.empty()

            stage_states = {s[0]: "idle" for s in stages}
            log_lines = []

            def update_ui():
                pipeline_placeholder.markdown(
                    '<div class="pipeline-wrapper">' +
                    "".join([
                        f'<div class="stage-card {stage_states[s[0]]}">'
                        f'<div class="stage-icon">{s[1]}</div>'
                        f'<div class="stage-name">{s[2]}</div>'
                        f'<div class="stage-status">{"Running…" if stage_states[s[0]]=="running" else "Complete ✓" if stage_states[s[0]]=="done" else "Failed ✗" if stage_states[s[0]]=="failed" else "Warning ⚠" if stage_states[s[0]]=="warning" else "Waiting…"}</div>'
                        f'</div>'
                        for s in stages
                    ]) + '</div>',
                    unsafe_allow_html=True
                )
                log_html = '<div class="log-terminal">' + "<br>".join(log_lines[-30:]) + '</div>'
                log_placeholder.markdown(log_html, unsafe_allow_html=True)

            def log(msg, level="info"):
                ts = time.strftime("%H:%M:%S")
                cls = {"info": "log-info", "ok": "log-ok", "warn": "log-warn", "err": "log-err", "dim": "log-dim"}.get(level, "")
                log_lines.append(f'<span class="log-dim">[{ts}]</span> <span class="{cls}">{msg}</span>')
                update_ui()

            update_ui()
            start_time = time.time()

            # Initialize state based on what the workflow expects
            initial_state = {
                "user_request": user_prompt,
                "messages": [],
                "terraform_code": {},
                "validation_errors": "",
                "is_valid": False,
                "retry_count": 0,
            }
            
            # Add RAG specific state items
            if "RAG" in selected_workflow:
                initial_state["retrieved_context"] = ""
                if selected_workflow == "Advanced RAG":
                    initial_state["citations"] = []
                    initial_state["cost_estimate"] = ""

            final_state_values = initial_state.copy()
            pipeline_error = None

            try:
                from langchain_core.tracers.context import tracing_v2_enabled
                proj_name = workflow_info.get("project_name", "Multi_Workflow_Agent")
                log(f"🚀 {selected_workflow} pipeline initializing…", "info")

                # Handle execution depending on config requirements
                with tracing_v2_enabled(project_name=proj_name) as cb:
                    if config:
                        if "callbacks" in config:
                            config["callbacks"].append(cb)
                        else:
                            config["callbacks"] = [cb]
                        stream_generator = app.stream(initial_state, config=config)
                    else:
                        stream_generator = app.stream(initial_state, config={"callbacks": [cb]})

                    for event in stream_generator:
                        for node_name, state_update in event.items():
                            # Accumulate state updates for stateless rendering
                            final_state_values.update(state_update)
                            
                            for s in stages:
                                if stage_states[s[0]] == "running":
                                    stage_states[s[0]] = "done"
        
                            if node_name == "Retriever_Node":
                                stage_states["Retriever_Node"] = "running"
                                log("🔍 Retriever: Searching knowledge base…", "info")
                                update_ui()
                                if "citations" in state_update:
                                    log(f"✅ Retriever: Retrieved {len(state_update['citations'])} sources.", "ok")
                                else:
                                    log(f"✅ Retriever: Search complete.", "ok")
                            
                            elif node_name == "Architect_Node":
                                stage_states["Architect_Node"] = "running"
                                log("🏗️ Architect: Generating Terraform blueprint…", "info")
                                update_ui()
                                n_files = len(state_update.get("terraform_code", {}))
                                log(f"✅ Architect: Generated {n_files} file(s).", "ok")
        
                            elif node_name == "Validator_Node":
                                stage_states["Validator_Node"] = "running"
                                log("🔎 Validator: Running syntax & security checks…", "info")
                                update_ui()
                                if state_update.get("is_valid"):
                                    stage_states["Validator_Node"] = "done"
                                    log("✅ Validator: All checks passed!", "ok")
                                else:
                                    stage_states["Validator_Node"] = "warning"
                                    log("⚠️  Validator: Errors found — routing to Fixer…", "warn")
        
                            elif node_name == "Fixer_Node":
                                attempt = state_update.get("retry_count", 1)
                                stage_states["Fixer_Node"] = "running"
                                log(f"🔧 Fixer: Self-healing attempt #{attempt}…", "warn")
                                update_ui()
                                log(f"✅ Fixer: Applied patches.", "ok")
        
                            elif node_name == "Cost_Estimator_Node":
                                stage_states["Cost_Estimator_Node"] = "running"
                                log("💰 Cost Estimator: Running analysis…", "info")
                                update_ui()
                                log("✅ Cost Estimator: Analysis complete.", "ok")

                            elif node_name == "Upload_Entry_Node":
                                stage_states["Upload_Entry_Node"] = "running"
                                log("📂 Upload: Injecting SRE files…", "info")
                                update_ui()
                                log("✅ Upload: Files injected.", "ok")

                            elif node_name == "HitL_Node":
                                stage_states["HitL_Node"] = "running"
                                log("⏸️  HitL: Waiting for human approval…", "warn")
                                update_ui()
                                
                            elif node_name == "Patcher_Node":
                                stage_states["Patcher_Node"] = "running"
                                log("🔧 Patcher: Applying surgical changes…", "info")
                                update_ui()
                                log(f"✅ Patcher: Parsed new files.", "ok")
        
                for s in stages:
                    if stage_states[s[0]] == "running":
                        stage_states[s[0]] = "done"

                duration = round(time.time() - start_time, 1)
                log(f"🏁 Pipeline complete in {duration}s.", "ok")
                update_ui()

                # If it has config, fetch the authoritative state from checkpointer
                if config:
                    final_state_values = app.get_state(config).values

            except Exception as e:
                for s in stages:
                    if stage_states[s[0]] == "running": stage_states[s[0]] = "failed"
                log(f"❌ Fatal error: {e}", "err")
                update_ui()
                pipeline_error = str(e)

            pipeline_placeholder.empty()
            log_placeholder.empty()

            if pipeline_error:
                st.error(f"Pipeline failed: {pipeline_error}")
            else:
                render_pipeline(stage_states, stages)
                st.divider()
                st.subheader("Architect Summary")
                render_final_state(final_state_values, start_time)
