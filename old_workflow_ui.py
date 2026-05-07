import streamlit as st
import time
import re
import os
import sqlite3
import uuid
from langchain_core.messages import AIMessage, HumanMessage

# Force LangSmith config
os.environ["LANGCHAIN_PROJECT"] = "Old_Workflow_RAG"

# Import the workflow app from old_workflow.py
try:
    from old_workflow import app
except ImportError as e:
    st.error(f"Could not import 'app' from 'old_workflow.py'. Error: {e}")
    st.stop()

st.set_page_config(page_title="Old Workflow Agent", page_icon="🏗️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: radial-gradient(circle at top right, #1a1f2c, #0e1117);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.95);
        border-right: 1px solid #30363d;
    }

    /* Chat message container */
    .stChatMessage {
        background-color: rgba(30, 39, 50, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        padding: 0px 20px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
        border-color: #1f6feb !important;
    }

    /* Custom success/error boxes */
    .stSuccess, .stError, .stInfo {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏗️ Chat Archive")

# 1. Fetch historical thread IDs from SQLite
thread_ids = []
thread_displays = {}
db_path = os.path.join(os.getcwd(), "state.db")
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # langgraph stores thread IDs in the checkpoints table
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cur.fetchall()
        thread_ids = sorted([r[0] for r in rows], reverse=True)
        
        for t_id in thread_ids:
            try:
                state = app.get_state({"configurable": {"thread_id": t_id}})
                if state and state.values:
                    req = state.values.get("user_request", "Empty Chat")
                    display_req = req[:30] + "..." if len(req) > 30 else req
                    thread_displays[t_id] = display_req
                else:
                    thread_displays[t_id] = t_id
            except Exception:
                thread_displays[t_id] = t_id
    except Exception:
        pass

def format_thread(t_id):
    if t_id == "New Chat":
        return "➕ Start New Blueprint"
    display = thread_displays.get(t_id, str(t_id)[:8])
    return f"💬 {display}"

selected_thread = st.sidebar.selectbox("Session History", ["New Chat"] + thread_ids, format_func=format_thread)

# 2. Session state management
if selected_thread == "New Chat":
    if "active_thread_id" not in st.session_state or st.session_state.active_thread_id in thread_ids:
        st.session_state.active_thread_id = str(uuid.uuid4())
else:
    st.session_state.active_thread_id = selected_thread

st.title("🏗️ Infrastructure AI Agent")
st.markdown("#### *RAG-Powered Terraform Architect*")
st.caption("Grounded in corporate knowledge, validated for security, and cost-optimized.")

def render_infra_state(state_values):
    if not state_values:
        return
        
    # 1. Validation Status Hero
    is_valid = state_values.get("is_valid", False)
    if is_valid:
        st.success("✨ **Production Ready:** This infrastructure code has passed all syntax and security validations.")
    else:
        st.error(f"⚠️ **Incomplete Validation:** The agent encountered errors during the healing process.")
        with st.expander("Diagnostics & Logs"):
            st.code(state_values.get("validation_errors", "No error log found."), language="text")
            
    # 2. AI Reasoning
    messages = state_values.get("messages", [])
    if messages:
         # Find the last AI message
         last_ai_msg = None
         for m in reversed(messages):
             if isinstance(m, AIMessage):
                 last_ai_msg = m
                 break
         
         if last_ai_msg:
             content = last_ai_msg.content
             # Clean out the code blocks for the commentary section
             text_only = re.sub(r"```[^\n]*\n.*?```", "", content, flags=re.DOTALL).strip()
             if text_only:
                 with st.container():
                     st.markdown("### 🧠 Architect's Notes")
                     st.info(text_only)

    # 3. Code Workbench
    files = state_values.get("terraform_code", {})
    if files:
        st.markdown("### 🛠️ Terraform Blueprint")
        tabs = st.tabs([f"📄 {f}" for f in files.keys()])
        for i, (filename, code) in enumerate(files.items()):
            with tabs[i]:
                st.code(code, language="hcl")
                
    # 4. Meta Info (Citations & Cost)
    st.divider()
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 📚 Knowledge Base")
        citations = state_values.get("citations", [])
        if citations:
            for c in citations:
                st.markdown(f"🔖 `{c}`")
        else:
            st.markdown("_No external knowledge sources were triggered._")
            
    with m_col2:
        st.markdown("### 💰 FinOps Analysis")
        cost = state_values.get("cost_estimate", "")
        if not cost:
            st.markdown("_Cost breakdown pending..._")
        elif any(x in cost.lower() for x in ["not installed", "failed", "error"]):
            st.warning(f"Cost estimation service unavailable: {cost}")
        else:
            st.code(cost, language="text")

# 3. Render Historical Messages
config = {"configurable": {"thread_id": st.session_state.active_thread_id}}
current_state = app.get_state(config)

if current_state and current_state.values:
    user_req = current_state.values.get("user_request", "")
    if user_req:
        with st.chat_message("user"):
            st.markdown(user_req)
    with st.chat_message("assistant"):
        render_infra_state(current_state.values)
        st.markdown("---")
        st.caption("🔒 *Verified state loaded from persistent SQLite vault.*")

# 4. Chat Input Workflow
if prompt := st.chat_input("Describe the infrastructure you want to deploy..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🚀 Initializing Agentic Pipeline...", expanded=True) as status:
            initial_state = {
                "user_request": prompt,
                "messages": [],
                "retrieved_context": "",
                "citations": [],
                "terraform_code": {},
                "validation_errors": "",
                "is_valid": False,
                "retry_count": 0,
                "cost_estimate": ""
            }
            
            try:
                # Stream the graph execution for real-time feedback
                for event in app.stream(initial_state, config=config):
                    for node_name, state_update in event.items():
                        if node_name == "Retriever_Node":
                            status.update(label="📚 Smart Search Pipeline: MultiQuery & Reranking...", state="running")
                            st.write(f"✅ Retriever finished. (Found {len(state_update.get('citations', []))} relevant docs)")
                        elif node_name == "Architect_Node":
                            status.update(label="🛠️ Architect Node: Designing Production-Grade Blueprint...", state="running")
                            st.write("✅ Senior Architect finished drafting the infrastructure design.")
                        elif node_name == "Validator_Node":
                            status.update(label="🔎 Validator Node: Running Syntax & Security Checks...", state="running")
                            if state_update.get("is_valid"):
                                st.write("✅ Code passed all validations! 🎉")
                            else:
                                st.write("⚠️ Syntax/Security validation failed. Routing to Fixer Node...")
                        elif node_name == "Fixer_Node":
                            attempt = state_update.get('retry_count', 1)
                            status.update(label=f"🔧 Fixer Node: Self-healing Code (Attempt {attempt})...", state="running")
                            st.write(f"✅ Fixer applied patches for attempt {attempt}.")
                        elif node_name == "Cost_Estimator_Node":
                            status.update(label="💰 Cost Estimator Node: Analyzing Monthly Cloud Spend...", state="running")
                            st.write("✅ Infracost analysis complete.")
            except Exception as e:
                status.update(label="Pipeline Failed", state="error")
                st.error(f"Fatal Error: {e}")
                st.stop()
                
            status.update(label="Pipeline Complete!", state="complete", expanded=False)
            
        # Display the final results
        final_state = app.get_state(config).values
        if final_state:
            render_infra_state(final_state)
