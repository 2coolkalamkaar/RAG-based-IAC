import streamlit as st
import time
import re
import os
import sqlite3
import uuid

# Force LangSmith config
os.environ["LANGCHAIN_PROJECT"] = "Agent_Workflow_Advanced_RAG"

from agent_workflow_advanced_rag import app

st.set_page_config(page_title="Terraform RAG Agent", page_icon="🏗️", layout="wide")

st.sidebar.title("Chat History")

# 1. Fetch historical thread IDs from SQLite
thread_ids = []
db_path = os.path.join(os.getcwd(), "state.db")
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # langgraph-checkpoint-sqlite stores threads in "checkpoints" table
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cur.fetchall()
        thread_ids = sorted([r[0] for r in rows], reverse=True)
    except Exception as e:
        pass

# 2. Sidebar selection
selected_thread = st.sidebar.selectbox("Past Conversations", ["New Chat"] + thread_ids)

# 3. Session state management for active thread
if selected_thread == "New Chat":
    if "thread_id" not in st.session_state or st.session_state.thread_id in thread_ids:
        st.session_state.thread_id = str(uuid.uuid4())
else:
    st.session_state.thread_id = selected_thread

st.title(" Terraform AI Agent")
st.markdown("Generates, validates, self-heals, and prices your AWS infrastructure code.")

# Helper function to render the UI components of a langgraph state
def render_infra_state(final_state):
    if not final_state:
        return
        
    # 1. Validation Status
    if final_state.get("is_valid", False):
        st.success("✅ **Validation Passed:** Code successfully passed all `terraform validate` and `tflint` security checks.")
    else:
        st.error(f"❌ **Validation Failed:** Agent exhausted maximum retries ({final_state.get('retry_count', 0)}/3).")
        with st.expander("View Underlying Validation Errors"):
            st.code(final_state.get("validation_errors", ""), language="text")
            
    # 2. Agent Commentary
    messages = final_state.get("messages", [])
    if messages:
         content = messages[-1].content
         # Remove raw codeblocks from the commentary
         text_only = re.sub(r"```[^\n]*\n.*?```", "", content, flags=re.DOTALL).strip()
         if text_only:
             st.info(text_only)

    # 3. Generated Code (Tabbed Files)
    st.markdown("### 🏗️ Generated Terraform Blueprint")
    files = final_state.get("terraform_code", {})
    if files:
        tabs = st.tabs(list(files.keys()))
        for i, (filename, code) in enumerate(files.items()):
            with tabs[i]:
                st.code(code, language="hcl")
                
    # 4. Citations & FinOps side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📚 Grounded Knowledge References")
        citations = final_state.get("citations", [])
        if citations:
            for c in citations:
                st.markdown(f"- 📄 `{c}`")
        else:
            st.markdown("_No external knowledge sources utilized._")
            
    with col2:
        st.markdown("### 💰 FinOps Monthly Cost Estimation")
        cost = final_state.get("cost_estimate", "")
        if cost and "CLI not installed" not in cost:
            st.code(cost, language="text")
        else:
            st.markdown("_Cost estimation unavailable._")

# Load chat history into memory natively from DB
config = {"configurable": {"thread_id": st.session_state.thread_id}}
historical_state = app.get_state(config)

if historical_state and historical_state.values:
    # Render user's original request
    user_req = historical_state.values.get("user_request", "")
    if user_req:
        with st.chat_message("user"):
            st.markdown(user_req)
    # Render the assistant's previous infrastructure
    with st.chat_message("assistant"):
        render_infra_state(historical_state.values)
        st.markdown("---")
        st.markdown("*Previous Session Reloaded from SQLite Checkpoint Database* ✅")

# Start actual chat input for a New task
if prompt := st.chat_input("What infrastructure do you want to build?"):
    
    # Let users update an existing chat or start a new one
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Initializing Workflow...", expanded=True) as status:
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
                for event in app.stream(initial_state, config=config):
                    for node_name, state_update in event.items():
                        if node_name == "Retriever_Node":
                            status.update(label="📚 Agent is searching internal corporate knowledge base...", state="running")
                            st.write(f"Retriever finished. (Found {len(state_update.get('citations', []))} relevant docs to ground the generation)")
                        elif node_name == "Architect_Node":
                            status.update(label="🛠️ Agent is writing Terraform framework...", state="running")
                            st.write("Architect finished drafting initial code.")
                        elif node_name == "Validator_Node":
                            status.update(label="🔎 Validating code against security and syntax policies...", state="running")
                            if state_update.get("is_valid"):
                                st.write("Validation passed! 🎉")
                            else:
                                st.write("Syntax/Security validation failed. ⚠️ Routing to Fixer Node...")
                        elif node_name == "Fixer_Node":
                            status.update(label=f"🔧 Agent is self-healing code (Attempt {state_update.get('retry_count', 1)})...", state="running")
                            st.write("Fixer applied patches.")
                        elif node_name == "Cost_Estimator_Node":
                            status.update(label="💰 Evaluating Cloud Deployment Cost...", state="running")
                            st.write("Infracost Analysis Complete.")
            except Exception as e:
                status.update(label="Workflow Failed", state="error")
                st.error(f"Error executing agent workflow: {e}")
                st.stop()
                
            status.update(label="Pipeline Complete!", state="complete", expanded=False)
            
        # Render the final new state
        final_state = app.get_state(config).values
        if final_state:
            render_infra_state(final_state)
