import streamlit as st
import os
import shutil
import time
from fullfunctional import rag_chain, save_code_to_temp, run_checkov, run_terraform_plan

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG IAC Architect",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR ESTHETICS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF9100);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9100 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #1a1c24;
        border: 1px solid #2d3139;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8297/8297341.png", width=64)
    st.title("Settings")
    st.info("This agent uses a RAG-based approach with Gemini to generate, validate, and secure Terraform code.")
    
    st.divider()
    
    st.subheader("Capabilities")
    st.markdown("""
    - **Generate** Terraform HCL
    - **Scan** with Checkov
    - **Validate** with `terraform plan`
    - **Self-correct** on errors
    """)
    
    st.divider()
    st.caption("v1.0.0 | Powered by LangChain & Gemini")

# --- MAIN LAYOUT ---
st.markdown('<div class="main-header">🏗️ RAG Infrastructure Architect</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Describe your desired AWS infrastructure, and let the AI architect, secure, and validate it for you.</div>', unsafe_allow_html=True)

query = st.text_area(
    "Infrastructure Requirements", 
    height=150, 
    placeholder="Example: Create a custom VPC with public and private subnets, an Internet Gateway, and a security group allowing HTTP access.",
    help="Be as specific as possible about resources, names, and security constraints."
)

col1, col2 = st.columns([1, 5])
with col1:
    generate_btn = st.button("🚀 Design & Deploy", type="primary", use_container_width=True)

if generate_btn:
    if rag_chain is None:
        st.error("❌ Infrastructure Knowledge Base (Vector Store) is missing!")
        st.info("Please run 'vector_store.py' to generate the embeddings first. Ensure you have the Terraform documentation available.")
    elif not query.strip():
        st.error("Please enter a description for the infrastructure.")
    else:
        # --- EXECUTION LOGIC ---
        final_output_dir = "./generated_infra"
        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)
        os.makedirs(final_output_dir)

        status_container = st.status("Initializing AI Architect...", expanded=True)
        
        try:
            # Set retries to 2 as requested
            max_retries = 2
            attempt = 0
            error_feedback = ""
            success = False
            partial_success = False
            
            # Start Timer
            start_time = time.time()

            while attempt < max_retries:
                step_str = f"Attempt {attempt + 1}/{max_retries}"
                status_container.write(f"**🔄 {step_str}**")
                
                # 1. Prepare Input
                if error_feedback:
                    current_input = f"{query}\n\nPREVIOUS ATTEMPT FAILED. FIX THESE ERRORS:\n{error_feedback}"
                else:
                    current_input = query
                
                # 2. Generate
                status_container.write("🧠 Querying LLM & Retrieving Context...")
                llm_response = rag_chain.invoke(current_input)
                
                # 3. Save
                status_container.write("💾 Parsing and saving Code...")
                temp_dir, files = save_code_to_temp(llm_response)
                
                if not files:
                    status_container.warning("⚠️ No code found in response. Retrying...")
                    error_feedback = "You did not provide any code blocks formatted with '# filename.tf'."
                    attempt += 1
                    continue
                
                status_container.markdown(f"Generated: `{files}`")

                # 4. Security Scan
                status_container.write("🛡️ Running Checkov Security Scan...")
                secure, security_msg = run_checkov(temp_dir)
                
                if not secure:
                    status_container.error("❌ Security Issues Found!")
                    with st.expander("See Security Errors", expanded=True):
                        st.code(security_msg, language="text")
                    error_feedback = security_msg
                    
                    # IF THIS WAS THE LAST ATTEMPT, WE STOP HERE BUT SAVE THE FILES
                    if attempt == max_retries - 1:
                        status_container.warning("⚠️ Max retries reached. Returning best effort with known vulnerabilities.")
                        partial_success = True
                        # Copy files before breaking
                        for filename in files:
                            shutil.copy(os.path.join(temp_dir, filename), final_output_dir)
                        shutil.rmtree(temp_dir)
                        break

                    attempt += 1
                    shutil.rmtree(temp_dir)
                    continue
                
                status_container.success("✅ Security Checks Passed.")

                # 5. Terraform Plan
                status_container.write("🌍 Running Terraform Plan...")
                valid, plan_msg = run_terraform_plan(temp_dir)
                
                if not valid:
                    status_container.error("❌ Terraform Plan Failed!")
                    with st.expander("See Plan Errors", expanded=True):
                        st.code(plan_msg, language="text")
                    error_feedback = plan_msg

                    # IF THIS WAS THE LAST ATTEMPT, WE STOP HERE BUT SAVE THE FILES
                    if attempt == max_retries - 1:
                        status_container.warning("⚠️ Max retries reached. Returning best effort with syntax errors.")
                        partial_success = True
                        # Copy files before breaking
                        for filename in files:
                            shutil.copy(os.path.join(temp_dir, filename), final_output_dir)
                        shutil.rmtree(temp_dir)
                        break

                    attempt += 1
                    shutil.rmtree(temp_dir)
                    continue
                
                status_container.success("✅ Terraform Plan Validated.")
                
                # 6. Success
                success = True
                
                # Copy files
                for filename in files:
                    shutil.copy(os.path.join(temp_dir, filename), final_output_dir)
                
                shutil.rmtree(temp_dir)
                status_container.update(label="Infrastructure Readiness: 100%", state="complete", expanded=False)
                break
            
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            if success:
                st.balloons()
                st.success(f"Infrastructure successfully generated in {duration} seconds!")
            elif partial_success:
                status_container.update(label="Generated with Warnings", state="complete", expanded=False)
                st.warning(f"Process completed in {duration} seconds, but issues remain. access the generated files below.")
                if error_feedback:
                     with st.expander("Last Known Errors", expanded=True):
                        st.error(error_feedback)
            else:
                status_container.update(label="Generation Failed", state="error")
                st.error(f"Failed to generate valid infrastructure after {max_retries} attempts.")
            
            # SHOW FILES IF SUCCESS OR PARTIAL_SUCCESS
            if success or partial_success:
                st.subheader("📁 Generated Files")
                st.caption(f"Files are saved locally at: `{os.path.abspath(final_output_dir)}`")
                files = os.listdir(final_output_dir)
                
                if files:
                    tabs = st.tabs(files)
                    for i, file in enumerate(files):
                        with tabs[i]:
                            file_path = os.path.join(final_output_dir, file)
                            with open(file_path, 'r') as f:
                                code_content = f.read()
                            
                            st.download_button(
                                label=f"Download {file}",
                                data=code_content,
                                file_name=file,
                                mime="text/plain",
                                key=f"dl_{i}"
                            )
                            st.code(code_content, language='hcl')

        except Exception as e:
            status_container.update(label="System Error", state="error")
            st.error(f"An unexpected error occurred: {str(e)}")

# --- FOOTER ---
st.divider()
st.markdown("<div style='text-align: center; color: #666;'>© 2024 Intelligent RAG Architect</div>", unsafe_allow_html=True)
