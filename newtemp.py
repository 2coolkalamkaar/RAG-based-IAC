import os
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# --- 1. Configuration and Setup ---

load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="Conversational Terraform Assistant",
    page_icon="🤖"
)

# The system prompt remains our foundational instruction set for the AI.
SYSTEM_PROMPT = """
You are a world-class DevOps architect and a Terraform expert with a deep specialization in creating secure, scalable, and highly available AWS infrastructure. Your task is to generate a complete, production-hardened, and reusable Terraform module configuration based on the user's request.

Follow these rules with absolute precision:

1.  **Modular File Structure:** To promote maintainability, split resource definitions into logical files:
    * `main.tf`: Provider configuration and `locals` block only.
    * `vpc.tf`: All networking resources (VPC, Subnets, IGW, NAT Gateways, Route Tables).
    * `security.tf`: All security-related resources (`aws_security_group`).
    * `compute.tf`: All compute resources (`aws_instance`, `data "aws_ami"`).
    * `variables.tf`: All input variables.
    * `outputs.tf`: All outputs.
    * `backend.tf`: Remote state configuration.

2.  **Highly Available & Scalable Network:**
    * **Multi-AZ by Default:** The network must be highly available. Create resources across a configurable number of Availability Zones (`az_count` variable, default to 2).
    * **HA NAT Gateways:** For true resilience, provision one **NAT Gateway** and one **Elastic IP** in *each* Availability Zone.
    * **Zonal Routing:** Create a separate private route table for *each* AZ. Ensure that each private subnet routes its outbound `0.0.0.0/0` traffic through the NAT Gateway located in its **own** Availability Zone. This prevents cross-AZ data transfer costs and improves fault tolerance.
    * **Parameterized Subnets:** Allow flexible subnet sizing. Use variables like `public_subnet_newbits` and `private_subnet_newbits` in the `cidrsubnet` function.

3.  **Security First (Principle of Least Privilege):**
    * **Mandatory SSH CIDR:** The variable for allowed SSH ingress CIDR blocks (`allowed_ssh_cidr_blocks`) **must not have a default value**. This forces the user to define a secure, specific IP range and prevents accidental exposure with `0.0.0.0/0`.
    * **Secure Instance Placement:** All `aws_instance` resources must be placed in **private subnets** by default.
    * **Dynamic Security Group Rules:** Use a `dynamic` block and a `list(object({}))` variable to allow users to add additional, specific ingress rules beyond SSH.

4.  **Outputs & Connectivity:**
    * The EC2 instance is in a private subnet and is inaccessible via direct SSH from the internet.
    * **Do not** generate a direct `ssh` command output.
    * Instead, provide an output named `ssm_connection_command` with the command to connect using **AWS Systems Manager (SSM) Session Manager**, which is the modern, secure standard for private instance access.

5.  **Code Quality & Reusability:**
    * The code must be clean, readable, and strictly follow HashiCorp's official style conventions.
    * All resource and variable names must use underscores (`_`).
    * Use a `locals` block for common tags (`Project`, `Environment`, `ManagedBy`) and apply them consistently to all taggable resources.
    * Never hardcode AMI IDs or AZ names. Use data sources to look them up dynamically.

6.  **Output Format:**
    * Provide **ONLY** the HCL code for all specified files.
    * Clearly mark the start of each file with a comment (e.g., `# main.tf`).
    * Do not include any other explanations, introductions, or closing remarks.

"""

# --- 2. Helper Function (Same as before) ---

def parse_terraform_code(response_content: str) -> dict:
    """Parses the raw LLM output string into a dictionary of filename: code."""
    files = {}
    pattern = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
    matches = re.finditer(pattern, response_content, re.DOTALL)
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
    return files


st.title("🤖 Conversational Terraform Assistant")
st.caption("Start by describing your infrastructure, then ask for modifications in subsequent messages.")

# Check for API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# Initialize the chat history in session_state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = [
        # We start with the system message, but we don't display it to the user.
        SystemMessage(content=SYSTEM_PROMPT)
    ]

# Display past messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        # The AI's response is the code, so we process and display it with tabs
        with st.chat_message("assistant"):
            terraform_files = parse_terraform_code(message.content)
            if not terraform_files:
                # If parsing fails, show the raw content
                st.code(message.content, language='text')
            else:
                filenames = list(terraform_files.keys())
                tabs = st.tabs(filenames)
                for i, filename in enumerate(filenames):
                    with tabs[i]:
                        st.code(terraform_files[filename], language='hcl', line_numbers=True)

# Get user input using the new chat_input widget
if user_input := st.chat_input("What changes would you like to make?"):
    # Add user message to history and display it
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0)
                
                # *** CRITICAL CHANGE ***
                # We now pass the *entire* message history to the model
                response = model.invoke(st.session_state.messages)
                
                # Add AI response to history
                st.session_state.messages.append(AIMessage(content=response.content))

                # Display AI response using the same tabbed logic
                terraform_files = parse_terraform_code(response.content)
                if not terraform_files:
                    st.code(response.content, language='text')
                else:
                    filenames = list(terraform_files.keys())
                    tabs = st.tabs(filenames)
                    for i, filename in enumerate(filenames):
                        with tabs[i]:
                            st.code(terraform_files[filename], language='hcl', line_numbers=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")