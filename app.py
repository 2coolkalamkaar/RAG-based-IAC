import os
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# --- 1. Configuration and Setup ---

# Load environment variables from a .env file
load_dotenv()

# Set Streamlit page configuration for a better layout
st.set_page_config(
    layout="wide",
    page_title="Production-Grade Terraform Generator",
    page_icon="🚀"
)

# The production-grade system prompt we engineered previously.
# It's better to define this as a constant at the top of the script.
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

# --- 2. Helper Function to Parse LLM Output ---

def parse_terraform_code(response_content: str) -> dict:
    """
    Parses the raw LLM output string into a dictionary of filename: code.
    This is crucial for displaying the code in a structured way.
    """
    files = {}
    # Uses regular expressions to find code blocks marked with '# filename.tf'
    pattern = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
    matches = re.finditer(pattern, response_content, re.DOTALL)
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
    return files

# --- 3. Streamlit User Interface ---

st.title("🚀 Production-Grade Terraform Generator")

# Check for API key at the start and provide a clear error message.
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# Use st.text_area for a larger input box, which is better for descriptive prompts.
user_input = st.text_area(
    "Describe the AWS infrastructure you want to create:",
    height=150,
    placeholder="e.g., A resilient two-tier web application with a public-facing web server and a private database server in the eu-central-1 region."
)

if st.button("✨ Generate Terraform Code", type="primary"):
    if not user_input.strip():
        st.warning("Please describe the infrastructure you want to build.")
    else:
        # Use a spinner to show the user that something is happening.
        with st.spinner("🧑‍💻 Architecting your infrastructure... This may take a moment."):
            try:
                # Initialize the model. Note the updated, valid model name.
                model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0)

                # Use the proper LangChain message format for clarity and reliability.
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_input)
                ]

                # Invoke the model
                result = model.invoke(messages)

                # Parse the raw output into separate files
                terraform_files = parse_terraform_code(result.content)

                if not terraform_files:
                    st.error("Failed to parse the output from the AI. The AI may not have followed the requested format.")
                    st.subheader("Raw AI Output:")
                    st.code(result.content, language='text')
                else:
                    st.success("Terraform code generated successfully!")
                    
                    # Use st.tabs to create a beautiful, organized display for each file.
                    filenames = list(terraform_files.keys())
                    tabs = st.tabs(filenames)

                    for i, filename in enumerate(filenames):
                        with tabs[i]:
                            st.code(terraform_files[filename], language='hcl', line_numbers=True)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
