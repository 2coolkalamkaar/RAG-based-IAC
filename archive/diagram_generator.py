
import os
import re
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def generate_diagram(terraform_code: str, output_name: str = "architecture_diagram") -> str:
    """
    Uses an LLM to analyze Terraform code and generate a Python script 
    that uses the 'diagrams' library to render the architecture.
    """
    
    # Check for API Key
    if "GOOGLE_API_KEY" not in os.environ:
        print("GOOGLE_API_KEY not found. Cannot generate diagram via LLM.")
        return None

    try:
        # 1. Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

        # 2. Define Prompt
        system_prompt = (
            "You are a Senior Solutions Architect and Python Expert. "
            "Your task is to visualize Terraform infrastructure using the Python 'diagrams' library.\n"
            "Input: Terraform Code (HCL).\n"
            "Output: A complete, executable Python script."
        )

        user_prompt = (
            f"Generate a Python script using the `diagrams` library to visualize the following Terraform code.\n"
            f"Save the diagram to filename='{output_name}'.\n"
            f"Use `show=False` in the Diagram constructor.\n"
            f"Group resources logically into Clusters (e.g., VPC, Subnets, K8s Clusters).\n"
            f"Infer relationships/connections between resources based on the Terraform references (e.g., `aws_instance` connecting to `aws_subnet`).\n"
            f"Import all necessary nodes from `diagrams.aws` (compute, network, database, storage, etc.).\n"
            f"Do not include any markdown formatting (backticks) or explanations. Just the raw Python code.\n"
            f"\n"
            f" Terraform Code:\n"
            f"```hcl\n{terraform_code}\n```"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        # 3. Generate Code
        chain = prompt | llm
        response = chain.invoke({})
        python_code = response.content

        # Clean formatting just in case
        python_code = re.sub(r"^```python\n", "", python_code)
        python_code = re.sub(r"^```\n", "", python_code)
        python_code = re.sub(r"\n```$", "", python_code)

        # 4. Save and Execute Script
        script_path = "temp_diagram_script.py"
        with open(script_path, "w") as f:
            f.write(python_code)
        
        # Execute the generated script
        # We run it as a subprocess to isolate it
        import subprocess
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Diagram generation script failed:\n{result.stderr}")
            return None
        
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

        # Return the expected image path
        expected_image = f"{output_name}.png"
        if os.path.exists(expected_image):
            return expected_image
        else:
            print("Script ran but image file not found.")
            return None

    except Exception as e:
        print(f"Error in diagram generation: {e}")
        return None
