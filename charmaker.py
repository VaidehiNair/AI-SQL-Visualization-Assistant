import os
import re
import json
from typing import TypedDict, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import pandas as pd
import uuid
from decimal import Decimal



# Load environment variables
load_dotenv()

# Define the state
class AgentState(TypedDict):
    query: str
    data: pd.DataFrame
    generated_code: str
    analysis: str
    chart_path: Optional[str]
    retry_count: int
    code_feedback: str
    chart_json: Optional[Dict[str, Any]]
    json_feedback: str


# Nodes
def data_extraction_node(state: AgentState):
    """
    Mock Data Extraction Node.
    """
    df = state['data']
    # Initialize retry count and feedback
    return {"data": df, "retry_count": 0, "code_feedback": "", "chart_json": None, "json_feedback": ""}

def code_generation_node(state: AgentState):
    """
    Generates Python code for analysis and plotting using Llama 3.
    """
    print("--- CODE GENERATION ---")
    df = state["data"]
    query = state["query"]
    feedback = state.get("code_feedback", "")
    retry_count = state.get("retry_count", 0)
    
    # Ensure exports directory exists
    os.makedirs("exports/charts", exist_ok=True)
    chart_path =  f"exports/charts/plot_{uuid.uuid4().hex}.png"
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    prompt = f"""
    You are a Python Data Analyst. You have a pandas DataFrame named `df` with the following columns: {df.columns.tolist()}.
    
    The DataFrame contains {len(df)} rows. Here is ALL the data:
    {df.to_string()}
    
    User Query: "{query}"
    
    Please generate Python code to:
    1. Create a relevant chart (e.g., bar chart, pie chart) using matplotlib based on the query.
    2. Use ALL {len(df)} rows from the DataFrame (do NOT use df.head() or limit the data).
    3. Save the chart to "{chart_path}".
    4. Do NOT use plt.show().
    
    IMPORTANT: 
    - The code must be valid Python.
    - Assume `df` and `plt` are already imported and available.
    - Do not wrap the code in markdown blocks (like ```python ... ```). Just provide the code.
    - Focus only on creating and saving the chart.
    - MUST use ALL rows in the DataFrame, not just a subset.
    """
    
    if feedback:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED. FEEDBACK:\n{feedback}\n\nPlease fix the errors and regenerate the code."
        retry_count += 1
    
    response = llm.invoke(prompt)
    code = response.content
    
    # Clean code
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)
    code = code.strip()
    
    return {"generated_code": code, "chart_path": chart_path, "retry_count": retry_count}

def feasibility_check_node(state: AgentState):
    """
    Uses LLM to check if the generated code is valid, safe, and sufficient for plotting.
    """
    print("--- FEASIBILITY CHECK ---")
    code = state["generated_code"]
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    prompt = f"""
    You are a Python code VALIDATOR (not a code generator). Your job is to ANALYZE and VALIDATE the code below, NOT to write new code.
    
    CONTEXT: The code should plot ALL data from a pandas DataFrame with {len(state['data'])} rows.
    The code assumes `df` and `plt` are already imported and available.
    
    CODE TO ANALYZE:
    ```python
    {code}
    ```
    
    VALIDATION CHECKLIST:
    1. **Syntax Validation**: Check if the code has valid Python syntax.
    2. **Security Check**: Ensure there are NO destructive operations like:
       - File deletion (os.remove, os.unlink, shutil.rmtree, etc.)
       - System commands (os.system, subprocess calls)
       - Network operations (requests, urllib, etc.)
       - File writing outside of the designated chart output path
       - Any other potentially harmful operations
    3. **Plotting Capability**: Verify that the code:
       - Uses matplotlib to create a chart
       - Saves the chart using plt.savefig() or similar
       - Does NOT use plt.show()
    4. **Data Completeness**: Ensure the code uses ALL rows from the DataFrame (should NOT use df.head() or limit data to a subset).
    5. **Code Quality**: Ensure the code is clean, follows best practices, and will execute successfully.
    
    CRITICAL: You MUST respond ONLY with valid JSON in this exact format (no code, no explanations, ONLY JSON):
    {{
        "is_valid": true,
        "feedback": ""
    }}
    
    OR if validation fails:
    {{
        "is_valid": false,
        "feedback": "Specific feedback on what needs to be fixed"
    }}
    
    DO NOT generate code. DO NOT provide explanations outside the JSON. ONLY return the JSON object.
    """
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Parse the JSON response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
        elif "```" in response_text:
            response_text = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
        
        import json
        result = json.loads(response_text)
        
        is_valid = result.get("is_valid", False)
        feedback = result.get("feedback", "Unknown error in validation")
        
        if is_valid:
            print("✓ Code validation passed")
            return {"code_feedback": ""}
        else:
            print(f"✗ Code validation failed: {feedback}")
            return {"code_feedback": feedback}
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {response_text}")
        # If we can't parse the response, assume it's invalid and use the raw response as feedback
        return {"code_feedback": f"Validation error: Could not parse LLM response. Raw response: {response_text}"}

def code_execution_node(state: AgentState):
    """
    Executes the validated code.
    """
    print("--- CODE EXECUTION ---")
    code = state["generated_code"]
    df = state["data"]
    chart_path = state["chart_path"]
    
    # Cleanup old charts
    if os.path.exists(chart_path):
        try:
            os.remove(chart_path)
        except Exception as e:
            print(f"Warning: Could not delete old chart: {e}")

    # Execute code
    local_vars = {"df": df, "plt": plt}
    try:
        exec(code, {}, local_vars)
        # Generate a simple analysis based on the data
        analysis = f"Chart generated successfully for {len(df)} products."
    except Exception as e:
        print(f"Error executing code: {e}")
        analysis = f"Error performing analysis: {e}"
        
    return {"analysis": analysis}


def json_generation_node(state: AgentState):
    """
    Uses LLM to generate JSON output with intelligent chart type selection.
    """
    print("--- JSON GENERATION ---")
    df = state["data"]
    query = state["query"]
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # Get data summary for LLM
    data_summary = df.to_string()
    columns = df.columns.tolist()
    
    prompt = f"""
    You are a data visualization expert. Analyze the following data and user query to generate JSON output suitable for Recharts (a React charting library).
    
    DATA:
    Columns: {columns}
    {data_summary}
    
    User Query: "{query}"
    
    TASK:
    1. Analyze the data structure and user query
    2. Determine the BEST chart type for this data from: "bar", "line", "pie", "area", "scatter"
    3. If the user explicitly requests a chart type (e.g., "pie chart", "line graph"), use that type
    4. However, if the requested chart type is NOT suitable for the data, explain why and suggest the best alternative
    5. Transform the data into a JSON format suitable for Recharts
    
    IMPORTANT GUIDELINES:
    - Bar charts: Good for comparing categories, discrete values
    - Line charts: Good for trends over time, continuous data
    - Pie charts: Good for showing parts of a whole (percentages), limited categories (< 10)
    - Area charts: Good for showing cumulative trends over time
    - Scatter charts: Good for showing relationships between two variables
    
    OUTPUT FORMAT (respond ONLY with valid JSON):
    {{
        "data": [
            {{"name": "Category1", "value": 100}},
            {{"name": "Category2", "value": 200}}
        ],
        "chartType": "bar",
        "reasoning": "Brief explanation of why this chart type was chosen"
    }}
    
    CRITICAL:
    - Use appropriate key names for Recharts (e.g., "name" for labels, "value" for numeric data)
    - Ensure all data from the DataFrame is included
    - chartType must be one of: "bar", "line", "pie", "area", "scatter"
    - Include a "reasoning" field explaining the chart type choice
    - Respond ONLY with the JSON object, no markdown code blocks
    """
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Clean response
    if "```json" in response_text:
        response_text = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
    elif "```" in response_text:
        response_text = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
    
    try:
        chart_json = json.loads(response_text)
        
        # Extract reasoning and remove from final JSON
        reasoning = chart_json.pop("reasoning", "No reasoning provided")
        
        print(f"✓ JSON generated: {len(chart_json.get('data', []))} items, chart type: {chart_json.get('chartType', 'unknown')}")
        print(f"  Reasoning: {reasoning}")
        
        return {"chart_json": chart_json}
        
    except Exception as e:
        print(f"✗ Error parsing LLM JSON response: {e}")
        print(f"Response was: {response_text}")
        
        # Fallback to simple conversion
        data_list = df.to_dict('records')
        chart_json = {
            "data": data_list,
            "chartType": "bar"
        }
        print(f"⚠ Using fallback JSON generation")
        return {"chart_json": chart_json}


def json_validation_node(state: AgentState):
    """
    Validates JSON output for safety and correctness before frontend rendering.
    """
    print("--- JSON VALIDATION ---")
    chart_json = state.get("chart_json")
    
    if not chart_json:
        print("✗ No JSON data to validate")
        return {"json_feedback": "No JSON data generated"}
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    json_str = json.dumps(chart_json)
    
    prompt = f"""
    You are a JSON security and quality validator for frontend applications. Analyze the following JSON data that will be rendered in a React.js frontend using Recharts.
    
    JSON TO VALIDATE:
    ```json
    {json_str}
    ```
    
    VALIDATION CHECKLIST:
    1. **Structure Validation**: Verify the JSON has:
       - A "data" field containing an array of objects
       - A "chartType" field with a valid chart type string
    2. **Security Check**: Ensure there are NO:
       - XSS attack vectors (script tags, event handlers, etc.)
       - SQL injection attempts
       - Malicious code or executable content
       - Suspicious URLs or external references
    3. **Data Integrity**: Verify:
       - All data values are appropriate types (numbers, strings)
       - No null or undefined values that could break rendering
       - Data is consistent and makes sense
    4. **Frontend Safety**: Ensure the JSON is safe to render in a React component
    
    CRITICAL: You MUST respond ONLY with valid JSON in this exact format (no explanations, ONLY JSON):
    {{
        "is_valid": true,
        "feedback": ""
    }}
    
    OR if validation fails:
    {{
        "is_valid": false,
        "feedback": "Specific security or structural issues found"
    }}
    
    DO NOT provide explanations outside the JSON. ONLY return the JSON object.
    """
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Parse the JSON response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
        elif "```" in response_text:
            response_text = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL).group(1)
        
        result = json.loads(response_text)
        is_valid = result.get("is_valid", False)
        feedback = result.get("feedback", "Unknown validation error")
        
        if is_valid:
            print("✓ JSON validation passed - safe for frontend")
            return {"json_feedback": ""}
        else:
            print(f"✗ JSON validation failed: {feedback}")
            return {"json_feedback": feedback}
            
    except Exception as e:
        print(f"Error parsing validation response: {e}")
        return {"json_feedback": f"Validation error: {e}"}


def should_continue(state: AgentState):
    """
    Determines whether to proceed to execution or retry generation.
    """
    feedback = state.get("code_feedback", "")
    retry_count = state.get("retry_count", 0)
    
    if not feedback:
        # No feedback means code is valid
        return "code_exec"
    
    if retry_count >= 3:
        # Max retries reached, proceed anyway (or could fail gracefully)
        print("--- MAX RETRIES REACHED, PROCEEDING WITH POTENTIALLY FLAWED CODE ---")
        return "code_exec"
    
    return "code_gen"

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("data_extract", data_extraction_node)
workflow.add_node("code_gen", code_generation_node)
workflow.add_node("feasibility_check", feasibility_check_node)
workflow.add_node("code_exec", code_execution_node)
workflow.add_node("json_gen", json_generation_node)
workflow.add_node("json_validation", json_validation_node)

workflow.set_entry_point("data_extract")

workflow.add_edge("data_extract", "code_gen")
workflow.add_edge("code_gen", "feasibility_check")

workflow.add_conditional_edges(
    "feasibility_check",
    should_continue,
    {
        "code_exec": "code_exec",
        "code_gen": "code_gen"
    }
)

workflow.add_edge("code_exec", "json_gen")
workflow.add_edge("json_gen", "json_validation")
workflow.add_edge("json_validation", END)

app = workflow.compile()



def frame_maker(sql_result):
    df = pd.DataFrame(sql_result["results"])

    # Convert Decimal to float
    for col in df.columns:
        if df[col].dtype == "object" and isinstance(df[col].iloc[0], Decimal):
            df[col] = df[col].astype(float)

    # Clean auto SQL column names
    rename_map = {}
    for col in df.columns:
        clean = col.split("(")[0].replace(")", "").strip()
        rename_map[col] = clean

    df.rename(columns=rename_map, inplace=True)
    return df


def dataframe_to_graph(sql_result, task):

    user_query = task.get("chartprompt")
    df = frame_maker(sql_result)
    inputs = {"query": user_query,"data":df}

    result = app.invoke(inputs)

    return {
        "resultpath": {
            "query": user_query,
            "data": df,
            "chart_path": result.get("chart_path")
        }
    }
