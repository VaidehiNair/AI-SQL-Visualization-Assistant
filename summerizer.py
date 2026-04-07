import os
import re
from typing import TypedDict, Annotated, Optional
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from decimal import Decimal
import uuid


# Load environment variables
load_dotenv()

# Define the state
class AgentState(TypedDict):
    query: str
    data: pd.DataFrame
    analysis: str
    summary: str
    chart_path: Optional[str]

# Nodes
def data_extraction_node(state: AgentState):
    """
    Mock Data Extraction Node.
    In a real app, this would execute the SQL against a database.
    """
    print("--- DATA EXTRACTION (MOCKED) ---")
    # Convert sample data to DataFrame
    df = state['data']
    return {"data": df}

def data_analysis_node(state: AgentState):
    """
    Uses Llama 3 to generate Python code for analysis and plotting.
    """
    print("--- DATA ANALYSIS (Custom LLM) ---")
    df = state["data"]
    query = state["query"]
    
    # Ensure exports directory exists
    os.makedirs("exports/charts", exist_ok=True)
    chart_path = f"exports/charts/plot_{uuid.uuid4().hex}.png"
    
    # Initialize LLM
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # Prompt for code generation
    prompt = f"""
    You are a Python Data Analyst. You have a pandas DataFrame named `df` with the following columns: {df.columns.tolist()}.
    Sample data:
    {df.head().to_string()}
    
    User Query: "{query}"
    
    Please generate Python code to:
    1. Perform a brief textual analysis of the data based on the query. Store this analysis in a variable named `analysis_result` (string).
    2. Create a relevant chart (e.g., bar chart) using matplotlib.
    3. Save the chart to "{chart_path}".
    4. Do NOT use plt.show().
    
    IMPORTANT: 
    - The code must be valid Python.
    - Assume `df` and `plt` are already imported and available.
    - Do not wrap the code in markdown blocks (like ```python ... ```). Just provide the code.
    - Assign the textual analysis to the variable `analysis_result`.
    """
    
    response = llm.invoke(prompt)
    code = response.content
    
    # Clean code (remove markdown blocks if present)
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)
    code = code.strip()
    
    print(f"Generated Code:\n{code}\n")
    
    # Execute code
    local_vars = {"df": df, "plt": plt, "analysis_result": "No analysis generated."}
    try:
        exec(code, {}, local_vars)
        analysis = local_vars.get("analysis_result", "No analysis generated.")
    except Exception as e:
        print(f"Error executing generated code: {e}")
        analysis = f"Error performing analysis: {e}"   
    
    return {"analysis": analysis, "chart_path": chart_path}

def summarization_node(state: AgentState):
    """
    Summarizes the findings in layman's terms.
    """
    print("--- SUMMARIZATION ---")
    query = state["query"]
    analysis = state["analysis"]
    data_str = state["data"].to_string()
    chart_path = state.get("chart_path")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    prompt = f"""
    User Query: {query}
    
    Data:
    {data_str}
    
    Analysis:
    {analysis}
    
    Please provide a concise, layman-friendly summary of these results. 
    Focus on the key insights (who spent the most, etc.) and avoid technical jargon.
    Mention that a chart has been generated at {chart_path} if applicable.
    """
    
    response = llm.invoke(prompt)
    return {"summary": response.content}

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("data_extract", data_extraction_node)
workflow.add_node("data_analysis", data_analysis_node)
workflow.add_node("summarize", summarization_node)

workflow.set_entry_point("data_extract")

workflow.add_edge("data_extract", "data_analysis")
workflow.add_edge("data_analysis", "summarize")
workflow.add_edge("summarize", END)

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
            "analysis": result.get("analysis"),
            "summary": result.get("summary"),
            "chart_path": result.get("chart_path")
        }
    }



    
