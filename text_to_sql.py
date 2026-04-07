PLANNER_PROMPT = """
You are given a database schema and a user's natural language request.

Return ONLY a valid JSON object. No explanations. No comments. No text outside JSON.

The JSON must follow this exact structure:

{
  "intent": "<one sentence summary>",
  "tables": ["..."],
  "joins": [
      {"left": "table.column", "right": "table.column"}
  ],
  "select": ["table.column"],
  "filters": [
      {"expr": "<SQL boolean expression>", "explain": "<short reason>"}
  ],
  "group_by": ["table.column"],
  "order_by": [
      {"expr": "table.column", "dir": "ASC" }
  ],
  "limit": 10,
  "Feasible": true
}

Rules:
1. Omit any field that is not needed.
2. Only use table and column names that appear in the schema.
3. No SQL except within 'expr' values inside filters or order_by.
4. If the request cannot be satisfied using the schema:
   - set "select": []
   - set "Feasible": false
   - describe the issue in "intent".

"""


from models.chatgrok import llm
from get_schema import getschema_agent

from typing import Annotated, List
from operator import add
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END



# from langchain.output_parsers import JsonOutputParser
# from langchain.prompts import ChatPromptTemplate


class ToolState(TypedDict, total=False):
    nlpquery: str
    schema: Annotated[list, add]
    feasibility: bool
    plan: object
    plan_feasibility: bool
    sql: Annotated[list, add]
    error: Annotated[list, add]
    results_sql: Annotated[list, add]



def superbot(state: ToolState):
    return {"schema": [getschema_agent()]}

def feasibility_checker(state: ToolState):
    prompt = f"""
You are a strict SQL-safety classifier.

Your job: Determine if the user's request would result in a destructive SQL operation.

Destructive SQL includes DELETE, DROP, TRUNCATE, ALTER, UPDATE, INSERT, CREATE, REPLACE, RENAME.

Rules:
- If destructive → respond ONLY: 'False'
- If read-only → respond ONLY: 'True'
- No explanations, just the one-word decision.

User request: "{state['nlpquery']} "
"""

    out = llm.invoke([{"role": "user", "content": prompt}])
    
    # FORCE conversion to plain string
    decision = str(out.content).strip().lower()

   

    safe = not ("false" in decision)
    return {"feasibility": safe}
    

import json

def planner_bot(state: ToolState):
    
    prompt = f"""
Schema:
{state['schema']}

User request:
{state['nlpquery']}

Follow JSON format exactly:
{PLANNER_PROMPT}
"""
    out = llm.invoke([{"role": "user", "content": prompt}])
    raw = out.content.strip()

    # Clean code fences
    cleaned = raw
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    # Try parse
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {"plan_feasibility": False, "plan": raw}

    if parsed.get("Feasible") is False:
        return {"plan_feasibility": False, "plan": parsed}

    return {"plan_feasibility": True, "plan": parsed}

import re

def extract_sql(text: str):
    # Remove code fences
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    # Look for SELECT... FROM
    m = re.search(r"(SELECT[\s\S]+?)(;|$)", text, re.I)
    if m:
        return m.group(1).strip()

    # If it starts with SELECT, return everything
    if text.strip().upper().startswith("SELECT"):
        return text.strip()

    return ""



def sql_synthesizer(state: ToolState):
    schema_text = "\n".join(state["schema"])
    plan_text = json.dumps(state["plan"], indent=2)

    prompt = f"""
Write only SQL. No explanations. No markdown. No text. Only SQL.

Schema:
{schema_text}

Plan:
{plan_text}
"""

    out = llm.invoke([{"role":"user","content":prompt}])
    sql = extract_sql(out.content)
    return {"sql": [sql]}

# ============================================================
#  5. SQL GROUNDING + AUTO-REPAIR LOOP
# ============================================================
from sqlalchemy import create_engine, text, inspect
host="localhost"
user="root"
password=""
database="ecommerce"
engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

def grounder_bot(state: ToolState):
    sql_query = state["sql"][-1]

    for i in range(3):
        try:
            with engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql_query}"))
            return {"sql":[sql_query]}

        except Exception as e:
            err = str(e)
            state["error"].append(err)

            repair_prompt = f"""
SQL failed validation:

SQL:
{sql_query}

Error:
{err}

Fix SQL. Return only SELECT statement.
"""

            fixed = llm.invoke([{"role":"user","content":repair_prompt}])
            sql_query = extract_sql(fixed.content)
        

    return False
    
#################################################################
# sql fire wall

import re

def sql_firewall(sql: str) -> bool:
    if not sql:
        return False

    s = sql.strip().lower()

    # Only allow SELECT queries
    if not s.startswith("select"):
        return False

    # List of destructive keywords to block
    destructive = [
        "delete",
        "drop",
        "truncate",
        "alter",
        "update",
        "insert",
        "create",
        "replace",
        "rename"
    ]

    # Reject if any destructive keyword is present
    for op in destructive:
        if re.search(rf"\b{op}\b", s):
            return False

    # Disallow multiple statements
    # if ";" in s[:-1]:
    #     return False

    return True
# ============================================================
#  7. EXECUTOR
# ============================================================

def executor_bot(state: ToolState):
    sql_query = state["sql"][-1].strip().lower()
    if not sql_firewall(sql_query):
        return {"error": ["SQL query failed firewall checks."]}
    sql_query = sql_query
    final_results = []

    with engine.connect() as conn:
        rows = conn.execute(text(sql_query)).fetchall()

        for r in rows:
            row_dict = dict(r._mapping)
            final_results.append(row_dict)

    return {"results_sql": final_results}



graph = StateGraph(ToolState)

graph.add_node("superbot", superbot)
graph.add_node("feasibility_checker", feasibility_checker)
graph.add_node("planner_bot", planner_bot)
graph.add_node("sql_synthesizer", sql_synthesizer)
graph.add_node("grounder_bot", grounder_bot)
graph.add_node("executor_bot", executor_bot)


graph.add_edge(START, "feasibility_checker")
graph.add_conditional_edges(
    "feasibility_checker",
    lambda out: "END" if out["feasibility"] is False else "superbot",
    {"END": END, "superbot": "superbot"},
)


graph.add_edge("superbot", "planner_bot")

graph.add_conditional_edges(
    "planner_bot",
    lambda out: "END" if out["plan_feasibility"] is False else "sql_synthesizer",
    {"END": END, "sql_synthesizer": "sql_synthesizer"},
)

graph.add_edge("sql_synthesizer","grounder_bot")

graph.add_conditional_edges(
    "grounder_bot",
    lambda out: "END" if out is False else "executor_bot",
    {"END": END, "executor_bot": "executor_bot"},
)

graph.add_edge("executor_bot", END)


app = graph.compile()




def run_text_to_sql(nlpquery: str):
    out = app.invoke({
        "nlpquery": nlpquery
    })

    result={}

    if out.get("feasibility") is False:
        result["success_status"] = False
        result["reason"] = "Request deemed unsafe by feasibility checker."
        return result
    elif out.get("plan_feasibility") is False:
        result["success_status"] = False
        result["reason"] = "Request cannot be satisfied with the given schema."
        # result["plan"] = out.get("plan")
        return result
    else:
        output_results = out.get("results_sql", [])
        if not output_results:
            result["success_status"] = False
            result["reason"] = "No results returned from SQL execution."
            return result
        else:
            result["success_status"] = True
            result["results"] = output_results
            return result

    

    

