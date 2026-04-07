from text_to_sql import run_text_to_sql
from models.openaiapipoint import llm
from get_schema import getschema_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import matplotlib.pyplot as plt
import pandas as pd
import re


# ---------------------------
# PROMPT
# ---------------------------
PROMPT = """
You analyze user requests that may involve data retrieval, SQL generation, or chart creation.

Extract and output ONLY valid JSON with these fields:

- sql_intent: A natural-language description of the data to retrieve.
  No SQL keywords. No query structure.
  Only describe the dataset, grouping, metrics, and filters required.
  Example: "Get total revenue grouped by month for the current year."
  If we cannot make requested, set sql_intent to `False`.


- chartprompt: If the user requests any visualization 
  (pie, bar, line, histogram, scatter, area, etc.),
  produce a clean, minimal instruction suitable for PandasAI SmartDataframe.
  The instruction must:
    * Clearly state the chart type.
    * Name the column(s) to plot.
    * Include a required figure size instruction like:
        "Set figure size to 10 by 6."
    * Use matplotlib.
    * Use ASCII text only.
    * Contain no SQL and no dataset reasoning.
  Example:
    "Create a histogram of sepal_length using matplotlib. Set figure size to 10 by 6. Use ASCII text only."

  If no chart is requested, set chartprompt to `False`.

Rules:
1. Never generate SQL or SQL-like language.
2. sql_intent must describe data logically, not programmatically.
3. chartprompt must only describe how to draw the chart, not what data to query.
4. Always include figure size instructions in chartprompt.
5. Output must be strictly valid JSON.

Schema for reference:
{schema}

User query:
{query}

Return ONLY valid JSON. No explanations.
"""


def parse_user_request(nlpquery: str,endresult_validator:dict):
    schema = getschema_agent()
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | llm | JsonOutputParser()
    endresult_validator["parser_useragent_status"]=True
    return chain.invoke({"schema": schema, "query": nlpquery})




