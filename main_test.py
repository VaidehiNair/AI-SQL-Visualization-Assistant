from text_to_sql import run_text_to_sql
from models.openaiapipoint import llm
from get_schema import getschema_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from query_parser import parse_user_request
# from summerizer import dataframe_to_graph
from imagereader import image_decriber
from charmaker import dataframe_to_graph

 


user_prompt=input("Enter the user prompt: ")

endresult_validator={}

tasks=parse_user_request(user_prompt,endresult_validator)
print("------------------------------------------------------------------")

print(tasks)
print("------------------------------------------------------------------")
if tasks.get('sql_intent'):
    sql_result=run_text_to_sql(tasks.get("sql_intent"))

    
print("------------------------------------------------------------------")

print(sql_result)


if tasks.get('chartprompt'):
    endresult_validator["chartprompt_status"]=True
else:
    endresult_validator["chartprompt_status"]=False


if sql_result.get('success_status') and tasks.get('chartprompt'):
    
    result = dataframe_to_graph(sql_result,tasks)
    endresult_validator["resultpath"]=result
    plot_path = result["resultpath"].get("chart_path")
    endresult_validator["analyis"]=image_decriber(plot_path)








print("------------------------------------------------------------------")

print(endresult_validator)







