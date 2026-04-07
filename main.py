import streamlit as st
from text_to_sql import run_text_to_sql
from query_parser import parse_user_request
# from summerizer import dataframe_to_graph
from charmaker import dataframe_to_graph
import pandas as pd
from decimal import Decimal
from imagereader import image_decriber


def clean_sql_results(sql_result):
    """Convert Decimal to float and return a pandas DataFrame."""
    df = pd.DataFrame(sql_result["results"])

    for col in df.columns:
        if df[col].dtype == "object":
            if isinstance(df[col].iloc[0], Decimal):
                df[col] = df[col].astype(float)

    return df


st.title("AI SQL + Visualization Assistant")

# 1. USER INPUT
user_prompt = st.text_input("Ask something about your data")

if user_prompt:

    # 2. NL PARSER
    with st.spinner("Understanding your request..."):
        validation = {}
        tasks = parse_user_request(user_prompt, validation)

    # st.write("### Parsed Request")
    # st.json(tasks)

    # 3. SQL EXECUTION
    if tasks.get("sql_intent"):
        with st.spinner("Running SQL..."):
            sql_result = run_text_to_sql(tasks["sql_intent"])

        # st.write("### SQL Result")
        # st.json(sql_result)

        if not sql_result["success_status"]:
            st.error("SQL execution failed")
            st.stop()

        df = clean_sql_results(sql_result)

        st.write("### Dataframe")
        st.dataframe(df)

        # 4. IF CHART REQUEST EXISTS
        if tasks.get("chartprompt"):
            with st.spinner("Generating plot..."):
                result = dataframe_to_graph(sql_result, tasks)
            if not result or "resultpath" not in result:
                st.error("Plot generation failed, no resultpath returned")
                st.stop()

            plot_path = result["resultpath"].get("chart_path")

            if not plot_path:
                st.error("Plot generation failed, no chart_path returned")
                st.stop()

            st.image(f"./{plot_path}")

            with st.spinner("Generating final summary..."):

                res=image_decriber(plot_path)

                if res:
                    st.write(res)
                    st.stop()
                else:
                    st.error("No summary available")
                    st.stop()

    st.stop()





