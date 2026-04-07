import matplotlib
matplotlib.use("Agg")   # <--- REQUIRED FIX
import matplotlib.pyplot as plt

import pandas as pd
from decimal import Decimal
from models.openaiapipoint import llm
from pandasai import SmartDataframe
import os
import uuid
from pandasai.helpers.cache import Cache



# Disable DuckDB cache
class NoOpCache(Cache):
    def __init__(self):
        self.connection = None

    def get(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None


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

    chatprompt = task.get("chartprompt")
    df = frame_maker(sql_result)

    # PandasAI configured safely
    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "cache": NoOpCache(),
            "log": False,              # IMPORTANT
            "enable_cache": False,     # IMPORTANT
            "save_logs": False         # Avoid Unicode issues
        }
    )

    # Ask PandasAI to generate only code, not save files
    result = sdf.chat(chatprompt)

    # Create folder
    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save ONLY your own file
    filename = f"plot_{uuid.uuid4().hex}.png"
    filepath = os.path.join(plot_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "plot_path": filepath,
        "agent_response": "Plot created successfully"
    }
