from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import base64  # <-- You forgot this import

googleapikey = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage



def image_decriber(filepath:str):

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    imagepath = f"./{filepath}"

    with open(imagepath, "rb") as imagefile:
        encoded_image = base64.b64encode(imagefile.read()).decode("utf-8")

    print("Image encoded successfully!")



    message_local=HumanMessage(

        content=[

            {"type":"text","text":"Act like a dataanalyst analyis the image of the plot that is got after anaysis and give a brief decription in 100 words" },
            {"type":"image_url","image_url":f"data:image/jpeg;base64,{encoded_image}"}
        ]
    )


    result=model.invoke([message_local])
    # print(result)
    return result.content


