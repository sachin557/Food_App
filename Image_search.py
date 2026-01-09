from dotenv import load_dotenv
import streamlit as st
load_dotenv()
import os
from google import genai
from PIL import Image
llm=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
prompt="""
From the given food image get the nutrition details for each seapartely if multiple food in image else just display for the single food nutrition,
Nutitions:
Calories:
Protien:
Fat:
Carbohydrate:
and at the end give the overall total nutrition of all the multiple food below like 
Total Calories:
Total Protien:
Total Fat:
Total Carbs:
"""
def get_response(prompt,image):
    response=llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[prompt,image]
    )
    return response.text
st.set_page_config(page_title="Food Nutrition Display")
st.header("Food App")

files=st.file_uploader("choose an image....",type=["jpg","jpeg","png"])
image=""
if files is not None:
    image=Image.open(files)
    st.image(image,caption="Uploaded image",use_column_width=True)
submit = st.button("Submit")
response=get_response(prompt,image)
st.write(response)    