import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSequence

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Multi-Lingual Content Localizer",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Multi-Lingual Content Localizer")
st.markdown("Localize your content using Gemini AI")

# ----------------------------
# API Key Setup
# ----------------------------
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.warning("Please add your GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

# ----------------------------
# Model
# ----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# ----------------------------
# Prompt Template
# ----------------------------
prompt_template = PromptTemplate(
    input_variables=["source_text", "target_language", "region"],
    template="""
You are a professional localization expert.

Translate the following text into {target_language} for the {region} region.

Make it culturally appropriate and natural sounding.

Text:
{source_text}
"""
)

# ----------------------------
# LCEL Chain
# ----------------------------
chain = RunnableSequence(
    prompt_template,
    model,
    StrOutputParser()
)

# ----------------------------
# UI Inputs
# ----------------------------
source_text = st.text_area("Enter text to localize")

target_language = st.selectbox(
    "Select Target Language",
    ["Hindi", "Spanish", "French", "German", "Japanese"]
)

region = st.text_input("Target Region (e.g., India, Mexico, Canada)")

# ----------------------------
# Generate Button
# ----------------------------
if st.button("Localize Content"):
    if source_text and region:
        with st.spinner("Localizing..."):
            result = chain.invoke({
                "source_text": source_text,
                "target_language": target_language,
                "region": region
            })
        st.success("Localized Content:")
        st.write(result)
    else:

        st.error("Please provide all inputs.")
