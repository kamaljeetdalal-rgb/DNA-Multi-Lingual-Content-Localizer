import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Cultural Transcreation System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 AI Multi-Lingual Cultural Transcreation")
st.write("Gemini-powered Localization & Transcreation System")

# -----------------------------
# LOAD API KEY (Streamlit Cloud)
# -----------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# -----------------------------
# INITIALIZE GEMINI MODEL
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
structured_prompt = PromptTemplate(
    input_variables=["source_text", "target_language", "region"],
    template="""
You are a professional cultural transcreation expert.

Your task is to transcreate the given content into {target_language}
for the region: {region}.

Ensure:
- Natural, culturally appropriate tone
- Native style of speech
- Emotional alignment with local expression

Return output strictly in valid JSON format with the following structure:

{{
  "original_text": "{source_text}",
  "target_language": "{target_language}",
  "region": "{region}",
  "culturally_adapted_text": "<final adapted text>",
  "tone": "<describe tone used>",
  "cultural_notes": "<brief explanation of cultural adaptation>"
}}
"""
)

chain = structured_prompt | llm

# -----------------------------
# UI INPUTS
# -----------------------------
source_text = st.text_area("Enter Source Text")
target_language = st.text_input("Target Language", value="Hindi")
region = st.text_input("Region", value="India")

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("Generate Transcreation"):

    if not source_text:
        st.warning("Please enter source text.")
    else:
        with st.spinner("Generating culturally adapted content..."):
            response = chain.invoke({
                "source_text": source_text,
                "target_language": target_language,
                "region": region
            })

            try:
                output = json.loads(response.content)

                st.success("Transcreation Generated Successfully!")

                st.subheader("🌎 Culturally Adapted Text")
                st.write(output["culturally_adapted_text"])

                st.subheader("🎭 Tone")
                st.write(output["tone"])

                st.subheader("📝 Cultural Notes")
                st.write(output["cultural_notes"])

                st.subheader("📦 Full JSON Output")
                st.json(output)

            except Exception:
                st.error("Model did not return valid JSON.")
                st.write(response.content)
