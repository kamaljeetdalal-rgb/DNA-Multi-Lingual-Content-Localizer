import streamlit as st
import os
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="AI Multi-Lingual Content Localizer",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Multi-Lingual Content Localizer")
st.markdown("Transcreate & culturally adapt your content using Gemini AI")

# -------------------------
# Sidebar - API Key
# -------------------------
st.sidebar.header("🔐 Configuration")

api_key = st.sidebar.text_input(
    "Enter your GOOGLE API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter your Google API Key in the sidebar.")
    st.stop()

# -------------------------
# Initialize Gemini
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    google_api_key=api_key
)

# -------------------------
# Structured Prompt
# -------------------------
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

# -------------------------
# User Inputs
# -------------------------
st.subheader("✍️ Enter Content")

source_text = st.text_area("Source Text")

col1, col2 = st.columns(2)

with col1:
    target_language = st.text_input("Target Language", placeholder="e.g., Hindi")

with col2:
    region = st.text_input("Region", placeholder="e.g., India")

# -------------------------
# Generate Button
# -------------------------
if st.button("🚀 Generate Transcreation"):

    if not source_text or not target_language or not region:
        st.error("Please fill all fields.")
    else:
        with st.spinner("Generating culturally adapted content..."):
            try:
                response = chain.invoke({
                    "source_text": source_text,
                    "target_language": target_language,
                    "region": region
                })

                raw_text = response.content.strip()

                # Clean markdown blocks if any
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"```json|```", "", raw_text).strip()

                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                if json_match:
                    output = json.loads(json_match.group(0))

                    st.success("✅ Transcreation Generated Successfully!")

                    st.subheader("🌎 Culturally Adapted Text")
                    st.write(output.get("culturally_adapted_text", ""))

                    st.subheader("🎭 Tone")
                    st.write(output.get("tone", ""))

                    st.subheader("📝 Cultural Notes")
                    st.write(output.get("cultural_notes", ""))

                    st.subheader("📦 Full JSON Output")
                    st.json(output)

                else:
                    st.error("Could not extract valid JSON.")
                    st.write(raw_text)

            except Exception as e:
                st.error(f"Error: {e}")
