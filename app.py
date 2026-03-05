import streamlit as st
import os
import base64
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Multi-Lingual Content Localizer",
    page_icon="3-Dots-Blog-Banner-_Multilingual-Content_Pune.webp",
    layout="wide"
)

# ------------------------------------------------
# BACKGROUND IMAGE
# ------------------------------------------------
def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

set_background("bg1.jpg")

# ------------------------------------------------
# GLOBAL UI STYLING
# ------------------------------------------------
st.markdown("""
<style>

/* MAIN TITLE */
p.main-title{
text-align:center;
font-size:64px;
font-weight:800;
background: linear-gradient(90deg,#06b6d4,#6366f1,#a855f7);
background-clip:text;
-webkit-background-clip:text;
color:transparent;
margin-bottom:10px;
}

/* SUBTITLE */
p.subtitle{
text-align:center;
font-size:28px;
font-weight:500;
color:white;
margin-bottom:40px;
}

/* GLASS CARD */
.glass{
background:rgba(255,255,255,0.15);
backdrop-filter:blur(12px);
padding:24px;
border-radius:18px;
border:1px solid rgba(255,255,255,0.3);
margin-bottom:12px;
}

/* LABELS */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label {
font-size:40px !important;
font-weight:700 !important;
color:white !important;
text-shadow:0px 0px 8px rgba(255,255,255,0.5);
}

/* TEXT AREA */
textarea{
font-size:18px !important;
}

/* PLACEHOLDER */
textarea::placeholder{
font-size:18px;
color:#ddd;
}

/* SELECT BOX TEXT */
div[data-baseweb="select"] *{
font-size:18px !important;
}

/* BUTTON */
div.stButton > button {
background:linear-gradient(90deg,#4A6CF7,#6A5ACD);
color:white;
border-radius:25px;
height:55px;
font-size:20px;
font-weight:700;
border:none;
width:100%;
}

/* RESULT CARDS */
.result-card {
background: rgba(255,255,255,0.18);
backdrop-filter: blur(14px);
padding:30px;
border-radius:18px;
border:1px solid rgba(255,255,255,0.35);
box-shadow:0px 10px 32px rgba(0,0,0,0.25);
margin-top:20px;
}

/* RESULT TITLES */
.result-title{
font-size:22px;
font-weight:700;
margin-bottom:20px;
color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown(
"""
<h1 style='
text-align:center;
font-size:64px;
font-weight:900;
color:#38bdf8;
text-shadow:0px 0px 20px rgba(56,189,248,0.7);
'>
🌐 AI Multi-Lingual Content Localizer
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Transcreate and culturally adapt your content using AI</p>',
unsafe_allow_html=True
)

# ------------------------------------------------
# API KEY
# ------------------------------------------------
google_api_key = None

if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

# ------------------------------------------------
# LLM
# ------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
    google_api_key=google_api_key
)

# ------------------------------------------------
# OUTPUT SCHEMA
# ------------------------------------------------
class TranscreationOutput(BaseModel):
    culturally_adapted_text: str
    tone: str
    cultural_notes: str

structured_llm = llm.with_structured_output(TranscreationOutput)

# ------------------------------------------------
# PROMPT
# ------------------------------------------------
structured_prompt = PromptTemplate(
    input_variables=["source_text", "target_language", "region"],
    template="""
You are a professional cultural transcreation expert.

STRICT INSTRUCTIONS:
- Preserve exact meaning
- Do not invent new information
- Adapt culturally but keep intent same

Source text:
"{source_text}"

Target language: {target_language}
Region: {region}
"""
)

chain = structured_prompt | structured_llm

# ------------------------------------------------
# REGION LANGUAGE MAP
# ------------------------------------------------
region_language_map = {
"India":["Hindi","English"],
"Spain":["Spanish"],
"Mexico":["Spanish"],
"Argentina":["Spanish"],
"France":["French"],
"Canada":["English","French"],
"Germany":["German"],
"Japan":["Japanese"],
"China":["Chinese"],
"UAE":["Arabic"]
}

# ------------------------------------------------
# INPUT UI
# ------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("""
            <h3 style='color:white;'>
                ✍️ Enter Content
            </h3>
            """, unsafe_allow_html=True)
#st.subheader("✍️ Enter Content")
source_text = st.text_area(
"Source Text",
placeholder="Type or paste the content you want to culturally adapt...",
height=160
)

col1,col2 = st.columns(2)

with col1:
    region = st.selectbox(
    "Region",
    ["Select Region"] + list(region_language_map.keys())
    )

if region == "Select Region":
    language_options = ["Select Language"]
    disable_language = True
else:
    language_options = ["Select Language"] + region_language_map[region]
    disable_language = False

with col2:
    target_language = st.selectbox(
    "Target Language",
    language_options,
    disabled=disable_language
    )

generate = st.button("🚀 Generate Transcreation")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# RESULT PLACEHOLDER
# ------------------------------------------------
response_placeholder = st.empty()

# ------------------------------------------------
# GENERATE
# ------------------------------------------------
if generate:

    response_placeholder.empty()

    if (not source_text or region == "Select Region" 
        or target_language == "Select Language"):
        st.error("Please fill all fields.")
    else:

        with st.spinner("🤖 AI is adapting content for the selected region..."):

            result = chain.invoke({
                "source_text": source_text,
                "target_language": target_language,
                "region": region
            })

        with response_placeholder.container():
            st.markdown("""
            <style>
            div[role="alert"][class*="stAlert"]{
                box-shadow:
                    0 6px 18px rgba(0, 0, 0, 0.22),
                    0 0 16px rgba(255, 255, 255, 0.40), 
                    0 0 32px rgba(255, 255, 255, 0.25);
                border-radius: 12px !important;}
            </style>
            """, unsafe_allow_html=True)
            st.success("✅ Transcreation Generated Successfully!")

            col1,col2,col3 = st.columns(3)

            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<p class="result-title">🌎 Adapted Text</p>', unsafe_allow_html=True)
                st.markdown(f'<p style = "color:white;">{result.culturally_adapted_text}</p>', unsafe_allow_html=True)
                #st.write(result.culturally_adapted_text)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<p class="result-title">🎭 Tone</p>', unsafe_allow_html=True)
                st.markdown(f'<p style = "color:white;">{result.tone}</p>', unsafe_allow_html=True)
                #st.write(result.tone)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<p class="result-title">📝 Cultural Notes</p>', unsafe_allow_html=True)
                st.markdown(f'<p style = "color:white;">{result.cultural_notes}</p>', unsafe_allow_html=True)
                #st.write(result.cultural_notes)

                st.markdown('</div>', unsafe_allow_html=True)










