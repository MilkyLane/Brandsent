import streamlit as st
import yt_dlp
import whisper
import os
import json
import shutil
import requests
import tempfile

# --- RAG LIBRARIES ---
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
OLLAMA_MODEL = "llama3.2" 
WHISPER_MODEL_SIZE = "base"

# --- PAGE SETUP ---
st.set_page_config(page_title="BrandSent", layout="wide")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;} 
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🛡️ BrandSent: Autonomous Video Intelligence Agent")
st.markdown("### GenAI-Powered Brand Safety & Alignment System")

# --- SESSION STATE ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

# --- LAYOUT ---
st.divider()
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Define Brand Identity")
    brand_name = st.text_input("Brand Name", "EcoLife Sneakers")
    brand_guidelines = st.text_area(
        "Brand Values & Guidelines",
        "We are a sustainable, high-energy, family-friendly brand. We value environmental protection and active lifestyles. "
        "Strictly avoid: politics, swearing, laziness, violence, or wasteful behavior.",
        height=150
    )

with col2:
    st.subheader("2. Input Content")
    video_url = st.text_input("Paste YouTube Short URL", "")
    process_btn = st.button("🚀 Analyze & Index Video", type="primary")

# --- HELPER FUNCTIONS ---

def download_audio(url):
    """Downloads audio from YouTube/TikTok"""
    output_filename = "audio"
    if os.path.exists(f"{output_filename}.mp3"):
        os.remove(f"{output_filename}.mp3")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'outtmpl': output_filename,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{output_filename}.mp3"

def transcribe_audio(file_path):
    """Transcribes audio using Whisper"""
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    result = model.transcribe(file_path)
    return result["text"]

def analyze_safety(transcript, brand, guidelines):
    """Safety Check using standard Ollama API"""
    prompt = f"""
    You are a Senior Brand Intelligence Officer for {brand}.
    GUIDELINES: {guidelines}
    
    TRANSCRIPT: "{transcript[:4000]}"... (truncated)
    
    TASK: Analyze the transcript and provide a JSON output with TWO scores.
    1. Safety Score (0-5): Is the content safe/non-toxic? (5 is perfectly safe)
    2. Alignment Score (0-100): How well does it fit the specific brand vibes/values? (100 is perfect match)
    
    OUTPUT FORMAT (JSON ONLY):
    {{
        "safety_score": <number 0-5>,
        "alignment_score": <number 0-100>,
        "reasoning": "<concise executive summary>",
        "flags": ["<risk1>", "<risk2>"]
    }}
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    )
    return response.json()['response']

def setup_rag(transcript_text):
    """Creates a TEMPORARY Vector DB from the transcript"""

    temp_dir = tempfile.mkdtemp()
    
    doc = Document(page_content=transcript_text, metadata={"source": "video"})
    embedding_function = OllamaEmbeddings(model=OLLAMA_MODEL)
    

    vectorstore = Chroma.from_documents(
        documents=[doc],
        embedding=embedding_function,
        persist_directory=temp_dir  
    )
    return vectorstore.as_retriever()

# --- MAIN LOGIC ---

if process_btn and video_url:
    with st.spinner("Processing Video Pipeline... (Download -> Whisper -> Vector DB)"):
        try:
           
            audio = download_audio(video_url)
            transcript = transcribe_audio(audio)
            st.session_state.transcript = transcript
        
            analysis = analyze_safety(transcript, brand_name, brand_guidelines)
            data = json.loads(analysis)
      
            st.session_state.analysis_data = data
       
            st.session_state.retriever = setup_rag(transcript)
            
            st.success("✅ Video Indexed for RAG Chat!")
            
        except Exception as e:
            st.error(f"Pipeline Error: {e}")

# --- DISPLAY ANALYSIS STATS ---
if st.session_state.analysis_data:
    st.divider()
    
    m1, m2, m3, m4 = st.columns([1, 1, 2, 1])
    
    data = st.session_state.analysis_data
    
    with m1:
        st.metric("Safety Score", f"{data['safety_score']}/5", delta="Toxic" if data['safety_score'] < 3 else "Safe")
    
    with m2:
        st.metric("Brand Alignment", f"{data['alignment_score']}%", delta="Misaligned" if data['alignment_score'] < 50 else "Aligned")
    
    with m3:
        st.caption("EXECUTIVE SUMMARY")
        st.write(f"**{data['reasoning']}**")
    
    with m4:
        st.caption("RISK FLAGS")
        if data['flags']:
            for flag in data['flags']:
                st.error(f"🚩 {flag}")
        else:
            st.success("No Risks Found")

if st.session_state.get("retriever"):
    st.divider()
    st.subheader("💬 Chat with Video (RAG Agent)")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if user_input := st.chat_input("Ask questions about the video context..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                
            
                template = f"""You are an intelligent video analyst for '{brand_name}'.
                
                TRANSCRIPT CONTEXT:
                {{context}}
                
                BRAND GUIDELINES:
                {brand_guidelines}
                
                USER QUESTION: {{question}}
                
                INSTRUCTIONS:
                1. FIRST, answer the user's question directly using ONLY facts from the transcript.
                2. If the transcript mentions specific objects/topics, confirm them.
                3. ONLY discuss brand alignment if the user explicitly asks about safety/risks.
                4. Be concise.
                """
                
                prompt = ChatPromptTemplate.from_template(template)
                llm = ChatOllama(model=OLLAMA_MODEL)
                
                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(user_input)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": response})