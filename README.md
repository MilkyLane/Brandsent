# 🛡️ BrandSent: Autonomous Video Intelligence Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **GenAI-Powered Brand Safety & Alignment System for Social Media Content**

BrandSent is an intelligent video analysis platform that combines **Speech-to-Text AI**, **Large Language Models**, and **Retrieval-Augmented Generation (RAG)** to automatically assess brand safety and alignment in social media videos. Built for marketing teams and content moderators to ensure brand integrity at scale.

---

## 🎯 Problem Statement

In today's fast-paced digital marketing landscape, brands face critical challenges:
- **Manual Content Review**: Reviewing influencer partnerships and UGC (User Generated Content) is time consuming
- **Brand Risk Management**: Ensuring content aligns with brand values and guidelines across thousands of videos
- **Scalability**: Human review doesn't scale with the volume of social media content

**BrandSent solves this** by automating video intelligence with AI-driven safety scoring, brand alignment analysis, and interactive RAG-based Q&A.

---

## ✨ Key Features

### 🎥 **Automated Video Processing Pipeline**
- **Multi-Platform Support**: Download and process videos from YouTube Shorts, TikTok, and other platforms
- **Audio Extraction**: Leverages `yt-dlp` for robust media handling
- **Speech Recognition**: Uses OpenAI's Whisper model for accurate transcription

### 🤖 **AI-Powered Brand Intelligence**
- **Dual Scoring System**:
  - **Safety Score (0-5)**: Detects toxic content, profanity, violence, and controversial topics
  - **Brand Alignment Score (0-100)**: Measures compatibility with custom brand values
- **Risk Flagging**: Automatically identifies and surfaces potential brand risks
- **Executive Summaries**: Generates concise analysis reports for decision-makers

### 💬 **RAG-Enabled Conversational Agent**
- **Vector Database Indexing**: Uses ChromaDB and Ollama embeddings for semantic search
- **Context-Aware Chat**: Ask natural language questions about video content
- **Brand-Guided Responses**: AI responses consider your specific brand guidelines

### 🎨 **Streamlit UI**
- **Clean, Professional Interface**: Designed for non-technical stakeholders
- **Real-Time Processing**: Live pipeline status and progress indicators
- **Chat History**: Persistent conversation memory within sessions

---

## 🏗️ Architecture

```
┌─────────────────┐
│  YouTube/TikTok │
│      Video      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   yt-dlp Audio  │
│   Extraction    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Whisper STT     │
│ Transcription   │
└────────┬────────┘
         │
         ├──────────────────┬─────────────────┐
         ▼                  ▼                 ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────┐
│ Ollama LLM      │  │  ChromaDB   │  │  LangChain  │
│ Safety Analysis │  │  Vector DB  │  │  RAG Chain  │
└─────────────────┘  └─────────────┘  └─────────────┘
         │                  │                 │
         └──────────────────┴─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Streamlit UI  │
                  │  Dashboard + Chat│
                  └─────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **LLM** | Ollama (Llama 3.2) | Local AI inference for safety analysis |
| **Speech-to-Text** | OpenAI Whisper | Audio transcription |
| **Embeddings** | Ollama Embeddings | Vector representations for RAG |
| **Vector DB** | ChromaDB | Semantic search and retrieval |
| **RAG Framework** | LangChain | Orchestration of retrieval-augmented generation |
| **Media Handling** | yt-dlp | Video/audio download and processing |

---

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **FFmpeg** (for audio processing)
- **Ollama** (for local LLM inference)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/BrandSent.git
cd BrandSent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg** (if not already installed)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

4. **Install and start Ollama**
```bash
# Download from https://ollama.ai
# Then pull the required model:
ollama pull llama3.2
```

5. **Run the application**
```bash
streamlit run brand_agent.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🚀 Usage Guide

### Step 1: Define Your Brand Identity
Enter your brand name and guidelines in the left panel:
```
Brand Name: EcoLife Sneakers
Guidelines: We are a sustainable, family-friendly brand...
```

### Step 2: Input Video Content
Paste a YouTube Short or TikTok URL and click **"Analyze & Index Video"**

### Step 3: Review Safety Metrics
- **Safety Score**: 0-5 scale (toxicity detection)
- **Alignment Score**: 0-100% (brand value match)
- **Risk Flags**: Automatically identified concerns
- **Executive Summary**: AI-generated analysis

### Step 4: Interactive Q&A
Use the RAG-powered chat to ask questions like:
- *"What products are mentioned in the video?"*
- *"Does the creator use any controversial language?"*
- *"Is this video suitable for our family-friendly brand?"*

---

## 📊 Example Output

```
┌─────────────────────────────────────────────────────┐
│ Safety Score: 4/5 (Safe)                           │
│ Brand Alignment: 87%                               │
│                                                    │
│ Executive Summary:                                 │
│ Content promotes active lifestyle and outdoor      │
│ activities, strongly aligned with brand values.    │
│ Minor concern: brief mention of competitor brand.  │
│                                                    │
│ Risk Flags:                                        │
│ 🚩  Uncontextualized reference                      │
└─────────────────────────────────────────────────────┘
```

---

## 🔮 Future Enhancements

- [ ] **Batch Processing**: Analyze multiple videos simultaneously
- [ ] **Advanced Moderation**: Computer vision for visual content analysis
- [ ] **Custom Model Fine-Tuning**: Train brand-specific classifiers
- [ ] **API Deployment**: REST API for programmatic access
- [ ] **Cloud Integration**: AWS/Azure deployment with scalable infrastructure
- [ ] **Multi-Language Support**: Transcription and analysis in 50+ languages
- [ ] **Export Reports**: PDF/CSV report generation for stakeholders

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Show Your Support

If this project helped you or you find it interesting, please give it a ⭐️ on GitHub!

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Models](https://ollama.ai/library)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)

---

<div align="center">
  <strong>Built with ❤️ for the future of AI-powered marketing technology</strong>
</div>
