# 🏥 Intelligent Clinical Assistant (Indian Health System)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-clinical-assistant-7.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-based, multimodal Clinical Decision Support System (CDSS) built on a Retrieval-Augmented Generation (RAG) architecture. It aligns clinical decision-making with Indian statutory health protocols, including **MoHFW**, **ICMR Standard Treatment Workflows (STWs)**, and **NVBDCP** guidelines.

🔗 **Live Production Demo:** [https://ai-clinical-assistant-7.streamlit.app](https://ai-clinical-assistant-7.streamlit.app)

---

## 🌟 Key Features

- **Guideline-Grounded RAG Pipeline:** Retrieves verified clinical protocols with vector similarity scores and authority citations (MoHFW, ICMR, NVBDCP) to eliminate LLM hallucinations.
- **Multimodal Document & Lab Report Analysis:** Parses patient records, Complete Blood Count (CBC) reports, and Dengue serology from uploaded PDFs or image files.
- **Multilingual Support & Persona Switching:** Seamlessly handles consultations in **English, Hindi, and regional languages** with specialized personas for both *Doctor / Medical Officer* and *Patient / Caregiver*.
- **Integrated Voice Accessibility:** Text-to-Speech (gTTS) audio playback for clinical summaries in selected regional languages.
- **One-Click Clinical Summary Export:** Generates standardized, single-page downloadable clinical consultation reports in PDF format using ReportLab.

---

## 🏗️ System Architecture

```text
[ Patient Query / Lab Report PDF ]
                │
                ▼
     [ Streamlit Interface ]
         │               │
  (Document Parsing)  (Vector Search)
         │               │
         ▼               ▼
     [ PyPDF ]     [ FAISS Vector Store ]
                         │
                 (Top-K Chunks + Metadata)
                         │
                         ▼
        [ LangChain RAG Orchestration ]
                         │
                         ▼
             [ Groq LLM Inference ]
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
 [ Clinical Triage & Advice ]     [ Multimodal Outputs ]
 - Statutory Citations            - gTTS Audio Player
 - Differential Diagnosis         - ReportLab Clinical PDF
 - Drug Regimen & Red Flags
```

---

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **LLM Inference:** Groq Cloud API
- **RAG & Orchestration:** LangChain, FAISS (Vector Store)
- **Embeddings:** Hugging Face Sentence Transformers
- **Document Ingestion:** PyPDF, Pillow
- **Audio & Export:** gTTS, ReportLab

---

## 🚀 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shivampal6123/ai-clinical-assistant.git
   cd ai-clinical-assistant
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets:**
   Create a file `.streamlit/secrets.toml` and add your Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📋 Evaluation & Validation Scenarios

The system has been evaluated and validated across real clinical test scenarios:
1. **Vector-Grounded NVBDCP Malaria Guidelines:** Verified drug regimens for *P. vivax* (Chloroquine + Primaquine) vs *P. falciparum* (ACT) with strict statutory citations.
2. **Regional Language Clinical Counseling:** Verified Hindi prompt responses for Dengue fever triage with gTTS audio synthesis.
3. **Lab Report Parsing & Triage:** Parsed acute CBC reports showing severe thrombocytopenia (Platelets 38,000/µL) and hemoconcentration (PCV 49%), accurately triggering Dengue Shock Syndrome (DSS) fluid resuscitation protocols and PDF summary generation.

---

## ⚖️ Disclaimer

*This tool is a clinical decision support prototype developed for academic and operational decision assistance. It is intended to assist medical professionals and should not replace primary clinical judgment.*






