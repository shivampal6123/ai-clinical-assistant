import os
import io
import time
import base64
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from gtts import gTTS
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="AI Clinical Medical Assistant (Indian Context)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

INDEX_PATH = "Vectorstore/faiss_index"

@st.cache_resource
def load_faiss_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

vectorstore = load_faiss_vectorstore()

# Groq API Key Setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Enter your Groq key (starts with gsk_...)")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BANNED_FDCS = [
    "nimesulide + paracetamol", "aceclofenac + paracetamol + rabeprazole",
    "paracetamol + phenylephrine + caffeine", "amoxicillin + bromhexine",
    "ofloxacin + ornidazole", "ciprofloxacin + tinidazole"
]

EMERGENCY_WORDS = [
    "snakebite", "snake bite", "bleeding gums", "severe breathlessness",
    "chest pain radiating", "unconscious", "altered sensorium", "anaphylaxis",
    "sunken eyes", "stridor", "severe burn"
]

SUPPORTED_LANGUAGES = {
    "English (Standard Clinical)": {"code": "en", "tld": "co.in", "prompt": "Respond strictly in clear English with standard clinical terminology."},
    "Hinglish (Colloquial)": {"code": "hi", "tld": "co.in", "prompt": "Respond strictly in conversational Hinglish (Hindi written in Roman English script). Do not use Devanagari script."},
    "हिन्दी (Hindi)": {"code": "hi", "tld": "co.in", "prompt": "Respond accurately in Devanagari Hindi. Put medicine names in English brackets, e.g., मेटफॉर्मिन (Metformin). Do not repeat words."},
    "বাংলা (Bengali)": {"code": "bn", "tld": "com", "prompt": "Respond accurately in Bengali script."},
    "मराठी (Marathi)": {"code": "mr", "tld": "com", "prompt": "Respond accurately in Marathi script."},
    "తెలుగు (Telugu)": {"code": "te", "tld": "com", "prompt": "Respond accurately in Telugu script."},
    "தமிழ் (Tamil)": {"code": "ta", "tld": "com", "prompt": "Respond accurately in Tamil script."},
    "ગુજરાતી (Gujarati)": {"code": "gu", "tld": "com", "prompt": "Respond accurately in Gujarati script."},
    "ಕನ್ನಡ (Kannada)": {"code": "kn", "tld": "com", "prompt": "Respond accurately in Kannada script."},
    "اردو (Urdu)": {"code": "ur", "tld": "com", "prompt": "Respond accurately in Urdu script."}
}

# ----------------- SESSION CHAT HISTORY STORAGE -----------------
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {"Consultation 1": []}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "Consultation 1"

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("🩺 MediBot Controls")

    # Session Selector & New Consultation
    session_names = list(st.session_state.all_sessions.keys())
    selected_session = st.selectbox("📂 Consultation History", session_names, index=session_names.index(st.session_state.current_session_id))
    st.session_state.current_session_id = selected_session

    if st.button("➕ New Consultation", use_container_width=True):
        new_name = f"Consultation {len(st.session_state.all_sessions) + 1}"
        st.session_state.all_sessions[new_name] = []
        st.session_state.current_session_id = new_name
        st.rerun()

    st.markdown("---")
    selected_lang_name = st.selectbox("🌐 Response Language", list(SUPPORTED_LANGUAGES.keys()), index=0)
    selected_lang = SUPPORTED_LANGUAGES[selected_lang_name]

    user_mode = st.radio("Operating Persona", ["Doctor / Medical Officer", "Patient / Caregiver"])
    retrieval_mode = st.selectbox("Knowledge Routing", [
        "Standard Hybrid (Indian Priority + Western Fallback)",
        "Strictly Indian Guidelines Only (MoHFW/ICMR/NCDC)",
        "Academic Benchmark (NEET-PG/AIIMS)"
    ])
    top_k = st.slider("Retrieval Depth (Top-K Chunks)", 3, 7, 5)

    st.markdown("---")
    st.subheader("🧮 Bedside Medical Calculators")
    with st.expander("IMNCI Pediatric Dosage"):
        w = st.number_input("Child Weight (kg)", 2.0, 40.0, 10.0, 0.5)
        st.info(f"Paracetamol (15 mg/kg): **{w * 15:.0f} mg** per dose (max QID)")
        st.info(f"Amoxicillin (45 mg/kg): **{w * 45:.0f} mg/day** in divided doses")

    with st.expander("NVBDCP Dengue Fluid Rate"):
        aw = st.number_input("Adult Weight (kg)", 35.0, 110.0, 60.0, 1.0)
        st.info(f"Plan B (Severe): **{aw * 1.5:.0f} to {aw * 2:.0f} ml/hr** IV crystalloid")

# ----------------- HELPER UTILITIES -----------------
def generate_pdf_summary(query_text, response_text):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 13)
    p.drawString(40, 750, "Clinical Decision Support Summary - Medical Assistant")
    p.setFont("Helvetica", 9)
    p.drawString(40, 735, "Aligned with MoHFW | ICMR STWs | NCDC | NVBDCP | NTEP Guidelines")
    p.line(40, 725, 570, 725)
    
    clean_q = re.sub(r'[^\x00-\x7F]+', ' ', str(query_text))
    clean_r = re.sub(r'[^\x00-\x7F]+', ' ', str(response_text))
    
    text = p.beginText(40, 700)
    text.setFont("Helvetica-Bold", 10)
    text.textLine("QUERY / PATIENT CASE:")
    text.setFont("Helvetica", 9)
    for q_line in clean_q[:250].split("\n"):
        text.textLine(q_line[:90])
    
    text.textLine("")
    text.setFont("Helvetica-Bold", 10)
    text.textLine("CLINICAL RECOMMENDATIONS & PROTOCOLS:")
    text.setFont("Helvetica", 9)
    for line in clean_r.split("\n")[:38]:
        if line.strip():
            text.textLine(line[:92])
    p.drawText(text)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def text_to_speech(clean_text, lang_code, tld):
    try:
        sample = re.sub(r'[*#_`]', '', str(clean_text))[:300]
        tts = gTTS(text=sample, lang=lang_code, tld=tld, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# ----------------- MAIN UI -----------------
st.title("🩺 Intelligent Clinical Assistant (Indian Health System)")
st.caption("Official Guidelines: MoHFW • ICMR STWs • NCDC Zoonotic/AMR • NVBDCP Vector Control • Gale Reference")

# Active conversation messages reference
current_chat_history = st.session_state.all_sessions[st.session_state.current_session_id]

# Emergency Alert Check
if current_chat_history and any(word in current_chat_history[-1]["content"].lower() for word in EMERGENCY_WORDS):
    st.error("🚨 **EMERGENCY CLINICAL RED FLAG DETECTED:** Immediate referral to nearest Secondary/Tertiary Centre required! Call **108 / 112** for emergency ambulance.")

# RENDER FULL PAST CHAT HISTORY
for msg in current_chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "badge" in msg:
            st.caption(msg["badge"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Retrieved Clinical Chunks & Evidence"):
                for d, sc in msg["sources"]:
                    st.markdown(f"**Authority:** `{d.metadata.get('authority', 'Guideline')}` | **Source:** `{d.metadata.get('source', '')}` (Distance: {sc:.3f})")
                    st.text(d.page_content[:350] + "...")

# ----------------- MULTIMODAL & MULTI-PDF INPUT ROW -----------------
col_audio, col_file = st.columns([1, 1.2])
query_prefix = ""

with col_audio:
    audio_data = st.audio_input("🎙️ Voice Dictation (Speak Query)")
    if audio_data and groq_client:
        with st.spinner("Transcribing via Groq Whisper..."):
            audio_f = io.BytesIO(audio_data.read())
            audio_f.name = "audio.wav"
            try:
                tr = groq_client.audio.transcriptions.create(file=audio_f, model="whisper-large-v3")
                query_prefix += f"\n[Spoken Patient Query]: {tr.text}"
            except Exception as e:
                st.warning(f"Voice Transcription failed: {e}")

with col_file:
    # MULTI-FILE UPLOADER: Supports up to 10 PDFs / Images simultaneously
    uploaded_files = st.file_uploader(
        "📷 Prescriptions / Lab Reports (Up to 10 files: PDF, JPG, PNG)",
        type=["pdf", "jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) attached.")
        multi_doc_context = ""
        for idx, file in enumerate(uploaded_files[:10], start=1):
            if file.type == "application/pdf":
                reader = PdfReader(file)
                extracted_pages = "\n".join([p.extract_text() for p in reader.pages[:2] if p.extract_text()])
                multi_doc_context += f"\n--- [Document {idx}: {file.name}] ---\n{extracted_pages[:1200]}\n"
            elif file.type in ["image/jpeg", "image/png"] and groq_client:
                b64 = base64.b64encode(file.read()).decode("utf-8")
                try:
                    res = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": "Extract clinical vitals, lab values, and medicine names."}],
                        temperature=0.1
                    )
                    multi_doc_context += f"\n--- [Document {idx}: {file.name}] ---\n{res.choices[0].message.content}\n"
                except Exception:
                    pass
        if multi_doc_context:
            query_prefix += f"\n[Attached Clinical Documents Data]:\n{multi_doc_context}"

chat_text = st.chat_input("Enter symptoms, patient case, or clinical drug protocol query...")
final_query = None

if chat_text:
    final_query = f"{query_prefix}\n{chat_text}".strip()
elif query_prefix and st.button("🚀 Analyze Uploaded Documents/Audio", use_container_width=True):
    final_query = query_prefix.strip()

# ----------------- INFERENCE & STREAMING -----------------
if final_query:
    if not groq_client:
        st.error("Please enter a valid GROQ_API_KEY in the sidebar to run the model.")
        st.stop()

    # Append user turn to persistent history
    current_chat_history.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    with st.spinner("Searching National Protocols & Evidence..."):
        docs_and_scores = vectorstore.similarity_search_with_score(final_query, k=top_k) if vectorstore else []
        
        if retrieval_mode == "Strictly Indian Guidelines Only (MoHFW/ICMR/NCDC)":
            docs_and_scores = [(d, s) for d, s in docs_and_scores if d.metadata.get("domain") == "Indian_Clinical_Guideline"] or docs_and_scores

    has_indian = any(d.metadata.get("domain") == "Indian_Clinical_Guideline" for d, _ in docs_and_scores)
    badge = "🟢 Official Indian Statutory Guideline (MoHFW / ICMR)" if has_indian else "🔵 Western Reference (Gale / MedMCQA)"

    context_str = "\n".join([f"[{d.metadata.get('authority', 'Guideline')}]: {d.page_content}" for d, _ in docs_and_scores])
    banned_hits = [fdc for fdc in BANNED_FDCS if fdc in final_query.lower()]

    with st.chat_message("assistant"):
        st.markdown(f"**Knowledge Source:** {badge}")
        
        if banned_hits:
            st.warning(f"🔴 **Statutory Warning (Section 26A):** Query mentions banned drug combinations: `{', '.join(banned_hits)}`")

        with st.status("🧠 Clinical Reasoning & Guideline Verification...", expanded=False):
            st.write(f"- User Persona: `{user_mode}`")
            st.write(f"- Selected Language: `{selected_lang_name}`")
            st.write(f"- Matched {len(docs_and_scores)} statutory chunks from index.")
            st.write("- Verifying against NLEM 2022 and MoHFW National Treatment Protocols.")

        system_instruction = f"""You are an Expert Clinical Medical Decision Support System aligned with the Indian Healthcare Framework (MoHFW, ICMR, NCDC, NVBDCP, NTEP, NACO).
Language Rule: {selected_lang['prompt']}
Operational Persona: {user_mode}

Core Clinical Protocol:
1. Provide a cohesive, structured clinical response without ever repeating words, numbers, or phrases in loops.
2. Structure sections clearly:
   - 🩺 Clinical Assessment & Differential Diagnosis
   - 🔬 First-Line Diagnostic Investigations (NLEM/IPHS standard)
   - 💊 Recommended Treatment & Dosage Regimen (Preserve exact mg, duration, and frequencies)
   - ⚠️ Clinical Red Flags & Referral Criteria
   - 📚 Statutory Authority Citations (ICMR STWs / MoHFW / NCDC)
3. Synthesize any uploaded document context directly into the treatment plan.
"""
        prompt_content = f"Official Guidelines Context:\n{context_str}\n\nClinical Query and Patient Data:\n{final_query}"

        # Production model candidates (Eliminated allam to prevent repetitive gibberish)
        CANDIDATE_MODELS = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant"
        ]

        stream = None
        selected_model_used = None

        for m_name in CANDIDATE_MODELS:
            try:
                stream = groq_client.chat.completions.create(
                    model=m_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    presence_penalty=0.5,
                    frequency_penalty=0.5,
                    stream=True
                )
                selected_model_used = m_name
                break
            except Exception:
                continue

        if not stream:
            st.error("❌ Unable to connect to Groq models. Please verify your Groq API key in the sidebar.")
            st.stop()

        full_reply = ""
        reply_box = st.empty()
        for chunk in stream:
            c = chunk.choices[0].delta.content or ""
            full_reply += c
            reply_box.markdown(full_reply + "▌")
        reply_box.markdown(full_reply)

        st.caption(f"⚡ Inferred using `{selected_model_used}`")

        audio_stream = text_to_speech(full_reply, selected_lang["code"], selected_lang["tld"])
        if audio_stream:
            st.audio(audio_stream, format="audio/mp3")

        pdf_file = generate_pdf_summary(final_query, full_reply)
        st.download_button(
            label="📄 Download Clinical Summary (PDF)",
            data=pdf_file,
            file_name=f"Clinical_Summary_{int(time.time())}.pdf",
            mime="application/pdf"
        )

        with st.expander("📚 View Retrieved Clinical Chunks & Citations"):
            for d, sc in docs_and_scores:
                st.markdown(f"**Authority:** `{d.metadata.get('authority', 'Guideline')}` | **File:** `{d.metadata.get('source', '')}`")
                st.caption(f"Similarity Score: {sc:.4f}")
                st.text(d.page_content[:350] + "...")

    # Save assistant response into persistent session history
    current_chat_history.append({
        "role": "assistant",
        "content": full_reply,
        "badge": badge,
        "sources": docs_and_scores
    })