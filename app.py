import re
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Maintenance Copilot",
    page_icon="🛠️",
    layout="wide"
)

# -----------------------------
# Colorful UI CSS
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #ec4899);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #f8fafc;
    margin-bottom: 30px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    color: white;
}

.answer-box {
    background: white;
    color: #111827;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
    line-height: 1.7;
    font-size: 16px;
}

.source-box {
    background: #f8fafc;
    color: #111827;
    border-left: 6px solid #06b6d4;
    padding: 16px;
    border-radius: 16px;
    margin-top: 12px;
}

.info-card {
    background: rgba(255,255,255,0.18);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 15px;
    color: white;
}

.stFileUploader {
    background: white;
    border-radius: 16px;
    padding: 15px;
}

.stTextInput input {
    border-radius: 20px;
    border: 2px solid #06b6d4;
}

.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 14px;
    border: none;
    font-weight: bold;
}

[data-testid="chatAvatarIcon-user"] {
    background-color: #2563eb;
}

[data-testid="chatAvatarIcon-assistant"] {
    background-color: #10b981;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class='title'>🛠️ Maintenance Copilot</div>
<div class='subtitle'>
AI-Powered Equipment Diagnostics using Maintenance Manuals and Image Evidence
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return re.sub(r"\s+", " ", text).strip()


def create_chunks(text, chunk_size=800):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


def generate_maintenance_answer(question, retrieved_text):
    q = question.lower()
    context = retrieved_text.lower()

    issue = "The issue may be related to equipment wear, abnormal operation, or a maintenance fault."

    actions = [
        "Stop the equipment safely before inspection.",
        "Check the affected component shown in the uploaded image.",
        "Follow the relevant maintenance procedure from the manual.",
        "Inspect for wear, overheating, looseness, cracks, leakage, or abnormal vibration.",
        "Repair or replace the damaged component if required.",
        "Record the maintenance activity and verify safe operation before restart."
    ]

    if "temperature" in q or "overheat" in q or "heat" in context:
        issue = "The equipment may be experiencing overheating or thermal stress."
        actions = [
            "Shut down the machine safely.",
            "Inspect the cooling fan and ventilation paths.",
            "Check for dust blockage or poor airflow.",
            "Verify lubrication condition.",
            "Allow the equipment to cool before restarting.",
            "Follow the overheating procedure from the manual."
        ]

    elif "vibration" in q or "bearing" in q or "vibration" in context:
        issue = "The equipment may have excessive vibration, bearing wear, or shaft alignment issues."
        actions = [
            "Inspect bearings for wear or damage.",
            "Check shaft alignment and mounting bolts.",
            "Verify lubrication condition.",
            "Replace worn bearings if vibration continues.",
            "Run the machine under observation after maintenance."
        ]

    elif "belt" in q or "belt" in context:
        issue = "The equipment may have belt wear, belt looseness, or pulley misalignment."
        actions = [
            "Inspect belt surface for cracks or wear.",
            "Check belt tension.",
            "Verify pulley alignment.",
            "Replace damaged belt if required.",
            "Ensure safety guards are fitted before restart."
        ]

    elif "leak" in q or "oil" in q or "leak" in context:
        issue = "The equipment may have oil leakage, seal failure, or lubrication-related fault."
        actions = [
            "Identify the leakage point.",
            "Inspect seals, joints, and lubrication lines.",
            "Clean spilled oil to prevent slip hazards.",
            "Replace damaged seals or loose fittings.",
            "Refill lubricant as per manual specification."
        ]

    elif "motor" in q or "motor" in context:
        issue = "The motor may have overheating, electrical, or mechanical load-related issues."
        actions = [
            "Disconnect power before inspection.",
            "Check motor temperature and wiring condition.",
            "Inspect load, coupling, and ventilation.",
            "Check for burning smell, noise, or unusual vibration.",
            "Contact electrical maintenance if the fault persists."
        ]

    action_text = ""
    for i, step in enumerate(actions, start=1):
        action_text += f"{i}. {step}<br>"

    return f"""
<b>Possible Issue:</b><br>
{issue}

<br><br>

<b>Recommended Maintenance Actions:</b><br>
{action_text}

<br>

<b>Evidence From Manual:</b><br>
{retrieved_text[:1200]}
"""


# -----------------------------
# Session State
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar Uploads
# -----------------------------
with st.sidebar:
    st.header("📤 Upload Inputs")

    pdf_file = st.file_uploader(
        "Upload Maintenance Manual PDF",
        type=["pdf"]
    )

    image_file = st.file_uploader(
        "Upload Related Equipment Image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("---")

    st.markdown("""
    <div class='info-card'>
    <b>Required Inputs</b><br><br>
    ✅ Maintenance Manual PDF<br>
    ✅ Related Equipment Image
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
    <b>Example Questions</b><br><br>
    • Why is the motor overheating?<br>
    • What maintenance action is needed?<br>
    • What should I do for bearing vibration?<br>
    • How can I fix oil leakage?<br>
    • What causes belt failure?
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Process PDF
# -----------------------------
if pdf_file:
    if (
        "processed_pdf" not in st.session_state
        or st.session_state.processed_pdf != pdf_file.name
    ):
        with st.spinner("Reading and indexing maintenance manual..."):
            text = extract_pdf_text(pdf_file)

            if not text:
                st.error("No readable text found in PDF.")
                st.stop()

            chunks = create_chunks(text)

            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(chunks)

            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.vectors = vectors
            st.session_state.processed_pdf = pdf_file.name
            st.session_state.messages = []

        st.sidebar.success("✅ PDF indexed successfully!")

# -----------------------------
# Display Image
# -----------------------------
if image_file:
    image = Image.open(image_file)
    st.sidebar.image(
        image,
        caption="Uploaded Equipment Image",
        use_container_width=True
    )

# -----------------------------
# Main Info Cards
# -----------------------------
if not pdf_file or not image_file:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='info-card'>
        <h3>📄 Manual Search</h3>
        Retrieves maintenance procedures from uploaded PDF manuals.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
        <h3>🖼️ Image Evidence</h3>
        Uses the uploaded equipment image as visual maintenance evidence.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-card'>
        <h3>🛠️ Diagnosis</h3>
        Suggests possible faults and maintenance actions.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Chat History
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f"<div class='answer-box'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.write(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input("Ask about equipment fault or maintenance action...")

if question:
    if st.session_state.vectors is None:
        st.warning("Please upload the maintenance manual PDF first.")

    elif image_file is None:
        st.warning("Please upload the related equipment image also.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        q_vector = st.session_state.vectorizer.transform([question])
        scores = cosine_similarity(q_vector, st.session_state.vectors).flatten()

        top_indexes = scores.argsort()[::-1][:3]

        retrieved_text = " ".join([
            st.session_state.chunks[i]
            for i in top_indexes
        ])

        answer = generate_maintenance_answer(question, retrieved_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(
                f"<div class='answer-box'>{answer}</div>",
                unsafe_allow_html=True
            )

            st.image(
                image_file,
                caption="Related Equipment Image Evidence",
                use_container_width=True
            )

            with st.expander("📖 Retrieved PDF Source Chunks"):
                for i in top_indexes:
                    st.markdown(
                        f"""
                        <div class='source-box'>
                        <b>Similarity Score:</b> {round(scores[i], 4)}<br><br>
                        {st.session_state.chunks[i]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
