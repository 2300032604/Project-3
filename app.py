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

st.markdown("""
<style>
.stApp {
    background-color: #212121;
    color: white;
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    margin-top: 20px;
}
.subtitle {
    text-align: center;
    color: #b4b4b4;
    font-size: 17px;
    margin-bottom: 30px;
}
.answer-box {
    background: #303030;
    padding: 18px;
    border-radius: 18px;
    margin-top: 10px;
    line-height: 1.7;
}
.source-box {
    background: #171717;
    border-left: 4px solid #10a37f;
    padding: 14px;
    border-radius: 10px;
    margin-top: 10px;
}
.stFileUploader {
    background: #2f2f2f;
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🛠️ Maintenance Copilot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a maintenance manual PDF and related equipment image</div>",
    unsafe_allow_html=True
)

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
    action = [
        "Stop the equipment safely before inspection.",
        "Check the affected component shown in the uploaded image.",
        "Follow the relevant maintenance procedure from the manual.",
        "Inspect for wear, overheating, looseness, cracks, leakage, or abnormal vibration.",
        "Replace or repair the damaged component if required.",
        "Record the maintenance action and verify safe operation before restart."
    ]

    if "temperature" in q or "overheat" in q or "heat" in context:
        issue = "The equipment may be experiencing overheating or thermal stress."
        action = [
            "Shut down the machine safely.",
            "Inspect cooling fan, ventilation, and airflow paths.",
            "Check for dust blockage or insufficient lubrication.",
            "Allow the equipment to cool before restarting.",
            "Follow the overheating procedure mentioned in the manual."
        ]

    elif "vibration" in q or "bearing" in q or "vibration" in context:
        issue = "The equipment may have excessive vibration, bearing wear, or alignment problems."
        action = [
            "Inspect bearings for wear or damage.",
            "Check shaft alignment and mounting bolts.",
            "Verify lubrication condition.",
            "Replace worn bearings if vibration continues.",
            "Run the equipment under observation after maintenance."
        ]

    elif "belt" in q or "belt" in context:
        issue = "The equipment may have belt wear, belt looseness, or belt misalignment."
        action = [
            "Inspect the belt for cracks, wear, or looseness.",
            "Check belt tension and pulley alignment.",
            "Replace the belt if damaged.",
            "Ensure guards are fitted before restarting."
        ]

    elif "leak" in q or "oil" in q or "leak" in context:
        issue = "The equipment may have oil leakage, seal failure, or lubrication-related fault."
        action = [
            "Identify the leakage point.",
            "Check seals, joints, and lubrication lines.",
            "Clean spilled oil and prevent slip hazards.",
            "Replace damaged seals or fittings.",
            "Refill lubricant as per manual specification."
        ]

    elif "motor" in q or "motor" in context:
        issue = "The motor may have electrical, overheating, or mechanical load-related issues."
        action = [
            "Disconnect power before inspection.",
            "Check motor temperature and wiring condition.",
            "Inspect load, coupling, and ventilation.",
            "Check for burning smell or unusual noise.",
            "Contact electrical maintenance if fault persists."
        ]

    actions_text = ""
    for i, step in enumerate(action, start=1):
        actions_text += f"{i}. {step}\n"

    return f"""
Based on the uploaded maintenance manual and related image evidence:

Possible Issue:
{issue}

Recommended Maintenance Actions:
{actions_text}

PDF Evidence Used:
{retrieved_text[:1200]}
"""


if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "messages" not in st.session_state:
    st.session_state.messages = []


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
    st.write("### Inputs Required")
    st.write("✅ PDF Manual")
    st.write("✅ Related Image")

    st.markdown("---")
    st.write("### Example Questions")
    st.write("What is the possible fault?")
    st.write("What maintenance action is needed?")
    st.write("Why is the motor overheating?")
    st.write("What should I do for bearing vibration?")


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

        st.sidebar.success("PDF indexed successfully!")


if image_file:
    image = Image.open(image_file)
    st.sidebar.image(image, caption="Uploaded Equipment Image", use_container_width=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


question = st.chat_input("Ask about the equipment issue or maintenance action...")

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
