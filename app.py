import base64
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
dotenv_path = ".env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# API Key Validation
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("The GROQ_API_KEY environment variable is not set. Please set it before running the script.")
    st.stop()

# Set up the background and styles
def set_background():
    desktop_image_path = "image.png"
    mobile_image_path = "mobile_bg.jpg"

    desktop_image = ""
    mobile_image = ""

    if os.path.exists(desktop_image_path):
        try:
            with open(desktop_image_path, "rb") as desktop_file:
                desktop_image = base64.b64encode(desktop_file.read()).decode()
        except Exception as e:
            st.error(f"Error reading desktop background image: {e}")
            desktop_image = ""
    else:
        st.warning(f"Desktop background image not found at {desktop_image_path}. Using default background.")

    if os.path.exists(mobile_image_path):
        try:
            with open(mobile_image_path, "rb") as mobile_file:
                mobile_image = base64.b64encode(mobile_file.read()).decode()
        except Exception as e:
            st.error(f"Error reading mobile background image: {e}")
            mobile_image = ""
    else:
        st.warning(f"Mobile background image not found at {mobile_image_path}. Using desktop background for mobile.")
        mobile_image = desktop_image

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{desktop_image}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #000;
        }}
        @media only screen and (max-width: 768px) {{
            .stApp {{
                background-image: url("data:image/jpeg;base64,{mobile_image}");
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
                background-attachment: scroll;
                background-color: #000;
            }}
        }}
        div[data-testid="stFileUploader"] {{
            background-color: transparent !important;
            color: white !important;
            border: none !important;
            border-radius: 10px;
            padding: 10px;
        }}
        div[data-testid="stUploadedFile"] {{
            background-color: transparent !important;
            color: white !important;
            font-size: 16px;
            padding: 5px;
            margin-top: 10px;
        }}
        div.stButton > button:first-child {{
            background-color: red;
            color: white;
            border-radius: 5px;
            border: none;
            font-size: 16px;
            padding: 10px 20px;
        }}
        div.stButton > button:first-child:hover {{
            background-color: darkred;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Display response with a solid background
def display_answer_with_background(answer):
    st.markdown(
        f"""
        <div style="
            background-color: #0d3b0d;  /* Dark green solid background */
            color: white;              /* White text color */
            padding: 15px;             /* Padding around the content */
            border-radius: 10px;       /* Rounded corners */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow for depth */
            font-size: 16px;           /* Adjust font size */
            line-height: 1.5;          /* Better line spacing */
        ">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

# Extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF {pdf_file_path}: {e}")
        return ""

# Load the sentence transformer model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Streamlit app UI
st.markdown("<div style='text-align: center; color: white; font-size: 36px;'>📄 Explain Mate</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: white; font-size: 18px;'>✨ Your friendly PDF assistant! Upload a document and let me handle the questions. 🎉</div>", unsafe_allow_html=True)

pdf_file = st.file_uploader("", type="pdf")
st.markdown(
    """
    <style>
    label {
        color: white !important;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if pdf_file is None:
        st.error("Please upload a PDF file.")
    elif not question.strip():
        st.error("Please enter a valid question.")
    else:
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        pdf_content = extract_text_from_pdf(temp_pdf_path)
        os.remove(temp_pdf_path)

        if not pdf_content:
            st.error("Could not extract content from the PDF.")
        else:
            chunk_size = 200
            words = pdf_content.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            if not chunks:
                st.error("Could not split the PDF content into meaningful chunks.")
            else:
                embeddings = model.encode(chunks, convert_to_numpy=True) 
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                query_embedding = model.encode([question], convert_to_numpy=True)

                k = min(5, len(chunks))
                distances, indices = index.search(query_embedding, k=k)
                context_chunks = [chunks[i] for i in indices[0]]
                context = " ".join(context_chunks)

                if not context:
                    st.error("Could not find relevant context in the PDF for your question.")
                else:
                    try:
                        client = Groq(api_key=api_key)
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that answers questions based only on the provided context.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {question}",
                                },
                            ],
                            model="llama3-8b-8192",
                            temperature=0.2,
                            max_tokens=300,
                        )
                        response = chat_completion.choices[0].message.content
                        display_answer_with_background(response)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")

st.markdown("<div style='text-align: center; color: white;'>Made with ❤️ by <b>Aswini</b></div>", unsafe_allow_html=True)
