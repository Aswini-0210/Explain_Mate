import base64
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
import streamlit as st
import torch

# Function to add responsive background
def set_background():
    # Assuming desktop_image.png and mobile_image.png exist in the same directory
    with open("image.png", "rb") as desktop_file:
        desktop_image = base64.b64encode(desktop_file.read()).decode()
    with open("mobile_bg.jpg", "rb") as mobile_file:
        mobile_image = base64.b64encode(mobile_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{desktop_image}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        @media only screen and (max-width: 768px) {{
            .stApp {{
                background-image: url("data:image/jpeg;base64,{mobile_image}");
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
                background-attachment: scroll;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize the SentenceTransformer model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check for meta tensors
for name, param in model.named_parameters():
    if param.device.type == "meta":
        raise RuntimeError(f"Parameter {name} is on meta device. Initialization failed.")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF {pdf_file_path}: {e}")
        return ""

# Call the background function
set_background()

# Streamlit app
st.markdown('<div class="center-content"><h1>📄 Explain Mate</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="center-content"><h4>✨ Your friendly PDF assistant! Upload a document and let me handle the questions. 🎉</h4></div>', unsafe_allow_html=True)

# File upload and question handling
pdf_file = st.file_uploader("", type="pdf")
question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if pdf_file is None:
        st.error("Please upload a PDF file.")
    elif not question:
        st.error("Please enter a question.")
    else:
        # Save uploaded PDF as a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())

        # Extract text from the uploaded PDF
        pdf_content = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")  # Remove temporary file

        if not pdf_content:
            st.error("Could not extract text from the PDF.")
        else:
            # Process the PDF content into chunks and create embeddings
            chunk_size = 100
            words = pdf_content.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            embeddings = model.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Generate embedding for the user query
            query_embedding = model.encode([question])

            # Search for relevant chunks
            k = min(5, len(chunks))
            distances, indices = index.search(query_embedding, k=k)
            context_chunks = [chunks[i] for i in indices[0]]
            context = " ".join(context_chunks)

            st.success(f"Context for your question: {context}")

