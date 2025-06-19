import base64
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
import streamlit as st

# Function to add background with an infographic
def set_background(image_path="image.png"):
    try:
        with open(image_path, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                min-height: 100vh;
            }}
            @media (max-width: 768px) {{
                .stApp {{
                    background-size: contain;
                }}
            }}
            .center-content {{
                text-align: center;
            }}
            .stFileUploader {{
                border: none;
                background-color: #fff;
                color: black !important;
                border-radius: 10px;
                padding: 10px;
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
    except FileNotFoundError:
        st.error("Background image file not found. Please ensure 'image.png' is in the script directory.")

# Call the background function
set_background()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Initialize Sentence Transformer model
try:
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device="cpu")  # Use CPU explicitly
except Exception as e:
    st.error(f"Error loading Sentence Transformer model: {e}")
    st.stop()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("The GROQ_API_KEY environment variable is not set. Please set it before running the script.")
    st.stop()

client = Groq(api_key=api_key)

# Streamlit App Layout
st.markdown('<div class="center-content"><h1>📄 Explain Mate</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="center-content"><h4>✨ Your friendly PDF assistant! Upload a document and let me handle the questions. 🎉</h4></div>', unsafe_allow_html=True)

# File upload and question input
pdf_file = st.file_uploader("", type="pdf")
question = st.text_input("Ask your question")

# Handle user interaction
if st.button("Get Answer"):
    if pdf_file is None:
        st.error("Please upload a PDF file.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        pdf_content = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")  # Clean up

        if not pdf_content:
            st.error("No content found in the PDF.")
        else:
            try:
                # Chunking and embedding
                chunk_size = 100
                words = pdf_content.split()
                chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
                embeddings = model.encode(chunks)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                # Query embedding and search
                query_embedding = model.encode([question])
                k = min(5, len(chunks))
                distances, indices = index.search(query_embedding, k=k)
                context_chunks = [chunks[i] for i in indices[0]]
                context = " ".join(context_chunks)

                # Groq API for answer generation
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from a PDF."},
                        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                    ],
                    model="llama3-8b-8192",
                )
                response = chat_completion.choices[0].message.content
                st.success(response)
            except Exception as e:
                st.error(f"Error processing your request: {e}")

# Footer
st.markdown("<div style='text-align: center;'>Made with ❤️ by <b>Aswini</b></div>", unsafe_allow_html=True)
