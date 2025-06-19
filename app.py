import base64
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import streamlit as st
import torch


# Function to add responsive background and custom styling
def set_background():
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
        /* Center content styling */
        .center-content {{
            text-align: center;
        }}
        /* File uploader styling */
        .stFileUploader {{
            border: none;
            background-color: #fff;
            color: black !important;
            border-radius: 10px;
            padding: 10px;
        }}
        /* Button styling */
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


# Function to summarize response
def summarize_response(response, max_length=500):
    return response[:max_length] + "..." if len(response) > max_length else response


# Initialize the SentenceTransformer model
def initialize_model(model_name):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(model_name)
        if hasattr(model, "to_empty"):
            model = model.to_empty()
        model.to(device)
        # Test model loading
        _ = model.encode(["Test input"])
        return model, device
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        st.stop()


# Load the model
model_name = "all-MiniLM-L6-v2"
model, device = initialize_model(model_name)


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
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        pdf_content = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")
        
        if not pdf_content:
            st.error("Could not extract text from the PDF.")
        else:
            words = pdf_content.split()
            chunks = [" ".join(words[i:i + 100]) for i in range(0, len(words), 100)]
            
            if not chunks:
                st.error("No meaningful chunks found in the PDF content.")
            else:
                try:
                    # Create embeddings for chunks
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

                    if not context:
                        st.error("No relevant context found for your question.")
                    else:
                        # Generate response from the model
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Provide clear and concise answers based on the provided context."},
                                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                            ],
                            model="llama3-8b-8192",
                        )
                        response = chat_completion.choices[0].message.content
                        st.success(summarize_response(response.strip()))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ by <b>Aswini</b></div>",
    unsafe_allow_html=True
)
