import base64
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
import streamlit as st


# Function to add background with an infographic
def set_background():
    with open("image.png", "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover; /* Ensures the image covers the entire screen */
            background-position: center; /* Centers the image */
            background-repeat: no-repeat; /* Prevents repeating */
            height: 100%;
            min-height: 100vh; /* Ensures it covers the viewport height */
        }}

        /* Responsive design for smaller screens */
        @media (max-width: 768px) {{
            .stApp {{
                background-size: contain; /* Adjust size to fit smaller screens */
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


# Call the background function
set_background()

# Function to extract text from a PDF file object
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


# Load the sentence transformer model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Initialize Groq client with API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("The GROQ_API_KEY environment variable is not set. Please set it before running the script.")
    st.stop()

client = Groq(api_key=api_key)

# Streamlit app
st.markdown('<div class="center-content"><h1>📄 Explain Mate</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="center-content"><h4>✨ Your friendly PDF assistant! Upload a document and let me handle the questions. 🎉</h4></div>', unsafe_allow_html=True)

# File upload section
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

            if not chunks:
                st.error("Could not split the PDF content into meaningful chunks.")
            else:
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
                    st.error("Could not find relevant context in the PDF for your question.")
                else:
                    # Use the context and question to get a response from Groq
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that answers questions based on provided context from a PDF.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on the following context, answer the question:\\n\\nContext: {context}\\n\\nQuestion: {question}",
                                },
                            ],
                            model="llama3-8b-8192",
                        )
                        response = chat_completion.choices[0].message.content
                        st.success(response)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ by <b>Aswini</b></div>",
    unsafe_allow_html=True
)
