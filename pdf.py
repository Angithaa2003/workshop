# Install required packages if you haven't already
# pip install gradio llama-index langchain openai faiss-cpu tiktoken PyPDF2 langchain-community langchainhub

import gradio as gr
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex

# Global index and query engine
index = None
query_engine = None

# Step 1: Load PDF + Create Index
def load_pdf(file):
    global index, query_engine

    if file is None:
        return "❌ Please upload a PDF."

    # Read PDF and create documents
    reader = PDFReader()
    documents = reader.load_data(file=file.name)

    # Build vector index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return "✅ PDF uploaded and indexed successfully! Now ask your questions below."

# Step 2: Chat Function
def chat_with_pdf(question):
    global query_engine

    if query_engine is None:
        return "⚠ Please upload a PDF first."

    response = query_engine.query(question)
    return str(response)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📘 PDF Chatbot — Upload your PDF and Ask Anything!")

    pdf_file = gr.File(label="Upload your PDF", file_types=[".pdf"])
    upload_btn = gr.Button("Process PDF")

    upload_output = gr.Textbox(label="Status")

    upload_btn.click(load_pdf, inputs=pdf_file, outputs=upload_output)

    question = gr.Textbox(label="Ask something from the PDF")
    answer = gr.Textbox(label="Answer")

    question.submit(chat_with_pdf, inputs=question, outputs=answer)

# Launch the app locally
demo.launch()
