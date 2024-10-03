import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import io

# Disable WANDB
os.environ['WANDB_DISABLED'] = "true"

# Constants
MODEL_PATH = "/home/lab/halyn/gemma/halyn/paper/models/gemma-2-9b-it"

app = FastAPI()

# Global variables to store the knowledge base and QA chain
knowledge_base = None
qa_chain = None

def load_pdf(pdf_file):
    """Load and extract text from a PDF file object."""
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def split_text(text):
    """Split the extracted text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_knowledge_base(chunks):
    """Create a FAISS knowledge base from text chunks."""
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

def load_model(model_path):
    """Load the HuggingFace model and tokenizer, and create a text-generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150, temperature=0.1)

@app.on_event("startup")
async def startup_event():
    global qa_chain
    load_dotenv()
    
    # Load the language model
    try:
        pipe = load_model(MODEL_PATH)
        llm = HuggingFacePipeline(pipeline=pipe)
        qa_chain = load_qa_chain(llm, chain_type="stuff")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load the language model")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global knowledge_base
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        text = load_pdf(pdf_file)
        chunks = split_text(text)
        knowledge_base = create_knowledge_base(chunks)
        return {"message": "PDF uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    global knowledge_base, qa_chain
    if not knowledge_base:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet")
    if not qa_chain:
        raise HTTPException(status_code=500, detail="QA chain is not initialized")
    
    try:
        docs = knowledge_base.similarity_search(question.text)
        response = qa_chain.run(input_documents=docs, question=question.text)

        if "Helpful Answer:" in response:
            response = response.split("Helpful Answer:")[1].strip()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)