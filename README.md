<!--- BADGES: START --->
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-gemma2_2b-yellow)](https://huggingface.co/google/gemma-2-2b-it) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-gemma2_9b-yellow)](https://huggingface.co/google/gemma-2-9b-it) [![Gemma2 Model](https://img.shields.io/badge/💻-Try%20Gemma2_%20Demo-blue)](https://pdf-question-answering-gemma2.streamlit.app//)
<!--- BADGES: END --->
# GemmaPaperQA: AI Research Assistant
> Team Members : 이의석, 유하린, 박준희
> 
![gemma2-image](image/gemma2.png)


### **GemmaPaperQA is a chatbot service for research papers using Gemma2**.  
**GemmaPaperQA** helps researchers efficiently find information within PDFs, including the latest studies, by providing context-aware answers and streamlining the research process with accurate and efficient responses tailored to up-to-date findings. This AI-powered assistant enables users to quickly navigate complex research, significantly reducing the time spent searching for relevant information.

### What is Gemma2?
**Gemma2** is a high-performance, lightweight AI model available in 2B, 9B, and 27B parameter sizes, offering efficient inference and enhanced safety features for researchers and developers.
>https://ai.google.dev/gemma  
https://blog.google/technology/developers/google-gemma-2  
https://ai.google.dev/gemma/docs/model_card_2  

<br>

## Demo Tutorial

1.  **Upload a PDF**: Drag and drop or select a PDF file of the research paper you'd like to analyze.
2.  **Ask a Question**: Use the chatbot interface to ask questions related to the content of the paper.
3.  **Receive Contextual Answers**: Get detailed answers, tailored to the specific sections of the paper.
4.  **Explore More**: Interact with the assistant to explore related studies, key insights, and summaries.  
<br>

The `gemma.py` script is the backbone of the **GemmaPaperQA** system, allowing users to upload and analyze PDFs using the **Gemma2** model to answer questions about research papers.

## Workflow

1.  **PDF Upload**: The script processes the uploaded PDF, extracting text using `PyPDF2`.
2.  **Text Splitting**: Extracted text is divided into smaller chunks for more manageable processing using `CharacterTextSplitter`.
3.  **Embeddings and Vector Store**: The text chunks are converted into embeddings using `HuggingFaceEmbeddings` and stored in a **FAISS** vector store for efficient similarity-based searches.
4.  **Local LLM Integration**: The **Gemma2** model is used through a Hugging Face pipeline for generating answers based on the provided context.
5.  **Question Answering**: Questions about the PDF are answered by searching the vector store for relevant sections and generating answers using the Gemma2 model.
