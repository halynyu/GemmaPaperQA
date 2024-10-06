<!--- BADGES: START --->
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-gemma2_2b-yellow)](https://huggingface.co/google/gemma-2-2b-it) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-gemma2_2b_it_finetuned_paperqa-yellow)](https://huggingface.co/halyn/gemma2-2b-it-finetuned-paperqa) [![Gemma2 Model](https://img.shields.io/badge/💻-Try%20Gemma2_%20Demo-blue)](https://huggingface.co/spaces/junipark/gemma_paper_qa)
<!--- BADGES: END --->
# GemmaPaperQA: AI Research Assistant
> Team Members : [이의석](https://github.com/Ui-Seok), [유하린](https://github.com/halynyu), [박준희](https://github.com/juni5184)
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

1.  **Choose a Code Execution Environment**: Choose between web-based operation method and Google Colab-based operation method.
2.  **Install Essential Libraries**: You can install libraries using the following command.  
   `pip install -r requirements.txt` 

    2-1.  **Google Colab-based Operation**: Run the `demo.ipynb`

    2-2.  **Web-based Operation**:  
        - In the `inference_chat.py` code, you need to make sure that the model address is the actual address.  
        - Run the backend server first (`python inference_chat.py`) and run streamlit-based web code (`streamlit run streamlit.py`)  

4.  **Upload a PDF**: Drag and drop or select a PDF file of the research paper you'd like to analyze.
5.  **Ask a Question**: Use the chatbot interface to ask questions related to the content of the paper.
6.  **Receive Contextual Answers**: Get detailed answers, tailored to the specific sections of the paper.
7.  **Explore More**: Interact with the assistant to explore related studies, key insights, and summaries.  
<br>

The `gemma.py` script is the backbone of the **GemmaPaperQA** system, allowing users to upload and analyze PDFs using the **Gemma2** model to answer questions about research papers.

## Workflow

1.  **PDF Upload**: The script processes the uploaded PDF, extracting text using `PyPDF2`.
2.  **Text Splitting**: Extracted text is divided into smaller chunks for more manageable processing using `CharacterTextSplitter`.
3.  **Embeddings and Vector Store**: The text chunks are converted into embeddings using `HuggingFaceEmbeddings` and stored in a **FAISS** vector store for efficient similarity-based searches.
4.  **Local LLM Integration**: The **Gemma2** model is used through a Hugging Face pipeline for generating answers based on the provided context.
5.  **Question Answering**: Questions about the PDF are answered by searching the vector store for relevant sections and generating answers using the Gemma2 model.
