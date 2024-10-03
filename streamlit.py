import streamlit as st
import requests
import io
from PyPDF2 import PdfReader

FASTAPI_URL = "http://localhost:8080"  # FastAPI 서버 주소
# FASTAPI_URL = "http://203.249.64.50:8505"  # FastAPI 서버 주소

def main_page():
    st.title("Welcome to GemmaPaperQA")
    st.subheader("Upload Your Paper")

    paper = st.file_uploader("Upload Here!", type="pdf", label_visibility="hidden")
    if paper:
        st.write(f"Upload complete! File name is {paper.name}")
        st.write("Please click the button below.")
        # pdf_reader = PdfReader(paper)
        # for page in pdf_reader.pages:
        #     paper_title.append(page.extract_text())
        #     break
        # paper_name = paper_title[0].split("\n")[0]

        # st.subheader(f"You upload the <{paper_name}> paper")

        if st.button("Click Here :)"):
            # FastAPI 서버에 PDF 파일 전송
            try:
                files = {"file": (paper.name, paper, "application/pdf")}
                response = requests.post(f"{FASTAPI_URL}/upload_pdf", files=files)
                if response.status_code == 200:
                    st.success("PDF successfully uploaded to the model! Please click the button again")
                    st.session_state.messages = []
                    st.session_state.paper_name = paper.name[:-4]
                    st.session_state.page = "chat"
                else:
                    st.error(f"Failed to upload PDF to the model. Error: {response.text}")
            except requests.RequestException as e:
                st.error(f"Error connecting to the server: {str(e)}")

def chat_page():
    st.title(f"Welcome to GemmaPaperQA")
    st.subheader(f"Ask anything about {st.session_state.paper_name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Chat here !"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from FastAPI server
        response = get_response_from_fastapi(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Go back to main page"):
        st.session_state.page = "main"

def get_response_from_fastapi(prompt):
    try:
        response = requests.post(f"{FASTAPI_URL}/ask", json={"text": prompt})
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Sorry, I couldn't generate a response. Error: {response.text}"
    except requests.RequestException as e:
        return f"Sorry, there was an error connecting to the server: {str(e)}"

# 초기 페이지 설정
if "page" not in st.session_state:
    st.session_state.page = "main"

# paper_name 초기화
if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""

# 페이지 렌더링
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "chat":
    chat_page()