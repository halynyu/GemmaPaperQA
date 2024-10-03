import subprocess
import streamlit as st

# inference_chat.py 실행
st.write("Initializing inference chat...")

# subprocess를 사용해 inference_chat.py를 백그라운드에서 실행
process = subprocess.Popen(['python3', 'inference_chat.py'])

# 이후에 Streamlit 코드 실행
st.title('Your Streamlit App')
st.write('This is a demo using Streamlit and inference_chat.py')

# 필요할 경우 inference_chat.py 프로세스 상태 확인 및 처리
# process.poll()로 실행 중인지 체크 가능
