import streamlit as st
import time
from src.helper import load_pdf_file,get_text_chunks,get_vector_store,get_conversational_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)
            
def main():
    st.set_page_config('Information Retrival')
    st.header('Information Retriaval system')

    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF file", accept_multiple_files=True,type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing.."):
                raw_text = load_pdf_file(pdf_docs)
                text_chunk=get_text_chunks(raw_text)
                vector_store=get_vector_store(text_chunk)
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("Finished")



if __name__ == "__main__":
    main()