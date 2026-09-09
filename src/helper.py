import os
from PyPDF2 import PdfReader
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")  
os.environ['HF_API_KEY'] =  HF_API_KEY

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"]=GROQ_API_KEY



def load_pdf_file(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        # Convert Streamlit UploadedFile to a file-like object
        pdf_reader = PdfReader(BytesIO(pdf.getvalue()))

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vector_store


def get_conversational_chain(vector_store):

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

#     llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         repo_id="meta-llama/Llama-3.1-8B-Instruct",
#         temperature=0.2,
#         max_new_tokens=512,
#         huggingfacehub_api_token=HF_API_KEY
#     )
# )
    

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}
        ),
        memory=memory
    )

    return conversation_chain