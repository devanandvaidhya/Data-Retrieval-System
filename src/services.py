from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os 
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")  
os.environ['HF_API_KEY'] =  HF_API_KEY

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        file_path = temp_file.name

    loader = PyPDFLoader(file_path)
    data = loader.load()

    return data


def get_Chunk(text):
    text_spitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    chunks =text_spitter.split_documents(text)
    return chunks


def get_vestor_store(chunks):
    # embedding_model= OllamaEmbeddings(model="embeddinggemma")
    embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
    #Vector store
    vectordb = Chroma.from_documents(documents=chunks,embedding=embedding_model)
    return vectordb

# def get_conversational_chain(vector_store):

#     llm = ChatGroq(
#         model="openai/gpt-oss-20b",
#         temperature=0.2
#     )
    

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(
#             search_kwargs={"k": 4}
#         ),
#         memory=memory
#     )

#     return conversation_chain

def get_conversational_chain(vector_store):

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    system_prompt = """
    You are a helpful AI assistant for a document question-answering system.

    Follow these rules:
    1. Answer questions using the provided document context whenever possible.
    2. Do not make up information that is not present in the context.
    3. If the answer cannot be found in the documents, clearly say that you
    don't have enough information.
    4. Keep answers concise and accurate.
    5. Use the conversation history to understand follow-up questions.

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt
        }
    )

    return conversation_chain