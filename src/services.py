from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tempfile:
        tempfile.write(uploaded_file.getbuffer())
        file_path=tempfile.name

    loader = PyPDFLoader(file_path)
    data = loader.load()
    return data


