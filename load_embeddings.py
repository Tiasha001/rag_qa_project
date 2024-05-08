from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import chromadb
import dotenv
import os


dotenv.load_dotenv()
INPUT_FILE_PATH = os.environ.get("INPUT_FILE_PATH")


class LoadEmbedding:
    def __init__(self) -> None:
        self.input_file = INPUT_FILE_PATH
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.client = chromadb.HttpClient(host='localhost', port=8000)
    
    def load_text_file(self):
        loader = PyPDFLoader(self.input_file)
        pages = loader.load()
        return pages


    def recursive_text_splitter(self, pages):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        docs = splitter.split_documents(pages)
        # print(docs)
        return docs
        

    def embed_data(self, docs):
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            client=self.client,
            documents=docs,
            embedding=embedding,
        )
        
    def execute(self):
        pages = self.load_text_file()
        docs = self.recursive_text_splitter(pages)
        db = self.embed_data(docs)
   
