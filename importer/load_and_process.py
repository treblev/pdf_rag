import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from supabase import create_client, Client

#from app.config import EMBEDDING_MODEL, PG_COLLECTION_NAME
load_dotenv()

loader = DirectoryLoader(
    #os.path.abspath("../source_docs"),
    "~/Coding/Github/Pdf_Rag/source_docs",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=UnstructuredPDFLoader,
    max_concurrency=40,
    sample_size=1
)
docs = loader.load()

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings()
)

chunks = text_splitter.split_documents(docs)

#print(chunks)

PGVector.from_documents(chunks
                        , embedding=embeddings
                        , collection_name= "pdf_rag"
                        , connection_string = "postgresql://postgres.svvssjtshgpsrnudvsou:[]@aws-0-us-west-1.pooler.supabase.com:5432/postgres" #"postgresql+psycopg://vvvijaya@127.0.0.1:5432/pdf_rag_vectors?password=vvvijaya"
                        , pre_delete_collection= True
                        )