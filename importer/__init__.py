import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader

loader = DirectoryLoader(
    os.path.abspath("../pdf_rag/source_docs"),
    glob="*/*.pdf",
    show_progress=True,
    loader_cls=UnstructuredPDFLoader,
    #max_concurrency=40,
    sample_size=1
)
docs = loader.load()
print(docs)