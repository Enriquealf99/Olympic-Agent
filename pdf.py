import os
from llama_index.core.storage import StorageContext 
from llama_index.core import VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index")
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir = index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index



pdf_path = os.path.join("data", "2024_Summer_Olympics.pdf")
olympics_pdf = PDFReader().load_data(file = pdf_path)
olympics_index = get_index(olympics_pdf, "olympics")
olympics_engine = olympics_index.as_query_engine()