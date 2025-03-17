import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from openai import api_key

from app.utils.resume_template import TEMPLATE
from flask import current_app



def generate_query_engine(file_paths,embed_model):
    try:
        if not file_paths:
            print("No file paths provided.")
            return None, None

        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()

        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

        key = current_app.config.get('GROQ_API_KEY')
        llm = Groq(model="qwen-2.5-32b", api_key=key, response_format={"type": "json_object"})

        Settings.embed_model = embed_model
        Settings.llm = llm

        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=text_splitter)

        persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage_mini"))
        vector_index.storage_context.persist(persist_dir=persist_dir)

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

        chat_engine = index.as_query_engine(similarity_top_k=2, response_mode="tree_summarize")

        return chat_engine, documents

    except Exception as e:
        print(f"An error occurred in generate_query_engine: {e}")
        return None, None
