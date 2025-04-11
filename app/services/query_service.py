import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from flask import current_app
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def generate_query_engine(file_paths,read_from_text=False):
    try:
        documents = None
        if not file_paths:
            print("No file paths provided.")
            return None, None
        if not read_from_text:
            reader = SimpleDirectoryReader(input_files=[file_paths])
            documents = reader.load_data()
        else:
            documents = [Document(text=file_paths)]

        text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
        nodes = text_splitter.get_nodes_from_documents(documents)

        key = current_app.config.get("GROQ_API_KEY")
        llm = Groq(
            model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=key, response_format={"type": "json_object"},temperature=0.1,
        )
        
        cohere_api_key = current_app.config.get("COHERE_API_KEY")
        embed_model = CohereEmbedding(api_key=cohere_api_key,model_name="embed-english-v3.0",input_type="search_document",)
        Settings.embed_model = embed_model
        Settings.llm = llm
        pinecone_api_key = current_app.config.get("PINECONE_API_KEY")
        pinecone_index_name = "nexus"

        # Initialize Pinecone connection (ensure index exists or create it)
        pc = pinecone.Pinecone(api_key=pinecone_api_key,)
        # check if index exists, create if necessary - logic omitted for brevity
        pinecone_index = pc.Index(pinecone_index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build the index using the cloud vector store
        # This sends data to Pinecone for indexing
        index = VectorStoreIndex(
        nodes=nodes, # Pass nodes directly
        storage_context=storage_context,
        show_progress=True,
        )

        # Setup retriever
        retriever = index.as_retriever(similarity_top_k=50)

        cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=30)
        # Final Query Engine with reranker and retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            rerank=cohere_rerank,
            llm=llm
        )

        return query_engine, documents


    except Exception as e:
        print(f"An error occurred in generate_query_engine: {e}")
        return None, None