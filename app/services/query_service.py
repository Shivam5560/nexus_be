import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document
)
import time
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from flask import current_app
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def generate_query_engine(file_paths,namespace_id,read_from_text=False,jd=False):
    try:
        documents = None
        # --- Document Loading (Still needed to get text) ---
        if not file_paths:
            print("No file paths provided.")
            return None, None # Return None for both expected values
        if not read_from_text:
            # Used for Resumes (jd=False)
            reader = SimpleDirectoryReader(input_files=[file_paths])
            documents = reader.load_data()
        else:
            # Used for JDs (jd=True, read_from_text=True)
            # We still create the Document object to hold the text
            documents = [Document(text=file_paths)]

        # --- LLM Setup (Needed for both Resume RAG and direct JD call) ---
        key = current_app.config.get("GROQ_API_KEY")
        llm = Groq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=key,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        # Set LLM globally - might be used by Resume RAG engine implicitly
        Settings.llm = llm

        # --- Conditional Logic: RAG for Resumes, Direct LLM for JDs ---
        if not jd:
            # --- Resume Case: Setup Full RAG Pipeline ---
            print("Resume Case: Setting up RAG Query Engine.")
            text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
            nodes = text_splitter.get_nodes_from_documents(documents)

            cohere_api_key = current_app.config.get("COHERE_API_KEY")
            # Setup embeddings (needed for indexing and querying resumes)
            doc_embed_model = CohereEmbedding(api_key=cohere_api_key,model_name="embed-english-v3.0",input_type="search_document",)
            Settings.embed_model = doc_embed_model # Set default for indexing

            query_embed_model = CohereEmbedding(api_key=cohere_api_key, model_name="embed-english-v3.0", input_type="search_query")

            pinecone_api_key = current_app.config.get("PINECONE_API_KEY")
            pinecone_index_name = "nexusresume"
            pc = pinecone.Pinecone(api_key=pinecone_api_key,)
            pinecone_index = pc.Index(pinecone_index_name)
            # Ensure namespace_id is appropriate for resumes (e.g., the resume_id)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index,namespace=namespace_id)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Index the resume nodes
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True,
            )
            while True:
                stats = pinecone_index.describe_index_stats()
                if stats["namespaces"].get(namespace_id, {}).get("vector_count", 0) > 0:
                    break
                time.sleep(1)
            # Setup retriever and query engine for resumes
            retriever = index.as_retriever(similarity_top_k=10)
            retriever.embed_model = query_embed_model # Use query-specific embed model
            cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=5)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[cohere_rerank],
                llm=llm
                # Add text_qa_template here if needed for custom resume prompting
            )
            # print(query_engine) # Optional
            print("Resume RAG Engine Ready.")
            return query_engine, documents
            # --- End Resume RAG Setup ---

        else:
            # --- JD Case: Skip RAG, Return LLM directly ---
            print("JD Case: Skipping RAG setup, returning LLM directly.")
            # No splitting, no embedding, no pinecone, no indexing, no retriever needed.
            # We already have the LLM instance and the documents containing the JD text.
            # The calling code will need to use the LLM directly.
            return llm, documents
            # --- End JD Direct LLM Case ---

    except Exception as e:
        print(f"An error occurred in generate_query_engine: {e}")
        traceback.print_exc() # Print detailed traceback
        # Return None for both expected objects on error
        return None, None