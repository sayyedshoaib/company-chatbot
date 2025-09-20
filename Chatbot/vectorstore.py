import os
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_google_vertexai import VertexAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


def vectorStore():

    embeddings = GoogleGenerativeAIEmbeddings (model="models/embedding-001")

    # embeddings = VertexAIEmbeddings(model="text-embedding-004")


    # embeddings - TextEmbeddingModel.from_pretrained('text-embedding-004')

    #file_paths = ["./backend/knowledgeBase/OracleUserGuide.pdf", "./backend/knowledgeBase/OracleApplicationExpress Tutorials.pdf"]
    file_paths = ['./Chatbot/knowledgebase/GenStreet policies.pdf','./Chatbot/knowledgebase/GenStreet return policy.pdf']

    all_splits = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)

        #load the documents by pages
        docs = loader.load()
        text_splitters = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            add_start_index = True
        )
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        splits = text_splitters.split_documents(docs)

        all_splits.extend(splits)

        vector_store.add_documents(documents= all_splits)

    return vector_store