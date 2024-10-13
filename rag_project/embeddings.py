import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "think_like_a_roman_emperor.pdf")
db_dir = os.path.join(current_dir, "db")

#check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"the file {file_path} does not exist. please check the file path."
    )

# read the text content from the file
loader = PyPDFLoader(file_path)
documents = loader.load()

#split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

# displaying the information about the split docuuments
print("\n --- Documents Chunks Informations--- \n")
print(f"Number of document chunks: {len(docs)}")
print(f"sample chunk: \n{docs[0].page_content}\n")

# creating a function that can create vector stores
def create_vector_stores(docs, embeddings, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"\n --- Creating the vector store {store_name}---\n")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_dir
        )
        print(f"\n --- Finished creating the vector store {store_name}--- \n")
    else:
        print(f"Hey babe the vector store is already created. So there is no need to initialize")


# initializing the openAI embedding model 
print("\n --- OpenAI embeddings --- \n")
openai_embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

# creating the vector store using the openai embedding
create_vector_stores(docs, openai_embeddings, "chroma_db_openai")

# initializing the hugging face embedding model 
print("\n --- hugging face embeddings --- \n")
hf_embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# creating the vector store using the hugging face embedding
create_vector_stores(docs, hf_embeddings, "chroma_db_huggingface")

# function to query a vector store
def query_the_vector_store(store_name, query, embedding_function):
    persisten_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persisten_dir):
        print(f"\n --- Querrying the Vector store {store_name}--- \n")
        db = Chroma(
            persist_directory=persisten_dir,
            embedding_function=embedding_function
        )

        retriever = db.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k" : 3, "score_threshold" : 0.2}
        )

        relevant_docs = retriever.invoke(query)

        #display the relevant results with metadata
        print(f"\n --- Relevant Documents for {store_name} --- \n")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"source : {doc.metadata.get('source', 'unknown')}")
    else:
        print(f"the vector store {store_name} does not exist.")


#defining user query
query = "what is marcus aurelius's real name"

# Querying each vector store
query_the_vector_store("chroma_db_openai", query, openai_embeddings)
query_the_vector_store("chroma_db_huggingface", query, hf_embeddings)

print("Querrying demonstrations have been completed")