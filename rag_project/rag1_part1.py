import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
 
#importing the environment variables
load_dotenv()
model = ChatOpenAI(model="gpt-4o")

#setting the file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "think_like_a_roman_emperor.pdf")
persistent_dir = os.path.join(current_dir,"db", "chroma_db")

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store")

    #ensure that the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"the file path {file_path} does not exist. Please check the path"
        )
    
    # Read the text content from the file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)

    #display the information about the split documents
    print("\n------Document chunks Information---------\n")
    print(f"Number of Document Chunks: {len(docs)}\n")
    print("\n----------------------------------\n")
    print(f"Sample chunk: \n{docs[0].page_content}\n")
    print("\n----------------------------------\n")

    #creating the embeddings
    print("\n--- Creating the Embeddings of the chunks created --- \n")
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )
    print("\n----- Finished creating Embeddings -----\n")

    
    #create the vector store and persist automatically
    print("\n---- Creating the vector store ----\n")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    print("\n---- Finished creating the vector store ----\n")

else:
    print("Vector store already exists. No need to initialize")