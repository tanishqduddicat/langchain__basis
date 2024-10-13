import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# defining the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books") 
db_dir = os.path.join(current_dir, "db") # here we are not mentioning any specific pdf name as we want to fetch all the pdfs in the folder 
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"books directory: {books_dir}")
print(f"persistent_directory: {persistent_dir}")

#check if the Chroma vector store already exists
if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing the vector store")

    #Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the db path"
        )
    
    # storing all the pdf's inside a list
    book_files = [b for b in os.listdir(books_dir) if b.endswith(".pdf")]

    #read the text content from each file and store it with the metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyPDFLoader(file_path)
        book_documents = loader.load()
        for doc in book_documents:
            doc.metadata = {"source" : book_file}
            documents.append(doc)

    # splitting the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    chunks = text_splitter.split_documents(documents)

    # displaying the information of the document chunks
    print("\n --- Document Chunks Information --- \n")
    print(f"Number of document chunks: {len(chunks)}")
    print("\n --------------------------------------------------- \n")

    # creating the embeddings
    print("\n --- Creating the Embeddings --- \n")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n --- Finished Creating the Embedding model --- \n")

    #Create the vector store and persist it
    print("\n --- Creating and persisting vector store --- \n")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_dir)
    print("\n --- Finished Creating and storing the vector store --- \n")

else:
    print("Vector Store Already Exists. No need to initialize.")
