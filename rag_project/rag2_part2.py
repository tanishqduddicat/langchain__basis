import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

#define the persistent directory that you would be using 
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

# defining the embedding model
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

#loading the existing vector store
db = Chroma(embedding_function=embeddings, persist_directory=persistent_dir)

# defining the user query
query = "who is the author of surrorunded by pshycopaths"

# Retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k" : 3, "score_threshold" : 0.1}
)

relevant_docs = retriever.invoke(query)

# Displaying the relevabt documents that have been retrieved using the prompt
print("\n --- Fetched Docuements that are relevant --- \n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")