import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# defining where the persistent dir exists
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db", "chroma_db")

#defining the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


#load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

#defining user querry
query = "How old was marcus when he was adopted by Emperor Antoninus Pius"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k" : 3, "score_threshold" : 0.4}
)

relevant_docs = retriever.invoke(query)


# displaying the documents which were retrieved
print("\n------- Relevant Documents -------\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")