import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

#load the environment variables from .env
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#load the existing vector store with the embedding function
db = Chroma(persist_directory=persist_dir,
            embedding_function=embeddings)

# define the users question
query = "list the names of all the characters that you can find from marcus's life"

#retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 1}
)

relevant_docs = retriever.invoke(query)

#display the relevant results with metadata
print("\n --- Relevant Documents ---\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

#combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question:"
    + query
    + "Relevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found just say that the answer couldnt be found due to lack of content"
)

# create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Defining the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

# display the result of the model

print(f"Users question: {query}")
print("\n------------------------------------- THE final response --------------------------------\n")
print("\n--------- FULL RESULT ----------\n")
print(result)
print("\n--------------------------- content of the AI response only --------------------------------\n")
print(result.content)

