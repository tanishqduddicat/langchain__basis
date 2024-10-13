import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# defining the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with")