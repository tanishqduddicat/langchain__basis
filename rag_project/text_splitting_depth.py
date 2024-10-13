import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter, #only work when you have .txt files
    TokenTextSplitter
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# defining the directory containing the file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.pardir.join(current_dir, "books", "think_like_a_roman_emperor.pdf")
db_dir = os.path_join(current_dir, "db")

#check if the pdf exists:
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. please check the path or its existence."
    )

# Read the text content from the file
loader = PyPDFLoader(file_path)
documents = loader.load()