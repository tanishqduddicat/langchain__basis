import os 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory


"""
steps needed to be done to store the data in a firebase server
1. Create a firebase account
2. Create a new firebase project
    - copy the project id
3. Create a firestore database in the Firebase project
4. Install the googlecloud CLI in the computer
5. Enable firestore API in the google cloud console  
"""

# setting up firebase credentials
PROJECT_ID = "chathistory-saving-gpt4o"
