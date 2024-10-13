import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
# video used for reference https://www.youtube.com/watch?v=yF9kGESAi3M&list=PLv9q2M6dFtZ-E8L0uTJ6DOdJ1r1TYkW3m&index=7

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

result = model.invoke("where is the leaning tower of pisa")
print(f"Result of the model--------------\n{result}")
print(f"To fetch the content only--------------\n{result.content}")
