import os 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# to use the below library you need to get the google gemini api key
# from langchain_google_genai import ChatGoogleGenerativeAI 

#this function automatically loads the environment variables
load_dotenv()

model = ChatOpenAI(model="gpt-4o")

user_question = "what is side of the triangle if the other two sides are 3 and 4"

messages = [
    SystemMessage(content="Solve the following math problems. give the final answer and there is no need of any explanation"),
    HumanMessage(content=user_question)
]

result = model.invoke(messages)

print(f"--------------------USER QUESTION----------\n{user_question}")
print(f"-----------THE RESPONSE OF THE MODEL----------------\n{result.content}")