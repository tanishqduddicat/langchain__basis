import os 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

#this function automatically loads the environment variables
load_dotenv()

model = ChatOpenAI(model = "gpt-4o")

system_prompt = "you are an excellent vocabulary expert. Your job is to take in users word which they have problem with and give a short explanation along with two example sentences using the word"

human_message = "can you explain me on how to use the word Abstinence?"

ai_response = "certain here is the explanation for the word Abstinence\nMeaning: the practice of not doing or having something that is wanted or enjoyable\nExamples:\n- On one hand, my faith teaches me that abstinence is a virtue, a testament to my dedication to God.\n- Some may opt for abstinence because they're fed up with hookup culture and crave an emotional connection."

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content= human_message),
    AIMessage(content=ai_response),
    HumanMessage(content="i dont understand the what redemption means")
]

result = model.invoke(messages)
print("\n----------------------------ENTIRE RESULT------------------------------\n")
print(result)
print("\n------------------------------------------------------------------------\n")
print("\n----------------------------THE FINAL AI RESPONSE------------------------------\n")
print(result.content)
print("\n----------------------------------------------------------------------------\n")


