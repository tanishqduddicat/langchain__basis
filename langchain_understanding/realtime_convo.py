import os 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv() #fetching environment variables

model = ChatOpenAI(model="gpt-4o") #loading the model from openai

chat_history = [] # we will store all our variables in this array

# setting up an initial System message and appending it to the chat_history
system_prompt = "you are an excellent vocabulary expert. Your job is to take in users word which they have problem with and give a short explanation along with TWO EXAMPLE SENTENCES using the word"
system_message = SystemMessage(content= system_prompt)
chat_history.append(system_message)

# appending the intial example of human message example
human_prompt_example = "can you explain me on how to use the word Abstinence?"
human_message_example = SystemMessage(content= human_prompt_example)
chat_history.append(human_message_example)

#appending the intial example of ai response for the above human message
ai_prompt_example = "certainly here is the explanation for the word Abstinence\nMeaning: the practice of not doing or having something that is wanted or enjoyable\nExamples:\n- On one hand, my faith teaches me that abstinence is a virtue, a testament to my dedication to God.\n- Some may opt for abstinence because they're fed up with hookup culture and crave an emotional connection."
ai_message_example = AIMessage(content=ai_prompt_example)
chat_history.append(ai_message_example)
#chat loop
while True:
    querry = input("user: ")
    if querry in ["quit", "q", "exit"]:
        break
    #appending the human message to the chat history as we didnt exit
    chat_history.append(HumanMessage(content=querry))

    result = model.invoke(querry)
    final_response = result.content
    chat_history.append(AIMessage(content=final_response))

    print(f"System: {final_response}")


print("-----------------------Message History-------------------------")
for i in chat_history:
    print(i)
    print("\n\n")




