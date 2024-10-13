from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(api_key=api_key)

template = "you are doctor house from the TV show House M.D and you should respond to me with his level of satire"
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages({
    ("system", template),
    ("human", human_template),
})

messages = chat_prompt.format_messages(text = "why should i work?")
result = chat_model.invoke(messages)

print(f"\n{result}\n\n")