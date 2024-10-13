import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model = "gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a vocabulary expert. you give {number_of_examples} example sentences using the given word"),
        ("human", "can you explain what this {word} mean?")
    ]
)

chain = prompt_template | model | StrOutputParser() #creating LCEL pipeline and storing it in the variable
# the string parser inside the chain gets only the content portion of the output for us. otherwise we usually would have to do print(result.content)


result = chain.invoke({"number_of_examples" : "3", "word" : "bliss"}) #invoking the whole chain and also passing in the variable values inside the prompt
print(result)