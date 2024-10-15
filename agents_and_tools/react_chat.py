from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

#loading the environment variables
load_dotenv()

#defining the tools
def get_current_time(*args, **kwargs):
    """returns the current time in the format of H:MM AM/PM"""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """searches the wikipedia and returns the summary of the first result"""
    from wikipedia import summary

    try:
        return summary(query, sentences = 2)
    except:
        return "couldnt find any information on that"
    


tools = [
    Tool(
        name = "current time tool",
        func= get_current_time,
        description= "fetches the current time for the llm"
    ),
    Tool(
        name="wikipedia tool",
        func=search_wikipedia,
        description="fetches some wikipedia sentences related to the query asked"
    )
]


prompt = hub.pull("michaelwei/structured-chat-agent")

# initialize a Chatopenai model
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, #use the conversation memory created to maintain context
    handle_parsing_errors=True #handles any parsing errors gracefully
)

# initial system message to provide context for the chat
inital_message = "You are an AI assistant that can provide helpful answers using available tools.\n if you are unable to provide a good answer just tell the user why"
memory.chat_memory.add_message(SystemMessage(content=inital_message))

while True:
    user_input = input("user: ")
    if user_input.lower() == "exit":
        break

    # add the users message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input" : user_input})
    print("Bot: ", response["output"])

    # add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))

