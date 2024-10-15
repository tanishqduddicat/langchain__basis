from dotenv import load_dotenv
from langchain import hub
from langchain.agents import(
    AgentExecutor,
    create_react_agent
)

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

#loading the environment variables
load_dotenv()

# define a simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """returns the time in H:MM AM/PM"""
    import datetime
    now = datetime.datetime.now() #get the current time
    return now.strftime("%I: %M %p") #formatting the time in H:MM AM/PM format


# list of tools that are available to the agent
tools = [
    Tool(
        name="Time tool",
        func = get_current_time,
        # description of the tool
        description= "useful when you need to know about the current time"
    )
]

# pull in a prompt template from the hub
# ReAct = Reason and Act

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model = "gpt-4o", temperature=0.1)

# creating the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose = True
)

response = agent_executor.invoke({"input": "what time is it?"})
print(response)