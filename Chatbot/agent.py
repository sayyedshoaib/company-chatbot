from google.adk.agents import LlmAgent
import os

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

vector_store = None
def rag_tool(query : str) -> str:
    '''
        This is tool to retrieve information about the company using rag.
    '''

    global vector_store
    if not vector_store:
        vector_store = vectorStore()

    prompt =  hub.pull("rlm/rag-prompt")

    retrieved_docs = vector_store.similarity_search(query)
    retrieved_docs_content = "/n/n".join(doc.page_content for doc in retrieved_docs)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke({"question":query,"context":retrieved_docs_content})

    return response

    # return f"{query}" + "Return policy is 10 days . product should unsend and with all tag intact , exchange policy is 10 days product unsed and tag intact."

def db_tool(query : str) -> str:
    if "order" or "product" in query:
        return "Your product is dipatched and will be delivered soon."
    return "Transanction is successfull"

def redirect_tool(query : str) -> str:
    return "Currently live agent is not available"




rag_agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.0-flash",
    description="Handles queries related to company policies, return/exchange rules, and general company information.",
    instruction='''You are responsible for answering customer queries that involve:
- Return and exchange policies
- Refund policies and eligibility
- Company mission, values, and history
- General FAQs about services, shipping, and operations

Use the provided rag_tool to retrieve relevant documents and generate accurate, helpful responses. If the query is about a specific product, order, or transaction, redirect to the DB agent. If the query is emotional, unresolved, or requires human empathy, escalate to the Customer Associate agent.
''',
    tools=[rag_tool]
)


db_agent = LlmAgent(
    name="db_agent",
    model="gemini-2.0-flash",
    description="Handles queries related to product details, order status, delivery tracking, and refund progress.",
    instruction='''You are responsible for answering customer queries that involve:
- Product availability, specifications, and pricing
- Order status, history, and tracking
- Delivery timelines and logistics
- Refund initiation and progress

Use the db_tool to fetch structured data from the database. If the query is about company policies or general FAQs, redirect to the RAG agent. If the query is unresolved or the user requests human support, escalate to the Customer Associate agent.
''',
    tools=[db_tool]
)

cust_associate_agent = LlmAgent(
    name="cust_associate_agent",
    model="gemini-2.0-flash",
    description="Escalates unresolved or complex queries to a live human customer support executive.",
    instruction='''You are responsible for handling queries that:
- Cannot be resolved by RAG or DB agents
- Are emotionally charged or involve complaints
- Explicitly request human assistance

Use the redirect_tool to connect the user to a live customer associate. Be empathetic, acknowledge the user's frustration, and ensure a smooth handoff.
''',
    tools=[redirect_tool]
)

root_agent = LlmAgent(
    name="chatbot",
    model="gemini-2.0-flash",
    description="You are the orchestrator agent responsible for routing user queries to the appropriate sub-agent.",
    instruction='''Your role is to classify incoming user queries and route them to the correct sub-agent:
- Route to **rag_agent** for queries about return/exchange policies, company FAQs, and general information.
- Route to **db_agent** for queries about product details, order status, delivery, and refunds.
- Route to **cust_associate_agent** for unresolved issues, emotional complaints, or explicit requests for human support.
- If there is any other related to the comapany but in scope of these three agents reply "Please look into company detail page or company's community page"
- If there is question un related to the chatbot use case reply "I cannot provide this answer to you."
Use intent detection and keyword matching to determine the correct routing. If uncertain, prefer escalation to the Customer Associate agent.

Few-shot examples:
[
  {
    "user_query": "Can I return a product if I opened it?",
    "route_to": "rag_agent"
  },
  {
    "user_query": "What is your refund policy for electronics?",
    "route_to": "rag_agent"
  },
  {
    "user_query": "Tell me about your company's sustainability practices.",
    "route_to": "rag_agent"
  },
  {
    "user_query": "Is the red dress available in size M?",
    "route_to": "db_agent"
  },
  {
    "user_query": "Where is my order #12345?",
    "route_to": "db_agent"
  },
  {
    "user_query": "I requested a refund last week, when will I get it?",
    "route_to": "db_agent"
  },
  {
    "user_query": "I received the wrong item and I want to talk to someone.",
    "route_to": "cust_associate_agent"
  },
  {
    "user_query": "None of your answers helped me, I need a human.",
    "route_to": "cust_associate_agent"
  },
  {
    "user_query": "I'm really frustrated, this is the third time my order is delayed.",
    "route_to": "cust_associate_agent"
  }
]
''',
sub_agents=[rag_agent, db_agent, cust_associate_agent],
)



# root_agent = LlmAgent(
#     name="chatbot",
#     model="gemini-2.0-flash",
#     description="You are chatbot to answer user query",
#     instruction = '''
#         You are chatbot to answer user query
#     ''',
#     subagents=[rag_agent,db_agent,cust_associate_agent],
#     tools=[]
#     )