import getpass
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("NVIDIA_API_KEY not found in environment variables.")
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

SYSTEM_PROMPT="""You are a helpful AI assistant."""

llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")

prompt = ChatPromptTemplate.from_messages([
                                              ("system", SYSTEM_PROMPT),
                                              ("user", "{user_input}")
                                          ])

chain = prompt | llm

response = chain.invoke({"user_input": "Hello! What's the weather like today?"})
print(response.content)

# or
response = llm.invoke("Hello! What's the weather like today?")
print(response.content)
