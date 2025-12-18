from langchain_openai import ChatOpenAI
from langsmith import traceable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@traceable(
    name="langsmith_verification_test",
    metadata={
        "phase": "setup",
        "system": "nexus-rag",
        "purpose": "verify-langsmith"
    }
)
def traced_llm_call():
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )
    return llm.invoke("Reply with exactly one word: hello")

def test_langsmith_is_working():
    response = traced_llm_call()

    # Functional assertion
    assert response.content is not None
    assert len(response.content.strip()) > 0