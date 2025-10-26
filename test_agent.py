from src.agent import SMEAgent

def test_profit():
    print("Testing May-23 profit calculation...")
    agent = SMEAgent(persist_dir="faiss_store", llm_model="llama2")
    response = agent.answer_query("What was our profit in May-23?")
    print(f"\nResponse:\n{response}")

def test_q1():
    print("\nTesting Q1 2023 analysis...")
    agent = SMEAgent(persist_dir="faiss_store", llm_model="llama2")
    response = agent.answer_query("How did we perform in Q1 2023 (Jan-Mar)?")
    print(f"\nResponse:\n{response}")
    
def test_trends():
    print("\nTesting performance trends...")
    agent = SMEAgent(persist_dir="faiss_store", llm_model="llama2")
    response = agent.answer_query("What were our best performing months in terms of profit margin?")
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    # Run all tests
    test_profit()
    test_q1()
    test_trends()