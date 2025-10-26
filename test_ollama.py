from langchain_ollama import OllamaLLM

def test_ollama():
    print("Testing Ollama connection...")
    try:
        llm = OllamaLLM(model="llama2")
        response = llm.invoke("Hello! Please respond with a short greeting.")
        print("Ollama test response:", response)
        print("\nSuccess! Ollama is working correctly.")
    except Exception as e:
        print(f"\nError connecting to Ollama: {str(e)}")
        print("\nPlease check:")
        print("1. Is Ollama installed? If not, download from https://ollama.ai/download")
        print("2. Is Ollama running? Check your system tray or start it")
        print("3. Is the llama2 model pulled? Run 'ollama pull llama2' if not")

if __name__ == "__main__":
    test_ollama()