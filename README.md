# SME Financial Analysis Assistant

A powerful AI-powered financial analysis tool for Small and Medium Enterprises (SMEs) that combines LangChain, Ollama, and FAISS for intelligent financial data analysis.

## Features

- Load and analyze financial data from CSV files
- Vector-based similarity search using FAISS
- Local LLM integration via Ollama
- Interactive chat interface with Streamlit
- Automated financial metrics calculation:
  - Profit and margins
  - Revenue trends
  - Customer analytics
  - Inventory costs
  - Marketing ROI

## Quick Start (Windows)

1. Install [Ollama](https://ollama.ai) and pull the llama2 model:
```powershell
ollama pull llama2
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Ingest sample data:
```powershell
python src/data_ingest.py --csv data/sme_financials.csv --persist_dir faiss_store
```

4. Run the Streamlit interface:
```powershell
streamlit run streamlit_app.py
```

5. Try sample queries:
- "What was our profit in May-23?"
- "Analyze our performance in Q1 2023"
- "Which months had the best profit margins?"
- "How do marketing spend and revenue correlate?"

## Project Structure

```
sme-analysis/
├── data/                    # Financial data
│   └── sme_financials.csv  # Sample dataset
├── src/                    # Source code
│   ├── agent.py           # LangChain + Ollama integration
│   ├── data_ingest.py     # Data processing
│   ├── utils.py           # Helper functions
│   └── vectorstore.py     # FAISS wrapper
├── tests/                  # Test files
│   └── test_agent.py      # Agent tests
├── streamlit_app.py        # Web interface
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## System Requirements

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- llama2 model pulled via Ollama
- 8GB+ RAM recommended

## Notes

- Make sure Ollama is running (the LangChain wrapper calls the local Ollama daemon)
- For production use:
  - Add authentication
  - Implement logging
  - Add input validation
  - Tune system prompts
  - Add error handling

## Files

- `src/data_ingest.py` - Data ingestion pipeline
- `src/agent.py` - LangChain agent with FAISS retrieval
- `src/utils.py` - Embedding and formatting helpers
- `streamlit_app.py` - Web UI implementation
- `data/sme_financials.csv` - Example dataset
- `tests/test_agent.py` - Agent test suite

Sample queries to try in the UI

- "What was the profit in May 2023?"
- "Suggest 2 strategies to improve profits in June 2023."
- "Summarize business performance in Q1 2023."

License: MIT
