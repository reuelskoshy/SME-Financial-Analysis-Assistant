"""SME Business Assistant Streamlit UI"""
import streamlit as st
import os
from tempfile import NamedTemporaryFile
from src.agent import SMEAgent, profit_in_month, summarize_period, recommend_for_month
from src.data_ingest import ingest

# Initialize session state
if "persist_dir" not in st.session_state:
    st.session_state["persist_dir"] = "faiss_store"
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "llama2"
if "agent" not in st.session_state:
    st.session_state["agent"] = None

st.set_page_config(page_title="SME Business Assistant", layout="wide")
st.title("SME Business Assistant (RAG + LLM)")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    persist_dir = st.text_input(
        "FAISS store dir",
        value=st.session_state.persist_dir,
        key="persist_dir_input"
    )
    llm_model = st.text_input(
        "LLM model name (Ollama)",
        value=st.session_state.llm_model,
        key="llm_model_input"
    )
    
    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")
    csv_path = "data/sme_financials.csv"  # default path
    
    if uploaded_file is not None:
        # Save uploaded file to temp
        with NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(uploaded_file.getvalue())
            csv_path = f.name
            
        # Ingest new data
        if st.button("Process uploaded file"):
            try:
                ingest(csv_path=csv_path, persist_dir=persist_dir)
                st.success("File processed successfully!")
                st.session_state['agent'] = None  # Force agent reload
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Cleanup temp file
                try:
                    os.unlink(csv_path)
                except:
                    pass
                
    if st.button("(Re)connect"):
        st.session_state['agent'] = None

# Check if we need to initialize/reinitialize agent
if ('agent' not in st.session_state or 
    st.session_state['agent'] is None or
    st.session_state.get('last_persist_dir') != persist_dir or
    st.session_state.get('last_llm_model') != llm_model):
    try:
        st.session_state['agent'] = SMEAgent(persist_dir=persist_dir, llm_model=llm_model)
        st.session_state['last_persist_dir'] = persist_dir
        st.session_state['last_llm_model'] = llm_model
        st.sidebar.success("Connected to FAISS & LLM (if Ollama running)")
    except Exception as e:
        st.sidebar.error(f"Agent initialization error: {e}")
        st.session_state['agent'] = None

agent = st.session_state.get('agent')

query = st.text_input("Ask a question about your business data:")
if st.button("Send") and query:
    if agent is None:
        st.error("Agent not initialized. Check settings and that Chroma persistence exists.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                resp = agent.answer_query(query)
                st.write(resp)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.subheader("Quick tools (CSV-based)")
col1, col2 = st.columns(2)
with col1:
    month = st.text_input("Profit month (e.g., May-23)")
    if st.button("Compute profit") and month:
        try:
            out = profit_in_month(csv_path, month)
            st.write(out)
        except Exception as e:
            st.error(e)
with col2:
    months_text = st.text_input("Summary months (comma separated, e.g., Jan-23,Feb-23,Mar-23)")
    if st.button("Summarize period") and months_text:
        months = [m.strip() for m in months_text.split(",")]
        try:
            out = summarize_period(csv_path, months)
            st.write(out)
        except Exception as e:
            st.error(e)

st.markdown("---")
st.subheader("Recommendations")
rec_month = st.text_input("Recommendations for month (e.g., Jun-23)")
if st.button("Get recommendations") and rec_month:
    try:
        recs = recommend_for_month(csv_path, rec_month)
        for r in recs:
            st.write(f"- {r}")
    except Exception as e:
        st.error(e)
