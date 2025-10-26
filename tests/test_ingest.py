import os
from src.data_ingest import ingest
import chromadb
from chromadb.config import Settings


def test_ingest_creates_collection(tmp_path):
    csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'sme_financials.csv')
    persist = tmp_path / 'chroma_test'
    persist.mkdir()
    ingest(csv, str(persist))
    client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=str(persist)))
    col = client.get_collection(name='sme_finance')
    assert col.count() == 10
