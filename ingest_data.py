from src.data_ingest import ingest

print("Starting data ingestion...")
ingest(csv_path="data/sme_financials.csv", persist_dir="faiss_store")
print("Data ingestion complete!")