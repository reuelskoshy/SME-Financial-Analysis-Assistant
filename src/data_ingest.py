"""Ingest SME financial CSV into FAISS vector store with embeddings.

Usage:
    python src/data_ingest.py --csv data/sme_financials.csv --persist_dir faiss_store
"""
import argparse
import os
import pandas as pd
from src.utils import EmbeddingHelper
from src.vectorstore import FAISSStore


def row_to_text(row: pd.Series) -> str:
    # Create a structured text record for embedding and easy parsing
    return (
        f"Financial Record for {row['Month']}:\n"
        f"- Sales Revenue: ₹{int(row['Sales (INR)']):,}\n"
        f"- Total Expenses: ₹{int(row['Expenses (INR)']):,}\n" 
        f"- Active Customers: {int(row['Customers'])}\n"
        f"- Inventory Cost: ₹{int(row['Inventory Cost (INR)']):,}\n"
        f"- Marketing Spend: ₹{int(row['Marketing Spend (INR)']):,}\n"
        f"- Profit: ₹{int(row['Sales (INR)']) - int(row['Expenses (INR)']):,}\n"
        f"- Profit Margin: {((int(row['Sales (INR)']) - int(row['Expenses (INR)'])) / int(row['Sales (INR)']) * 100):.1f}%"
    )


def ingest(csv_path: str, persist_dir: str = "faiss_store"):
    os.makedirs(persist_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    model = EmbeddingHelper()

    # Initialize FAISS store
    store = FAISSStore(dimension=384)  # match EmbeddingHelper dimension

    documents = []
    metadatas = []
    embeddings = []

    for idx, row in df.iterrows():
        text = row_to_text(row)
        # Our EmbeddingHelper returns a list of vectors, get first (only) one
        emb = model.embed([text])[0]
        
        documents.append(text)
        metadatas.append({
            "month": row["Month"],
            "sales": int(row["Sales (INR)"]),
            "expenses": int(row["Expenses (INR)"]),
            "customers": int(row["Customers"]),
            "inventory_cost": int(row["Inventory Cost (INR)"]),
            "marketing_spend": int(row["Marketing Spend (INR)"])
        })
        embeddings.append(emb)

    store.add(documents=documents, embeddings=embeddings, metadatas=metadatas)
    store.save(persist_dir)
    print(f"Ingested {len(documents)} records into FAISS store at: {persist_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--persist_dir", type=str, default="faiss_store")
    args = parser.parse_args()
    ingest(args.csv, args.persist_dir)