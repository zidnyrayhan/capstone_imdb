# ingest_imdb.py
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# --- Load environment variables dari file .env
load_dotenv()

# --- Ambil API Keys dari .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Pastikan key terbaca
if not all([QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY]):
    raise ValueError("❌ One or more API keys are missing! Please check your .env file.")

# --- Load dataset IMDB
df = pd.read_csv("imdb_top_1000.csv")
print(f"✅ Total data loaded: {len(df)} movies")

# --- Pilih kolom penting
df = df[
    ["Series_Title", "Released_Year", "Genre", "IMDB_Rating", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"]
].fillna("")

# --- Gabungkan jadi satu teks panjang untuk embedding
df["combined_text"] = (
    "Title: " + df["Series_Title"].astype(str)
    + "; Year: " + df["Released_Year"].astype(str)
    + "; Genre: " + df["Genre"].astype(str)
    + "; Rating: " + df["IMDB_Rating"].astype(str)
    + "; Director: " + df["Director"].astype(str)
    + "; Stars: " + df["Star1"] + ", " + df["Star2"] + ", " + df["Star3"] + ", " + df["Star4"]
    + "; Overview: " + df["Overview"]
)

# --- Inisialisasi model embeddings OpenAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# --- Masukkan data ke Qdrant Cloud
collection_name = "imdb_movies"
print("🚀 Uploading data to Qdrant Cloud... (this may take a few minutes)")

qdrant = QdrantVectorStore.from_texts(
    texts=df["combined_text"].tolist(),
    embedding=embeddings,
    metadatas=[{
        "title": row.Series_Title,
        "year": str(row.Released_Year),
        "genre": row.Genre,
        "rating": str(row.IMDB_Rating),
        "director": row.Director
    } for _, row in df.iterrows()],
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name
)

print("✅ Data successfully inserted into Qdrant Cloud!")
