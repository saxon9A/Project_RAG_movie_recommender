import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "data/tmdb_5000_movies.csv"
INDEX_PATH = "embeddings/faiss_index.bin"
CHUNKS_PATH = "embeddings/chunks.json"


def parse_genres(genres_str):
    try:
        genres = json.loads(genres_str)
        return ", ".join([g["name"] for g in genres])
    except Exception:
        return ""


def movie_to_text(row):
    genres = parse_genres(row["genres"])
    return (
        f"Title: {row['title']}\n"
        f"Overview: {row['overview']}\n"
        f"Genres: {genres}\n"
        f"Release Date: {row['release_date']}\n"
        f"Rating: {row['vote_average']}\n"
        f"Language: {row['original_language']}"
    )


def main():
    print(" Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["title", "overview"])  # Supprime les films sans synopsis

    documents = []

    print("🧹 Préparation des documents...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = movie_to_text(row)

        documents.append({
            "content": text,
            "metadata": {
                "title": row["title"],
                "rating": float(row["vote_average"]),
                "id": int(i)
            }
        })

    print(f" Nombre de films : {len(documents)}")

    print("🧠 Création des embeddings...")
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")


    faiss.normalize_L2(embeddings)

    print(" Création de l'index FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f" {index.ntotal} vecteurs indexés")

    print("💾 Sauvegarde...")
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)

    print("🎉 Indexation terminée !")


if __name__ == "__main__":
    main()