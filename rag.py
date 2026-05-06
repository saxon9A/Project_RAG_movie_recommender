import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

# ======================
# CONFIG
# ======================
INDEX_PATH = "embeddings/faiss_index.bin"
CHUNKS_PATH = "embeddings/chunks.json"
TOP_K = 5

# ======================
# TRADUCTIONS FR → EN + SYNONYMES
# ======================
FR_TO_EN_TITLES = {
    # IA / Tech
    "l'ia": "artificial intelligence",
    "ia": "artificial intelligence",
    "intelligence artificielle": "artificial intelligence",
    "robot": "robot",
    "androïde": "android",
    "androide": "android",
    # Titres français courants
    "matrice": "the matrix",
    "interstellaire": "interstellar",
    "le parrain": "the godfather",
    "parrain": "the godfather",
    "le seigneur des anneaux": "the lord of the rings",
    "seigneur des anneaux": "the lord of the rings",
    "la guerre des étoiles": "star wars",
    "guerre des étoiles": "star wars",
    "les évadés": "the shawshank redemption",
    "le silence des agneaux": "the silence of the lambs",
    "silence des agneaux": "the silence of the lambs",
    "le loup de wall street": "the wolf of wall street",
    "loup de wall street": "the wolf of wall street",
    "le retour du roi": "the return of the king",
    "les deux tours": "the two towers",
    "la communauté de l'anneau": "the fellowship of the ring",
    "il faut sauver le soldat ryan": "saving private ryan",
    "gladiateur": "gladiator",
    "gladiator": "gladiator",
    "la liste de schindler": "schindler's list",
    "liste de schindler": "schindler's list",
    "le pianiste": "the pianist",
    "pianiste": "the pianist",
    "vol au-dessus d'un nid de coucou": "one flew over the cuckoo's nest",
    "2001 l'odyssée de l'espace": "2001 a space odyssey",
    "le bon la brute et le truand": "the good the bad and the ugly",
    "il était une fois dans l'ouest": "once upon a time in the west",
    "orange mécanique": "a clockwork orange",
    "shining": "the shining",
    "retour vers le futur": "back to the future",
    "le roi lion": "the lion king",
    "roi lion": "the lion king",
    "la belle et la bête": "beauty and the beast",
    "la reine des neiges": "frozen",
    "reine des neiges": "frozen",
    "là-haut": "up",
    "vice-versa": "inside out",
    "les indestructibles": "the incredibles",
    "indestructibles": "the incredibles",
    "monstres et compagnie": "monsters inc",
    "le monde de nemo": "finding nemo",
    "monde de nemo": "finding nemo",
    "raiponce": "tangled",
    "vaiana": "moana",
    "moi moche et méchant": "despicable me",
    "l'âge de glace": "ice age",
    "age de glace": "ice age",
    "les avengers": "the avengers",
    "avengers": "the avengers",
    "gardiens de la galaxie": "guardians of the galaxy",
    "premier contact": "arrival",
    "seul sur mars": "the martian",
    "le transperceneige": "snowpiercer",
    "transperceneige": "snowpiercer",
    "le labyrinthe de pan": "pan's labyrinth",
    "labyrinthe de pan": "pan's labyrinth",
    "la vie est belle": "life is beautiful",
    "vie est belle": "life is beautiful",
    "le voyage de chihiro": "spirited away",
    "voyage de chihiro": "spirited away",
    "princesse mononoké": "princess mononoke",
    "le château ambulant": "howl's moving castle",
    "château ambulant": "howl's moving castle",
    "mon voisin totoro": "my neighbor totoro",
    "voisin totoro": "my neighbor totoro",
    "les affranchis": "goodfellas",
    "affranchis": "goodfellas",
    "le chevalier noir": "the dark knight",
    "chevalier noir": "the dark knight",
    "dark knight": "the dark knight",
    "usual suspects": "the usual suspects",
    "fight club": "fight club",
    "forrest gump": "forrest gump",
    "seul au monde": "cast away",
    "piège de cristal": "die hard",
    "l'arme fatale": "lethal weapon",
    "arme fatale": "lethal weapon",
    "96 heures": "taken",
    "dunkerque": "dunkirk",
    "apocalypse now": "apocalypse now",
    "taxi driver": "taxi driver",
    "pulp fiction": "pulp fiction",
    "kill bill": "kill bill",
    "parasite": "parasite",
    "snowpiercer": "snowpiercer",
    "dune": "dune",
    "gravity": "gravity",
    "transcendance": "transcendence",
    "her": "her",
    "ex machina": "ex machina",
    "your name": "your name",
    "inglorious basterds": "inglourious basterds",
    "inglorious bastards": "inglourious basterds",
    "django": "django unchained",
    "batman begins": "batman begins",
    "batman": "the dark knight",
    "inception": "inception",
    "interstellar": "interstellar",
    "titanic": "titanic",
    "avatar": "avatar",
}

# Mots à ignorer
STOPWORDS = {
    "film", "movie", "cinema", "the", "une", "un", "des",
    "les", "du", "dans", "pour", "avec", "sur", "qui",
    "peut", "donner", "donne", "synopsis", "cherche",
    "similaire", "comme", "connais", "sais", "parle",
    "histoire", "raconte", "voir", "regarder", "recommande",
    "suggestion", "similaires", "ressemble", "genre",
    "meilleur", "meilleurs", "meilleure", "meilleures",
    "thème", "theme", "sujet", "about", "petit", "résumé",
    "resume", "description", "veux", "voudrais", "chercher",
    "trouver", "liste", "top", "quels", "quelles", "quel",
    "quelle", "connait", "parlant", "ayant", "portant",
    "concernant", "penses", "pense", "connais-tu", "parle",
}

# ======================
# LOAD ENV
# ======================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ======================
# LOAD DATA
# ======================
print("📦 Chargement de l'index...")
index = faiss.read_index(INDEX_PATH)
print("📄 Chargement des chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

model = SentenceTransformer("all-mpnet-base-v2")

# Liste des titres pour rapidfuzz
all_titles_lower = [doc["metadata"]["title"].lower() for doc in documents]

# ======================
# RECHERCHE PAR TITRE (rapidfuzz)
# ======================
def search_by_title(query: str):
    query_lower = query.lower()

    for fr, en in FR_TO_EN_TITLES.items():
        if fr in query_lower:
            query_lower = query_lower.replace(fr, en)

    query_words = [
        w for w in query_lower.split()
        if len(w) >= 3 and w not in STOPWORDS
    ]

    if not query_words:
        return []

    results = []
    seen_ids = set()

    for idx, doc in enumerate(documents):
        title_lower = all_titles_lower[idx]

        if len(title_lower) <= 2:
            continue

        title_words = [
            w for w in title_lower.split()
            if len(w) >= 3 and w not in STOPWORDS
        ]

        if not title_words:
            continue

        # ✅ Stratégie 1 : titre complet présent dans la question
        title_in_query = title_lower in query_lower

        # ✅ Stratégie 2 : fuzzy sur chaque mot du TITRE contre la question
        # (et non l'inverse) pour éviter les faux positifs
        word_scores = [
            max(fuzz.ratio(tw, qw) for qw in query_words)
            for tw in title_words
        ]
        matched_words = sum(1 for s in word_scores if s >= 85)
        match_ratio = matched_words / len(title_words)

        # Seuil selon longueur du titre
        if len(title_words) == 1:
            strong_match = matched_words >= 1
        elif len(title_words) == 2:
            strong_match = matched_words >= 2  # les 2 mots doivent matcher
        else:
            strong_match = match_ratio >= 0.6 and matched_words >= 2

        if not (title_in_query or strong_match):
            continue

        movie_id = doc["metadata"]["id"]
        if movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)

        results.append({
            "content": doc["content"],
            "metadata": doc["metadata"],
            "score": 0.0,
            "match_type": f"titre ({int(match_ratio*100)}%)",
            "relevance": match_ratio
        })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    for r in results:
        r.pop("relevance", None)

    return results[:TOP_K]

# ======================
# RECHERCHE SÉMANTIQUE
# ======================
def search_semantic(query: str, score_threshold: float = 0.95):
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, TOP_K)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        score = float(distances[0][i])
        if score > score_threshold:
            continue
        results.append({
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"],
            "score": score,
            "match_type": "sémantique"
        })
    return results

# ======================
# RECHERCHE HYBRIDE
# ======================
def search(query: str):
    title_results = search_by_title(query)
    threshold = 0.95 if title_results else 1.1  # était 1.0
    semantic_results = search_semantic(query, score_threshold=threshold)
    ...
    # Fusionne sans doublons, priorité au titre
    seen_ids = set()
    combined = []
    for r in title_results + semantic_results:
        movie_id = r["metadata"]["id"]
        if movie_id not in seen_ids:
            seen_ids.add(movie_id)
            combined.append(r)

    return combined[:TOP_K]

# ======================
# PROMPT
# ======================
def build_prompt(context, question):
    if not context:
        return (
            f'Aucun film trouvé dans la base pour : "{question}"\n'
            "Dis clairement que tu n'as pas cette information. "
            "Ne génère rien de ta propre connaissance."
        )

    context_text = "\n\n".join([
        f"[Film {i+1}]\n"
        f"Titre: {c['metadata']['title']}\n"
        f"Note: {c['metadata']['rating']}/10\n"
        f"Détails: {c['content']}"
        for i, c in enumerate(context)
    ])

    return f"""
Tu es un assistant expert en films. Tu réponds UNIQUEMENT en te basant sur le CONTEXTE ci-dessous.

RÈGLES STRICTES :
1. N'invente JAMAIS d'informations absentes du contexte.
2. Si la réponse est absente du contexte, réponds : "Je n'ai pas cette information dans ma base de données."
3. Cite toujours le titre exact et la note tels qu'ils apparaissent dans le contexte.
4. N'utilise JAMAIS tes connaissances générales sur les films.
5. Réponds TOUJOURS en français. Traduis les synopsis et descriptions anglais en français.
6. Pour un synopsis, traduis et résume le champ Overview présent dans les détails du film.
7. Pour des films similaires, base-toi sur les genres et thèmes des films du contexte. Ne propose pas les films de la même série.
8. Si l'utilisateur décrit un film sans donner son titre, identifie le film dont le synopsis correspond le mieux à la description parmi le contexte.

"9. Pour les questions d'opinion ('que penses-tu', 'est-ce bon', 'vaut-il le coup','parle moi'), "
"donne une présentation du film basée sur sa note et son synopsis du contexte.\n"
CONTEXTE :
{context_text}

QUESTION : {question}

RÉPONSE :
"""

# ======================
# GENERATE
# ======================
def generate_answer(question, context):
    prompt = build_prompt(context, question)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant expert en films. "
                    "Tu réponds UNIQUEMENT avec les informations du contexte fourni. "
                    "Tu n'inventes JAMAIS. Si l'info est absente, tu le dis clairement. "
                    "Tu réponds TOUJOURS en français et traduis les textes anglais."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return response.choices[0].message.content

# ======================
# MAIN LOOP
# ======================
def main():
    print("🎬 RAG Film prêt ! Tape 'quit' pour quitter.\n")
    while True:
        question = input("❓ Ta question : ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Au revoir !")
            break
        if not question:
            continue

        context = search(question)

        if context:
            print(f"\n🔍 Films trouvés ({len(context)}) :")
            for c in context:
                print(f"  [{c['match_type']}] {c['metadata']['title']} (score: {c['score']:.4f})")
        else:
            print("\n⚠️  Aucun film trouvé dans la base.")

        answer = generate_answer(question, context)
        print("\n🎯 Réponse :\n")
        print(answer)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()