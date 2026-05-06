import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from rapidfuzz import fuzz


st.set_page_config(
    page_title="CinéBot Elite",
    page_icon="🎬",
    layout="wide"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    /* Fond & Base */
    .stApp { background-color: #08090b; color: #e1e1e1; font-family: 'Plus Jakarta Sans', sans-serif; }
    header, footer { visibility: hidden; }

    /* Fix du Centrage (Supprime l'effet dézoomé et l'espace vide énorme) */
    .main .block-container {
        max-width: 850px !important;
        padding: 2rem 1rem 8rem !important;
        margin: auto;
    }

    /* Bulles de Chat */
    [data-testid="stChatMessage"] {
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    [data-testid="stChatMessageUser"] {
        background: rgba(232, 184, 75, 0.08) !important;
        border-color: rgba(232, 184, 75, 0.2) !important;
    }

    [data-testid="stChatMessageAssistant"] {
        background: rgba(255, 255, 255, 0.03) !important;
    }

    /* Cartes de Films */
    .movie-card {
        background: #111418;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #e8b84b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .movie-title { font-weight: 800; color: #fff; font-size: 1.1rem; }
    .rating-badge { background: #e8b84b; color: #000; padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem; }

    /* Barre de recherche flottante */
    .stChatInputContainer {
        max-width: 800px !important;
        bottom: 2rem !important;
        background: rgba(8, 9, 11, 0.95) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* Accueil */
    .hero-section { text-align: center; padding: 5rem 0; }
    .hero-title { font-size: 3.5rem; font-weight: 800; background: linear-gradient(90deg, #fff, #e8b84b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0b0d11 !important; border-right: 1px solid rgba(255,255,255,0.05) !important; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def load_all():
    # Gestion sécurisée de la clé API (Cloud vs Local)[cite: 1]
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("🚨 Clé API Groq introuvable. Ajoutez-la dans les Secrets Streamlit.")
        st.stop()

    client = Groq(api_key=api_key)
    index = faiss.read_index("embeddings/faiss_index.bin")
    with open("embeddings/chunks.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    model = SentenceTransformer("all-mpnet-base-v2")
    return client, index, docs, model


try:
    client, index, documents, model = load_all()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []


# ======================
# LOGIQUE DE RECHERCHE & GÉNÉRATION
# ======================
def search(query, k=4):
    vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(vector)
    distances, indices = index.search(vector, k)
    return [documents[idx] for idx in indices[0] if idx != -1]


def generate_response(question, context):
    context_str = "\n\n".join(
        [f"Film: {c['metadata']['title']} (Note: {c['metadata']['rating']}/10)\nSynopsis: {c['content']}" for c in
         context])
    prompt = f"Tu es un expert cinéma. Réponds en français.\nContexte:\n{context_str}\n\nQuestion: {question}"

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content


# ======================
# BARRE LATÉRALE (SIDEBAR)
# ======================
with st.sidebar:
    st.markdown("<h2 style='color:#e8b84b;'>🎬 CinéBot Elite</h2>", unsafe_allow_html=True)
    st.divider()

    st.write("📌 **Suggestions**")
    for s in ["Films thrillers psychologiques", "Meilleurs films de Science fiction", "Un drame poignant","Films d'action Futuristes"]:
        if st.button(s, use_container_width=True):
            st.session_state.current_query = s

    st.divider()
    if st.button("🗑️ Reset Discussion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if not st.session_state.messages and "current_query" not in st.session_state:
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Trouvez votre prochain film.</h1>
            <p style="color:#808495; font-size:1.2rem;">Analyse sémantique de 5000 films pour des recommandations sur mesure.</p>
        </div>
    """, unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


query = st.chat_input("Décrivez un film ou posez une question...")
if "current_query" in st.session_state:
    query = st.session_state.pop("current_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Exploration de la base de données..."):
            results = search(query)


            if results:
                for doc in results:
                    st.markdown(f"""
                    <div class="movie-card">
                        <span class="movie-title">🎥 {doc['metadata']['title']}</span>
                        <span class="rating-badge">⭐ {doc['metadata']['rating']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            answer = generate_response(query, results)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})