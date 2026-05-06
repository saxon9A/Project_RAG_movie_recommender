# Project_RAG_movie_recommender


Un système de recommandation de films basé sur l'architecture RAG (Retrieval-Augmented Generation), construit de zéro avec Python, FAISS et Groq — sans LangChain ni LlamaIndex.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-llama--3.1--8b-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)

---

##  Présentation

Ce projet implémente un assistant conversationnel capable de :

- Donner le **synopsis** d'un film (en français, même si le titre est en anglais)
- Faire des **recommandations** basées sur un thème, un genre ou une description
- Trouver un film à partir d'une **description vague** sans connaître le titre
- Répondre aux questions sur les films du dataset **TMDB 5000**

Le système combine une **recherche sémantique vectorielle** (FAISS) et une **recherche par titre** (fuzzy matching avec rapidfuzz) pour maximiser la pertinence des résultats.

---

## 🏗️ Architecture
