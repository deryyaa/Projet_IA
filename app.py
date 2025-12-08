import streamlit as st
import json
import matplotlib.pyplot as plt

from core.embeddings import EmbeddingModel
from core.preprocessing import preprocess_text
from core.similarity import SemanticScorer
from core.generator import GenAIClient

st.set_page_config(page_title="AISCA", page_icon="🧠")

st.title("AISCA – Agent Intelligent Sémantique pour la Cartographie des Compétences")

st.markdown("""
Bienvenue dans le MVP d'AISCA.

1. Saisis une description de tes compétences et expériences (en français ou en anglais).
2. Nous analysons sémantiquement ton texte par rapport à un référentiel de compétences.
3. Nous calculons un score par bloc de compétences.
4. Nous te proposons ensuite les métiers les plus alignés avec ton profil.
""")

# Charger les référentiels
with open("data/competencies.json", "r") as f:
    competencies = json.load(f)

with open("data/jobs.json", "r") as f:
    jobs = json.load(f)

user_text = st.text_area(
    "Décris tes compétences et expériences (projets, outils, technologies, missions réalisées) :",
    height=200,
    placeholder="Exemple : J'ai nettoyé des données en Python, fait des dashboards, et entraîné des modèles de régression..."
)

if st.button("Analyser mon profil"):
    if not user_text.strip():
        st.warning("Merci de saisir au moins une phrase.")
    else:
        with st.spinner("Analyse sémantique en cours..."):
            # 1) Prétraitement
            cleaned_text = preprocess_text(user_text)

            # 2) Enrichissement GenAI (EF4.1) si configuré
            genai_client = None
            try:
                genai_client = GenAIClient()
                enriched_text = genai_client.enrich_text_if_needed(cleaned_text)
            except Exception as e:
                # Clé manquante, modèle invalide, etc. -> on continue sans GenAI
                enriched_text = cleaned_text
                st.warning(
                    f"GenAI (Gemini) n'a pas pu être initialisée ({e}). "
                    "Le texte est analysé sans enrichissement automatique."
                )

            # 3) Embeddings SBERT sur le texte (potentiellement enrichi)
            embedder = EmbeddingModel()
            user_embeddings = embedder.encode([enriched_text])

            # 4) Scoring par bloc
            scorer = SemanticScorer(competencies)
            block_scores = scorer.compute_block_scores(user_embeddings, embedder.model)
            global_score = scorer.compute_global_score(block_scores)

            # 5) Scoring par métier
            job_scores = scorer.compute_job_scores(block_scores, jobs)

            # Tri des métiers par score décroissant
            sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
            top_3_jobs = sorted_jobs[:3]

            # 6) Génération du plan de progression et de la bio (si GenAI dispo)
            plan_text = None
            bio_text = None
            if genai_client is not None:
                top_job_names = [job for job, _ in top_3_jobs]

                # Plan de progression
                try:
                    plan_text = genai_client.generate_plan(block_scores)
                except Exception as e:
                    st.warning(
                        f"Impossible de générer le plan de progression avec GenAI ({e}). "
                        "L'analyse SBERT reste disponible."
                    )
                    plan_text = None

                # Bio professionnelle
                try:
                    bio_text = genai_client.generate_bio(block_scores, top_job_names)
                except Exception as e:
                    st.warning(
                        f"Impossible de générer la bio professionnelle avec GenAI ({e}). "
                        "L'analyse SBERT reste disponible."
                    )
                    bio_text = None

        # === Affichage des résultats ===
        st.subheader("Scores par bloc de compétences")
        st.json(block_scores)

        # === Graphique barres des scores par bloc ===
        if block_scores:
            st.subheader("Visualisation des scores par bloc")

            fig, ax = plt.subplots()
            blocks = list(block_scores.keys())
            scores = list(block_scores.values())

            ax.bar(blocks, scores)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score de similarité")
            ax.set_title("Scores par bloc de compétences")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            st.pyplot(fig)

        st.subheader("Score global de couverture")
        st.write(round(global_score, 3))

        st.subheader("Top 3 métiers recommandés")
        if top_3_jobs:
            for job, score in top_3_jobs:
                st.write(f"**{job}** — score : `{round(score, 3)}`")
        else:
            st.write("Aucun métier ne peut être recommandé pour l'instant (vérifier le référentiel).")

        # === Graphique barres horizontales pour tous les métiers ===
        if job_scores:
            st.subheader("Scores détaillés par métier")

            fig2, ax2 = plt.subplots()
            job_names = list(job_scores.keys())
            job_values = list(job_scores.values())

            ax2.barh(job_names, job_values)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Score de similarité")
            ax2.set_title("Scores par métier")

            st.pyplot(fig2)

        # Bonus : afficher les scores bruts en JSON
        with st.expander("Voir tous les métiers et leurs scores (JSON brut)"):
            st.json(job_scores)

        # === Section GenAI : Plan de progression & Bio ===
        st.markdown("---")
        st.subheader("Plan de progression personnalisé (GenAI)")

        if plan_text:
            st.write(plan_text)
        else:
            st.info("Le plan de progression sera généré lorsque la GenAI sera correctement configurée ou disponible.")

        st.subheader("Bio professionnelle synthétique (GenAI)")

        if bio_text:
            st.write(bio_text)
        else:
            st.info("La bio professionnelle sera générée lorsque la GenAI sera correctement configurée ou disponible.")
