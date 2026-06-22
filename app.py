import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from core.embeddings import EmbeddingModel
from core.preprocessing import preprocess_text
from core.similarity import SemanticScorer
from core.generator import GenAIClient

st.set_page_config(page_title="AISCA", page_icon="🧠")

USER_RESULTS_PATH = "outputs/user_results.json"


def save_result(raw_answers: dict):
    """
    Sauvegarde une nouvelle entrée dans outputs/user_results.json.
    Ne garde que les 15 résultats les plus récents (basés sur 'timestamp').
    raw_answers : dict contenant réponses utilisateur + scores + reco.
    """
    # Charger l'existant
    if os.path.exists(USER_RESULTS_PATH):
        try:
            with open(USER_RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    # On trie par date décroissante et on ne garde que les 15 plus récents
        data = sorted(
            data,
            key=lambda r: r.get("timestamp", ""),
            reverse=True
        )[:15]


    # Ajouter le nouveau résultat
    data.append(raw_answers)

    # Trier par timestamp décroissant (plus récent -> plus ancien)
    # 'timestamp' est créé avec datetime.now().isoformat(timespec="seconds")
    data = sorted(
        data,
        key=lambda r: r.get("timestamp", ""),
        reverse=True
    )

    # Ne garder que les 15 plus récents
    data = data[:15]

    # Sauvegarder
    os.makedirs(os.path.dirname(USER_RESULTS_PATH), exist_ok=True)
    with open(USER_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def get_global_profile_label(global_score: float) -> str:
    """
    Retourne un label de profil global à partir du score agrégé.
    """
    if global_score >= 0.7:
        return "Data Scientist"
    elif global_score >= 0.5:
        return "ML Engineer"
    else:
        return "Entry-level Analyst"


st.title("AISCA – Trouve ton métier idéal dans la Data")

st.markdown("""
Bienvenue dans le MVP d'AISCA.

1. Réponds au questionnaire ci-dessous.
2. Nous analysons sémantiquement ton texte par rapport à un référentiel de compétences.
3. Nous calculons un score par bloc de compétences.
4. Nous te proposons ensuite les métiers les plus alignés avec ton profil.
""")

# Charger les référentiels
with open("data/competencies.json", "r") as f:
    competencies = json.load(f)

with open("data/jobs.json", "r") as f:
    jobs = json.load(f)

# ================== QUESTIONNAIRE STRUCTURÉ ==================
st.subheader("Questionnaire structuré")

first_name = st.text_input("Prénom")
last_name = st.text_input("Nom")


python_level = st.slider(
    "Ton niveau en Python (1 = débutant, 5 = avancé)",
    min_value=1, max_value=5, value=3
)
ml_level = st.slider(
    "Ton niveau en Machine Learning (1 = débutant, 5 = avancé)",
    min_value=1, max_value=5, value=3
)
nlp_level = st.slider(
    "Ton niveau en NLP (1 = aucun, 5 = très à l'aise)",
    min_value=1, max_value=5, value=2
)

has_projects = st.selectbox(
    "As-tu déjà réalisé au moins un projet complet en data / IA ?",
    ["Non", "Oui"]
)

tools_used = st.multiselect(
    "Quels outils as-tu déjà utilisés ?",
    ["Python", "R", "SQL", "Power BI", "Tableau", "TensorFlow", "PyTorch", "Scikit-learn", "Autre"]
)

tokenization_used = st.selectbox(
    "As-tu déjà utilisé des techniques de tokenization (découpage de texte en tokens) en NLP ?",
    ["Non", "Oui"]
)

# ================== QUESTION OUVERTE ==================
st.subheader("Description détaillée de ton profil")

skills_text = st.text_area(
    "Décris tes compétences clés :",
    height=120,
    placeholder="Exemple : Python, analyse de données, visualisation, statistiques..."
)

experience_text = st.text_area(
    "Décris tes expériences (stages, alternance, projets académiques, jobs) :",
    height=120,
    placeholder="Exemple : Stage en data analyst, projets de classification, etc."
)

projects_text = st.text_area(
    "Décris quelques projets importants que tu as réalisés :",
    height=120,
    placeholder="Exemple : Projet de prédiction, dashboard Power BI, chatbot, etc."
)

likes_text = st.text_area(
    "Décris ce que tu aimes faire (ce qui t'intéresse le plus en data / IA / tech) :",
    height=120,
    placeholder="Exemple : J'aime surtout le NLP, l'explicabilité des modèles, les visualisations, etc."
)

# On combine tout pour l'analyse sémantique
combined_text = "\n".join([
    skills_text.strip(),
    experience_text.strip(),
    projects_text.strip(),
    likes_text.strip()
]).strip()

# ================== ANALYSE ==================
if st.button("Analyser mon profil"):
    if not first_name.strip() or not last_name.strip():
        st.warning("Merci de renseigner ton prénom et ton nom.")
    elif not combined_text.strip():
        st.warning("Merci de remplir au moins une des zones de texte.")
    else:
        with st.spinner("Analyse sémantique en cours..."):
            # 1) Prétraitement
            cleaned_text = preprocess_text(combined_text)

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
            # Convertir les sliders (1-5) en valeurs (0-1) par bloc
            numeric_boosts = {
                "Data Analysis":               (python_level - 1) / 4,
                "Machine Learning":            (ml_level - 1) / 4,
                "Natural Language Processing": (nlp_level - 1) / 4,
                "Business Intelligence":       (python_level - 1) / 4,
                "Data Engineering":            (python_level - 1) / 4,
                "MLOps & Cloud":               (ml_level - 1) / 4,
            }

            block_scores = scorer.compute_block_scores(user_embeddings, embedder.model, numeric_boosts)
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
                    plan_text = genai_client.generate_plan(block_scores, combined_text)
                except Exception as e:
                    st.warning(
                        f"Impossible de générer le plan de progression avec GenAI ({e}). "
                        "L'analyse SBERT reste disponible."
                    )
                    plan_text = None

                # Bio professionnelle
                try:
                    bio_text = genai_client.generate_bio(block_scores, top_job_names, combined_text)
                except Exception as e:
                    st.warning(
                        f"Impossible de générer la bio professionnelle avec GenAI ({e}). "
                        "L'analyse SBERT reste disponible."
                    )
                    bio_text = None

        # === Sauvegarde structurée des résultats ===
        timestamp = datetime.now().isoformat(timespec="seconds")
        profile_label = get_global_profile_label(global_score)

        result_record = {
            "timestamp": timestamp,
            "user": {
                "first_name": first_name,
                "last_name": last_name
            },
            "questionnaire": {
                "python_level": python_level,
                "ml_level": ml_level,
                "nlp_level": nlp_level,
                "has_projects": has_projects,
                "tools_used": tools_used,
                "tokenization_used": tokenization_used,
                "skills_text": skills_text,
                "experience_text": experience_text,
                "projects_text": projects_text,
                "likes_text": likes_text,
                "combined_text": combined_text
            },
            "analysis": {
                "block_scores": block_scores,
                "global_score": global_score,
                "profile_label": profile_label,
                "job_scores": job_scores,
                "top_3_jobs": top_3_jobs
            },
            "genai": {
                "plan_text": plan_text,
                "bio_text": bio_text
            }
        }

        save_result(result_record)

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

            # === Radar chart des scores par bloc (bonus) ===
            if len(blocks) >= 3:
                st.subheader("Radar des compétences par bloc")

                labels = blocks
                stats = scores

                num_vars = len(labels)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

                # Fermer le graphe
                stats_cycle = stats + [stats[0]]
                angles_cycle = np.concatenate([angles, [angles[0]]])

                fig_radar, ax_radar = plt.subplots(subplot_kw=dict(polar=True))
                ax_radar.plot(angles_cycle, stats_cycle)
                ax_radar.fill(angles_cycle, stats_cycle, alpha=0.25)
                ax_radar.set_thetagrids(angles * 180 / np.pi, labels)
                ax_radar.set_ylim(0, 1)
                ax_radar.set_title("Profil de compétences par bloc")

                st.pyplot(fig_radar)

        st.subheader("Score global de couverture")
        st.write(round(global_score, 3))

        st.write(f"**Profil global suggéré :** {profile_label}")

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


        # === Historique des utilisateurs ===
        st.markdown("---")
        st.subheader("Historique des utilisateurs et métier recommandé")

        if os.path.exists(USER_RESULTS_PATH):
            try:
                with open(USER_RESULTS_PATH, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except json.JSONDecodeError:
                data = []
        else:
            data = []

        if not data:
            st.info("Aucun utilisateur enregistré pour le moment.")
        else:
            # On construit un tableau simple : Nom, Prénom, Date, Métier recommandé principal
            rows = []
            for entry in data:
                user = entry.get("user", {})
                analysis = entry.get("analysis", {})

                first = user.get("first_name", "")
                last = user.get("last_name", "")
                ts = entry.get("timestamp", "")

                top_jobs = analysis.get("top_3_jobs") or []
                # top_3_jobs est une liste de [job, score]
                if top_jobs and isinstance(top_jobs[0], list):
                    main_job = top_jobs[0][0]
                elif top_jobs and isinstance(top_jobs[0], tuple):
                    main_job = top_jobs[0][0]
                else:
                    main_job = "N/A"

                rows.append({
                    "Date": ts,
                    "Prénom": first,
                    "Nom": last,
                    "Métier recommandé (n°1)": main_job
                })

            st.dataframe(rows)

