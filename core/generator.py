import os
import json
import re
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

CACHE_PATH = "outputs/cache.json"


class GenAIClient:
    """
    Client GenAI basé sur Groq (LLaMA), avec:
    - chargement de la clé API depuis .env
    - caching des réponses dans outputs/cache.json
    - fonctions dédiées: enrichissement, plan de progression, bio
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquant dans le fichier .env")
        self.client = Groq(api_key=api_key)
        self.cache = self._load_cache()

    # ---------- Gestion du cache ----------

    def _load_cache(self) -> Dict[str, str]:
        try:
            with open(CACHE_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def _save_cache(self) -> None:
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _cached_generate(self, key: str, prompt: str) -> str:
        if key in self.cache:
            return self.cache[key]

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        text = response.choices[0].message.content.strip()
        self.cache[key] = text
        self._save_cache()
        return text

    # ---------- Enrichissement EF4.1 ----------

    def enrich_short_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return sentence

        words = sentence.split()
        if len(words) >= 5:
            return sentence

        key = f"enrich::{sentence}"
        prompt = (
            "Tu es un assistant qui enrichit des phrases trop courtes décrivant des compétences "
            "en data / IA / développement. Allonge la phrase suivante en une seule phrase, "
            "en gardant le même sens et en ajoutant du contexte technique (outils, méthodes, exemples).\n\n"
            f"Phrase: \"{sentence}\""
        )
        return self._cached_generate(key, prompt)

    def enrich_text_if_needed(self, text: str) -> str:
        raw_segments = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
        enriched_segments = []

        for seg in raw_segments:
            seg = seg.strip()
            if not seg:
                continue
            words = seg.split()
            if len(words) < 5:
                enriched_segments.append(self.enrich_short_sentence(seg))
            else:
                enriched_segments.append(seg)

        return " ".join(enriched_segments)

    # ---------- Plan de progression EF4.2 ----------

    def generate_plan(self, block_scores: Dict[str, float], user_text: str | None = None) -> str:
        if not block_scores:
            return "Aucun bloc de compétences détecté, impossible de générer un plan de progression."

        sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1])
        weak_blocks = [b for b, s in sorted_blocks[:3]]

        short_text = (user_text or "")[:120].replace("\n", " ")
        key = "plan::" + ";".join(weak_blocks) + "::" + short_text

        prompt = (
            "Tu es un coach en data / IA. À partir des blocs de compétences ci-dessous "
            "et de leurs scores (0 à 1, plus le score est élevé, plus le bloc est maîtrisé), "
            "génère un plan de progression personnalisé en 3 à 5 étapes principales.\n\n"
            "Concentre-toi sur les blocs les plus faibles.\n\n"
            "Blocs et scores:\n"
        )
        for b, s in sorted_blocks:
            prompt += f"- {b}: {round(s, 3)}\n"

        if user_text:
            prompt += (
                "\nVoici la description détaillée fournie par l'utilisateur "
                "(compétences, expériences, projets, centres d'intérêt) :\n"
                f"\"\"\"\n{user_text}\n\"\"\"\n"
            )

        prompt += (
            "\nStructure ta réponse ainsi :\n"
            "Étape 1: ...\nÉtape 2: ...\netc.\n"
            "Pour chaque étape, propose des actions concrètes (projets, exercices, ressources à étudier) "
            "adaptées au profil décrit.\n"
        )
        return self._cached_generate(key, prompt)

    # ---------- Bio professionnelle EF4.3 ----------

    def generate_bio(self, block_scores: Dict[str, float], top_jobs: List[str], user_text: str | None = None) -> str:
        sorted_job_names = sorted(top_jobs)
        short_text = (user_text or "")[:120].replace("\n", " ")
        key = "bio::" + ";".join(sorted_job_names) + "::" + short_text

        prompt = (
            "Écris une courte biographie professionnelle (4 à 6 lignes) en français, à la 3e personne, "
            "pour un profil orienté data / IA / ingénierie.\n\n"
            "Tu disposes des informations suivantes :\n\n"
            "Blocs de compétences et scores:\n"
        )
        for b, s in block_scores.items():
            prompt += f"- {b}: {round(s, 3)}\n"

        prompt += "\nMétiers cibles possibles:\n"
        for job in top_jobs:
            prompt += f"- {job}\n"

        if user_text:
            prompt += (
                "\nVoici la description détaillée fournie par la personne "
                "(compétences, expériences, projets, centres d'intérêt) :\n"
                f"\"\"\"\n{user_text}\n\"\"\"\n"
            )

        prompt += (
            "\nLa bio doit être claire, professionnelle, valorisante, "
            "refléter autant que possible ce texte, "
            "mais ne pas inventer de diplômes précis ou de noms d'entreprises. "
            "Mets en avant le potentiel, les compétences et l'orientation métier.\n"
        )
        return self._cached_generate(key, prompt)