import os
import json
import re
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

CACHE_PATH = "outputs/cache.json"


class GenAIClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquant dans le fichier .env")
        self.client = Groq(api_key=api_key)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        try:
            with open(CACHE_PATH, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
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

    def enrich_short_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) >= 5:
            return sentence
        key = f"enrich::{sentence}"
        prompt = (
            "Tu es un assistant qui enrichit des phrases trop courtes décrivant des compétences "
            "en data / IA. Allonge en une seule phrase avec contexte technique.\n\n"
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
            if len(seg.split()) < 5:
                enriched_segments.append(self.enrich_short_sentence(seg))
            else:
                enriched_segments.append(seg)
        return " ".join(enriched_segments)

    def generate_plan(self, block_scores: Dict[str, float], user_text: str | None = None) -> str:
        if not block_scores:
            return "Aucun bloc de compétences détecté."
        sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1])
        weak_blocks = [b for b, s in sorted_blocks[:3]]
        short_text = (user_text or "")[:120].replace("\n", " ")
        key = "plan::" + ";".join(weak_blocks) + "::" + short_text
        prompt = "Tu es un coach en data / IA. Génère un plan de progression en 3 à 5 étapes.\n\nBlocs et scores:\n"
        for b, s in sorted_blocks:
            prompt += f"- {b}: {round(s, 3)}\n"
        if user_text:
            prompt += f"\nProfil:\n\"\"\"\n{user_text}\n\"\"\"\n"
        prompt += "\nStructure: Étape 1: ... Étape 2: ... etc.\n"
        return self._cached_generate(key, prompt)

    def generate_bio(self, block_scores: Dict[str, float], top_jobs: List[str], user_text: str | None = None) -> str:
        sorted_job_names = sorted(top_jobs)
        short_text = (user_text or "")[:120].replace("\n", " ")
        key = "bio::" + ";".join(sorted_job_names) + "::" + short_text
        prompt = "Écris une biographie professionnelle (4-6 lignes) en français, à la 3e personne.\n\nBlocs:\n"
        for b, s in block_scores.items():
            prompt += f"- {b}: {round(s, 3)}\n"
        prompt += "\nMétiers cibles:\n"
        for job in top_jobs:
            prompt += f"- {job}\n"
        if user_text:
            prompt += f"\nProfil:\n\"\"\"\n{user_text}\n\"\"\"\n"
        prompt += "\nBio claire, professionnelle, sans inventer de diplômes.\n"
        return self._cached_generate(key, prompt)
