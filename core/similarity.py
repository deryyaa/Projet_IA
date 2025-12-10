from sentence_transformers import util
import numpy as np


class SemanticScorer:
    def __init__(self, competencies: dict):
        """
        competencies : dict {block_name: [comp1, comp2, ...]}
        """
        self.competencies = competencies

    def compute_block_scores(self, user_embeddings, model):
        """
        Calcule un score de similarité sémantique pour chaque bloc de compétences.

        Ici, on ne score plus chaque mini-compétence individuellement.
        On construit une phrase descriptive par bloc (en concaténant les mini-compétences),
        puis on calcule la similarité entre :
            - le texte utilisateur (user_embeddings)
            - le bloc de compétences (embedding unique par bloc)

        Retourne un dict {block_name: score}.
        """
        block_scores = {}

        for block, skills in self.competencies.items():
            # 1) Construire une phrase qui représente le BLOC de compétences
            #    (on concatène les mini-compétences pour décrire le bloc)
            block_description = ". ".join(skills)

            # 2) Embedding du bloc (UNE seule phrase par bloc)
            block_embedding = model.encode([block_description], convert_to_tensor=True)

            # 3) Similarité cosinus entre le texte utilisateur (agrégé) et ce bloc
            similarities = util.cos_sim(user_embeddings, block_embedding)
            # similarities est un tensor 1x1 -> on récupère la valeur scalaire
            block_scores[block] = float(similarities[0][0])

        return block_scores


    def compute_global_score(self, block_scores: dict) -> float:
        """
        Calcule un score global en moyennant les scores des blocs.
        """
        if not block_scores:
            return 0.0
        return float(np.mean(list(block_scores.values())))

    def compute_job_scores(self, block_scores: dict, jobs: dict) -> dict:
        """
        Calcule un score pour chaque métier en fonction des blocs requis.

        jobs : dict {job_title: {block_name: weight, ...}}
        Retourne un dict {job_title: score}
        """
        job_scores = {}

        for job_title, required_blocks in jobs.items():
            weighted_sum = 0.0
            total_weight = 0.0

            # required_blocks est maintenant un dict {block_name: weight}
            for block_name, weight in required_blocks.items():
                if block_name in block_scores and weight > 0:
                    weighted_sum += block_scores[block_name] * weight
                    total_weight += weight

            if total_weight > 0:
                job_scores[job_title] = float(weighted_sum / total_weight)
            else:
                # Si aucun bloc trouvé, on met 0 pour ce métier
                job_scores[job_title] = 0.0

        return job_scores

