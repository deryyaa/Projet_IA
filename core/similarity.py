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
        Retourne un dict {block_name: score}.
        """
        block_scores = {}

        for block, skills in self.competencies.items():
            # Embeddings des compétences du bloc
            comp_embeddings = model.encode(skills, convert_to_tensor=True)

            # Similarités entre le texte utilisateur et chaque compétence du bloc
            similarities = util.cos_sim(user_embeddings, comp_embeddings)

            # Pour chaque entrée utilisateur, on prend la similarité max
            max_sim = [float(sim.max()) for sim in similarities]

            # Score du bloc = moyenne des max_sim
            block_scores[block] = float(np.mean(max_sim))

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

