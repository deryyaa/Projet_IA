from sentence_transformers import util
import numpy as np


class SemanticScorer:
    def __init__(self, competencies: dict):
        self.competencies = competencies

    def compute_block_scores(self, user_embeddings, model, numeric_boosts: dict = None):
        raw_scores = {}

        for block, skills in self.competencies.items():
            skill_embeddings = model.encode(skills, convert_to_tensor=True)
            similarities = util.cos_sim(user_embeddings, skill_embeddings)
            sim_values = similarities[0].cpu().numpy()
            top_k = min(3, len(sim_values))
            top_scores = np.sort(sim_values)[::-1][:top_k]
            raw_scores[block] = float(np.mean(top_scores))

        values = list(raw_scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val - min_val > 0.01:
            normalized = {
                block: (score - min_val) / (max_val - min_val)
                for block, score in raw_scores.items()
            }
        else:
            normalized = raw_scores

        if numeric_boosts:
            for block, boost in numeric_boosts.items():
                if block in normalized:
                    normalized[block] = 0.70 * normalized[block] + 0.30 * boost

        return normalized

    def compute_global_score(self, block_scores: dict) -> float:
        if not block_scores:
            return 0.0
        return float(np.mean(list(block_scores.values())))

    def compute_job_scores(self, block_scores: dict, jobs: dict) -> dict:
        job_scores = {}
        for job_title, required_blocks in jobs.items():
            weighted_sum = 0.0
            total_weight = 0.0
            for block_name, weight in required_blocks.items():
                if block_name in block_scores and weight > 0:
                    weighted_sum += block_scores[block_name] * weight
                    total_weight += weight
            job_scores[job_title] = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
        return job_scores
