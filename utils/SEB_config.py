import torch
import torch.nn.functional as F

SEB_CONFIG = {
    'entropy_threshold': 2.5,
    'semantic_similarity_threshold': 0.90,
    'top_k_tokens': 10,
    'noise_std': 0.30,
}

class SemanticEntropyBrake:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_matrix = None
        self._extract_embedding_matrix()

    def _extract_embedding_matrix(self):
        try:
            if hasattr(self.model, 'lm_head'):
                self.embedding_matrix = self.model.lm_head.weight.detach()
        except Exception as e:
            print(f"Warning: Could not extract embeddings: {e}")

    def compute_shannon_entropy(self, probs):
        probs = probs / probs.sum()
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy

    def check_semantic_variance(self, top_k_indices):
        if self.embedding_matrix is None:
            return False, 0.0

        top_k_embeddings = self.embedding_matrix[top_k_indices]
        top_k_embeddings = F.normalize(top_k_embeddings, p=2, dim=1)

        similarity_matrix = torch.mm(top_k_embeddings, top_k_embeddings.t())
        upper_tri = similarity_matrix.triu(diagonal=1)
        n_pairs = (len(top_k_indices) * (len(top_k_indices) - 1)) // 2

        if n_pairs == 0:
            return False, 0.0

        avg_similarity = upper_tri.sum().item() / n_pairs
        is_collapse = avg_similarity > self.config['semantic_similarity_threshold']

        return is_collapse, avg_similarity