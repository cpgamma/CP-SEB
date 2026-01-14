import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

class EnhancedMetricsComputer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tv_distances = []
        self.entropies = []
        self.perplexities = []
        self.token_probabilities = []
        self.model_perplexities = {}

    def add_step_metrics(self, tv_distance=None, entropy=None, perplexity=None,
                        model_name=None, model_perplexity=None, token_prob=None):
        if tv_distance is not None:
            self.tv_distances.append(tv_distance)
        if entropy is not None:
            self.entropies.append(entropy)
        if perplexity is not None:
            self.perplexities.append(perplexity)
        if model_name and model_perplexity is not None:
            self.model_perplexities.setdefault(model_name, []).append(model_perplexity)
        if token_prob is not None:
            self.token_probabilities.append(token_prob)

    def compute_variance_metrics(self, values):
        if not values or len(values) < 2:
            return {'variance': None, 'std': None, 'iqr': None, 'range': None, 'q1': None, 'q3': None}

        arr = np.array(values)
        q1, q3 = np.percentile(arr, [25, 75])

        return {
            'variance': float(np.var(arr)),
            'std': float(np.std(arr)),
            'iqr': float(q3 - q1),
            'range': float(np.max(arr) - np.min(arr)),
            'q1': float(q1),
            'q3': float(q3)
        }

    def compute_confidence_interval(self, values, confidence=0.95):
        if not values or len(values) < 2:
            return {'mean': None, 'ci_lower': None, 'ci_upper': None, 'ci_margin': None}

        arr = np.array(values)
        mean = np.mean(arr)
        sem = scipy_stats.sem(arr)
        ci = scipy_stats.t.interval(confidence, len(arr)-1, loc=mean, scale=sem)

        return {
            'mean': float(mean),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'ci_margin': float(ci[1] - mean)
        }

    def compute_perplexity(self, probs, token_id):
        token_prob = probs[0, token_id].item()
        token_prob = max(token_prob, 1e-10)
        return np.exp(-np.log(token_prob))

    def compute_entropy(self, probs):
        log_probs = torch.log(probs + 1e-10)
        return -torch.sum(probs * log_probs).item()

    def compute_normalized_inverse_perplexity(self):
        if not self.perplexities:
            return []

        inverse_perps = [1.0/max(p, 1e-10) for p in self.perplexities]
        min_val, max_val = min(inverse_perps), max(inverse_perps)

        if max_val - min_val < 1e-10:
            return [0.5] * len(inverse_perps)

        return [(v - min_val)/(max_val - min_val) for v in inverse_perps]

    def get_comprehensive_summary_stats(self):
        summary_stats = {}

        if self.perplexities:
            summary_stats['perplexity'] = {
                **self.compute_confidence_interval(self.perplexities),
                **self.compute_variance_metrics(self.perplexities)
            }
            violation_scores = self.compute_normalized_inverse_perplexity()
            if violation_scores:
                summary_stats['violation_score'] = {
                    'mean': np.mean(violation_scores),
                    'std': np.std(violation_scores)
                }

        if self.entropies:
            summary_stats['entropy'] = {
                **self.compute_confidence_interval(self.entropies),
                **self.compute_variance_metrics(self.entropies)
            }

        if self.tv_distances:
            summary_stats['tv'] = {
                **self.compute_confidence_interval(self.tv_distances),
                **self.compute_variance_metrics(self.tv_distances)
            }

        for model_name, perps in self.model_perplexities.items():
            if perps:
                summary_stats[f'{model_name}_perplexity'] = {
                    'mean': np.mean(perps),
                    'std': np.std(perps)
                }

        return summary_stats