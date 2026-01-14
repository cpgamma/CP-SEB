import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from difflib import SequenceMatcher
import Levenshtein
import jellyfish

class MIAAnalyzer:
    def __init__(self, train_texts, embeddings, sentence_model):
        self.train_texts = train_texts
        self.embeddings = embeddings
        self.sentence_model = sentence_model
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    @staticmethod
    def compute_bert_score(candidate, reference):
        try:
            P, R, F1 = bertscore_score(
                [candidate],
                [reference],
                model_type="bert-base-uncased",
                lang="en",
                verbose=False
            )
            return {
                "bert_precision": float(P[0]),
                "bert_recall": float(R[0]),
                "bert_f1": float(F1[0]),
            }
        except Exception as e:
            return {
                "bert_precision": None,
                "bert_recall": None,
                "bert_f1": None,
            }

    @staticmethod
    def ngrams(text, n=2):
        tokens = re.findall(r"\w+", text.lower())
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    @staticmethod
    def jaccard_ngrams(t1, t2, n=2):
        ng1 = MIAAnalyzer.ngrams(t1, n)
        ng2 = MIAAnalyzer.ngrams(t2, n)
        if not ng1 and not ng2:
            return 1.0
        return len(ng1 & ng2) / len(ng1 | ng2)

    def compute_all_similarities(self, generated_text, candidate_idx):
        candidate_text = self.train_texts[candidate_idx]
        
        gen_embedding = self.sentence_model.encode([generated_text])
        cand_embedding = self.embeddings[candidate_idx].reshape(1, -1)
        cosine_sim = cosine_similarity(gen_embedding, cand_embedding)[0][0]
        
        rouge_scores = self.rouge_scorer.score(generated_text, candidate_text)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        rouge_l_recall = rouge_scores['rougeL'].recall
        rouge_l_precision = rouge_scores['rougeL'].precision
        
        bert_results = self.compute_bert_score(generated_text, candidate_text)
        bert_f1 = bert_results['bert_f1']
        bert_precision = bert_results['bert_precision']
        bert_recall = bert_results['bert_recall']
        
        gen_words = set(generated_text.lower().split())
        cand_words = set(candidate_text.lower().split())
        jaccard_words = len(gen_words & cand_words) / max(len(gen_words | cand_words), 1)
        
        bigram_jaccard = self.jaccard_ngrams(generated_text, candidate_text, n=2)
        trigram_jaccard = self.jaccard_ngrams(generated_text, candidate_text, n=3)
        
        max_len_for_edit = 5000
        if len(generated_text) < max_len_for_edit and len(candidate_text) < max_len_for_edit:
            levenshtein_dist = Levenshtein.distance(generated_text, candidate_text)
            levenshtein_ratio = Levenshtein.ratio(generated_text, candidate_text)
            jaro_winkler = jellyfish.jaro_winkler_similarity(generated_text, candidate_text)
            
            max_len = min(max(len(generated_text), len(candidate_text)), 2000)
            gen_padded = generated_text[:max_len].ljust(max_len)
            cand_padded = candidate_text[:max_len].ljust(max_len)
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(gen_padded, cand_padded))
            hamming_ratio = 1.0 - (hamming_dist / max_len)
            
            string_match = SequenceMatcher(None, candidate_text, generated_text).ratio()
        else:
            levenshtein_dist = None
            levenshtein_ratio = None
            jaro_winkler = None
            hamming_dist = None
            hamming_ratio = None
            string_match = None
        
        return {
            'candidate_idx': candidate_idx,
            'cosine': cosine_sim,
            'rougeL_f1': rouge_l_f1,
            'rougeL_recall': rouge_l_recall,
            'rougeL_precision': rouge_l_precision,
            'bert_f1': bert_f1,
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'jaccard_words': jaccard_words,
            'bigram_jaccard': bigram_jaccard,
            'trigram_jaccard': trigram_jaccard,
            'levenshtein_dist': levenshtein_dist,
            'levenshtein_ratio': levenshtein_ratio,
            'jaro_winkler': jaro_winkler,
            'hamming_dist': hamming_dist,
            'hamming_ratio': hamming_ratio,
            'string_match': string_match,
        }

    def analyze_generated_text(self, generated_text, source_indices, threshold=0.87, max_candidates=15):
        gen_embedding = self.sentence_model.encode([generated_text])
        
        similarities = cosine_similarity(gen_embedding, self.embeddings)[0]
        
        above_threshold = np.where(similarities >= threshold)[0]
        top_k_indices = np.argsort(similarities)[::-1][:max_candidates]
        
        candidate_indices = set(top_k_indices.tolist())
        candidate_indices.update(source_indices)
        candidate_indices.update(above_threshold.tolist())
        candidate_indices = sorted(list(candidate_indices),
                                   key=lambda x: similarities[x], reverse=True)[:max_candidates]
        
        results = []
        for idx in candidate_indices:
            metrics = self.compute_all_similarities(generated_text, idx)
            metrics['embedding_cosine_topk'] = similarities[idx]
            metrics['is_source_index'] = idx in source_indices
            results.append(metrics)
        
        return pd.DataFrame(results)