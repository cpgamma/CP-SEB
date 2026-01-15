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

    def compute_all_similarities(self, generated_text, target_idx):
        target_text = self.train_texts[target_idx]
        
        gen_embedding = self.sentence_model.encode([generated_text])
        target_embedding = self.embeddings[target_idx].reshape(1, -1)
        cosine_sim = cosine_similarity(gen_embedding, target_embedding)[0][0]
        
        rouge_scores = self.rouge_scorer.score(generated_text, target_text)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        rouge_l_recall = rouge_scores['rougeL'].recall
        rouge_l_precision = rouge_scores['rougeL'].precision
        
        bert_results = self.compute_bert_score(generated_text, target_text)
        bert_f1 = bert_results['bert_f1']
        bert_precision = bert_results['bert_precision']
        bert_recall = bert_results['bert_recall']
        
        gen_words = set(generated_text.lower().split())
        target_words = set(target_text.lower().split())
        jaccard_words = len(gen_words & target_words) / max(len(gen_words | target_words), 1)
        
        bigram_jaccard = self.jaccard_ngrams(generated_text, target_text, n=2)
        trigram_jaccard = self.jaccard_ngrams(generated_text, target_text, n=3)
        
        max_len_for_edit = 5000
        if len(generated_text) < max_len_for_edit and len(target_text) < max_len_for_edit:
            levenshtein_dist = Levenshtein.distance(generated_text, target_text)
            levenshtein_ratio = Levenshtein.ratio(generated_text, target_text)
            jaro_winkler = jellyfish.jaro_winkler_similarity(generated_text, target_text)
            
            max_len = min(max(len(generated_text), len(target_text)), 2000)
            gen_padded = generated_text[:max_len].ljust(max_len)
            target_padded = target_text[:max_len].ljust(max_len)
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(gen_padded, target_padded))
            hamming_ratio = 1.0 - (hamming_dist / max_len)
            
            string_match = SequenceMatcher(None, target_text, generated_text).ratio()
        else:
            levenshtein_dist = None
            levenshtein_ratio = None
            jaro_winkler = None
            hamming_dist = None
            hamming_ratio = None
            string_match = None
        
        return {
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

    def analyze_generated_text(self, generated_text, source_idx):
        target_text = self.train_texts[source_idx]
        
        gen_embedding = self.sentence_model.encode([generated_text])
        target_embedding = self.embeddings[source_idx].reshape(1, -1)
        cosine_sim = cosine_similarity(gen_embedding, target_embedding)[0][0]
        
        metrics = self.compute_all_similarities(generated_text, source_idx)
        metrics['embedding_cosine'] = cosine_sim
        
        return metrics