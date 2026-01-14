import argparse
import json
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--articles', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--min-cluster-size', type=int, default=10)
    parser.add_argument('--min-samples', type=int, default=5)
    return parser.parse_args()

def load_articles(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        articles.append({'index': len(articles), 'text': item['text']})
                except json.JSONDecodeError:
                    continue
    return articles

def group_clusters(clusters_dict):
    cluster_sizes = {cid: len(indices) for cid, indices in clusters_dict.items()}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    J1, J2 = set(), set()
    N1, N2 = 0, 0
    
    for cluster_id, nj in sorted_clusters:
        if N1 <= N2:
            J1.add(cluster_id)
            N1 += nj
        else:
            J2.add(cluster_id)
            N2 += nj
    
    return J1, J2

def semantic_sharding(embeddings, articles, min_cluster_size, min_samples):
    distance_matrix = cosine_distances(embeddings).astype(np.float64)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        gen_min_span_tree=True,
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    clusters_dict = {}
    noise_indices = []
    
    for idx, label in enumerate(cluster_labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            clusters_dict.setdefault(label, []).append(idx)
    
    J1, J2 = group_clusters(clusters_dict)
    
    D1_core, D2_core = [], []
    for cluster_id, indices in clusters_dict.items():
        if cluster_id in J1:
            D1_core.extend(indices)
        elif cluster_id in J2:
            D2_core.extend(indices)
    
    D1, D2 = set(D1_core), set(D2_core)
    
    A1 = embeddings[D1_core]
    A2 = embeddings[D2_core]
    
    def score_to_anchors(z_embedding, anchor_embeddings):
        similarities = cosine_similarity([z_embedding], anchor_embeddings)[0]
        return np.max(similarities)
    
    for z_idx in tqdm(noise_indices, desc="Assigning noise points"):
        z_embedding = embeddings[z_idx]
        score_1 = score_to_anchors(z_embedding, A1)
        score_2 = score_to_anchors(z_embedding, A2)
        
        if score_1 >= score_2:
            D1.add(z_idx)
        else:
            D2.add(z_idx)
    
    D1, D2 = sorted(list(D1)), sorted(list(D2))
    
    metadata = {
        'num_clusters': len(clusters_dict),
        'num_noise': len(noise_indices),
        'D1_size': len(D1),
        'D2_size': len(D2),
        'balance_ratio': min(len(D1), len(D2)) / max(len(D1), len(D2))
    }
    
    return D1, D2, A1, A2, metadata

def save_shard(indices, articles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            article = articles[idx]
            json.dump({'text': article['text'], 'original_index': idx}, f)
            f.write('\n')

def main():
    args = parse_args()
    
    articles = load_articles(args.articles)
    embeddings = np.load(args.embeddings)
    
    print(f"Articles: {len(articles)}, Embeddings: {embeddings.shape}")
    
    D1, D2, A1, A2, metadata = semantic_sharding(
        embeddings, articles, args.min_cluster_size, args.min_samples
    )
    
    print(f"D1: {len(D1)} samples, D2: {len(D2)} samples")
    print(f"Balance: {metadata['balance_ratio']:.4f}")
    
    save_shard(D1, articles, f"{args.output_dir}/shard_1_semantic.jsonl")
    save_shard(D2, articles, f"{args.output_dir}/shard_2_semantic.jsonl")
    
    np.save(f"{args.output_dir}/anchors_shard1.npy", A1)
    np.save(f"{args.output_dir}/anchors_shard2.npy", A2)
    
    with open(f"{args.output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Semantic sharding complete!")

if __name__ == "__main__":
    main()