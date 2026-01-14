import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='sentence-transformers/sentence-t5-xxl')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()

def load_articles(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        articles.append(item['text'])
                except json.JSONDecodeError:
                    continue
    return articles

def main():
    args = parse_args()
    
    print(f"Loading articles from {args.input}...")
    articles = load_articles(args.input)
    print(f"Loaded {len(articles)} articles")
    
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    
    print("Computing embeddings...")
    embeddings = []
    for i in tqdm(range(0, len(articles), args.batch_size)):
        batch = articles[i:i + args.batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    np.save(args.output, embeddings)
    print(f"Saved embeddings: {embeddings.shape} -> {args.output}")
    
    metadata_path = args.output.replace('.npy', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_articles': len(articles),
            'embedding_dim': embeddings.shape[1],
            'model': args.model
        }, f, indent=2)
    print(f"Saved metadata -> {metadata_path}")

if __name__ == "__main__":
    main()