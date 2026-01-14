import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    return parser.parse_args()

def create_summary_statistics(df):
    summary_stats = []
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        stats = {
            'method': method,
            'total_prompts': len(method_df),
            'avg_generation_time': method_df['generation_time'].mean(),
            'avg_perplexity_mean': method_df.get('perplexity_mean', pd.Series([None])).mean(),
            'avg_entropy_mean': method_df.get('entropy_mean', pd.Series([None])).mean(),
            'avg_cosine_source': method_df['cosine'].mean(),
            'max_cosine_source': method_df['cosine'].max(),
            'avg_rougeL_f1_source': method_df['rougeL_f1'].mean(),
            'avg_bert_f1_source': method_df.get('bert_f1', pd.Series([None])).mean(),
            'avg_hidden_similarity': method_df['hidden_state_similarity'].mean(),
            'max_hidden_similarity': method_df['hidden_state_similarity'].max(),
            'hidden_leakage_count': method_df['is_leaked'].sum(),
            'hidden_leakage_rate': method_df['is_leaked'].mean(),
            'combined_leakage': ((method_df['cosine'] >= 0.85) & (method_df['is_leaked'] == True)).sum(),
            'seb_intervention_rate': method_df.get('seb_intervention_rate', pd.Series([None])).mean(),
        }
        
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)

def main():
    args = parse_args()
    
    print(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    print("\nComputing summary statistics...")
    summary_df = create_summary_statistics(df)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    summary_output = f"{args.output_dir}/SUMMARY_STATISTICS.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"\nSaved: {summary_output}")
    
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()