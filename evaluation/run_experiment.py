import argparse
import json
import time
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from evaluation.load_models import load_all_models
from evaluation.generation_functions import generate_with_base_model, generate_with_cp_delta, generate_with_seb
from utils.mia_analysis import MIAAnalyzer
from utils.hidden_state import HIDDEN_STATE_CONFIG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, required=True)
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--base-model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-tokens', type=int, default=250)
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser.parse_args()

def load_train_data(file_path):
    train_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get('text'):
                        train_texts.append(item['text'])
                except json.JSONDecodeError:
                    continue
    return train_texts

def save_checkpoint(output_dir, checkpoint_data):
    checkpoint_file = f"{output_dir}/checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(output_dir):
    checkpoint_file = f"{output_dir}/checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_results_incremental(output_dir, generation_results, mia_results, hidden_results):
    if generation_results:
        gen_df = pd.DataFrame(generation_results)
        gen_output = f"{output_dir}/ALL_GENERATION_RESULTS.csv"
        gen_df.to_csv(gen_output, index=False, quoting=1, escapechar='\\')
    
    if mia_results:
        mia_df = pd.DataFrame(mia_results)
        mia_output = f"{output_dir}/ALL_MIA_RESULTS.csv"
        mia_df.to_csv(mia_output, index=False)
    
    if hidden_results:
        hidden_df = pd.DataFrame(hidden_results)
        hidden_output = f"{output_dir}/ALL_HIDDEN_STATE_RESULTS.csv"
        hidden_df.to_csv(hidden_output, index=False)

def run_experiment(prompts_data, models, train_texts, embeddings, output_dir, config):
    print(f"\nStarting experiment with {len(prompts_data)} prompts")
    
    mia_analyzer = MIAAnalyzer(train_texts, embeddings, models['sentence_model'])
    
    checkpoint = load_checkpoint(output_dir)
    
    if checkpoint:
        print(f"Checkpoint found! Resuming from prompt {checkpoint['last_completed_prompt'] + 1}")
        all_generation_results = checkpoint['generation_results']
        all_mia_results = checkpoint['mia_results']
        all_hidden_results = checkpoint['hidden_results']
        start_prompt_idx = checkpoint['last_completed_prompt'] + 1
    else:
        print("Starting fresh experiment")
        all_generation_results = []
        all_mia_results = []
        all_hidden_results = []
        start_prompt_idx = 0
    
    try:
        for prompt_idx in range(start_prompt_idx, len(prompts_data)):
            pdata = prompts_data[prompt_idx]
            prompt_id = pdata['prompt_id']
            source_idx = pdata['source_idx']
            prompt = pdata['evaluation_prompt']
            
            print(f"\nPrompt {prompt_idx+1}/{len(prompts_data)} (ID: {prompt_id})")
            
            methods_data = [
                ('base_model', lambda: generate_with_base_model(
                    models['base_model'], models['tokenizer'], prompt,
                    max_new_tokens=config['max_tokens'],
                    temperature=config['temperature']
                )),
                ('cp_delta_random', lambda: generate_with_cp_delta(
                    models['model_q1'], models['model_q2'], models['tokenizer'], prompt,
                    max_new_tokens=config['max_tokens'],
                    temperature=config['temperature']
                )),
                ('cp_delta_semantic', lambda: generate_with_cp_delta(
                    models['model_ca'], models['model_cb'], models['tokenizer'], prompt,
                    max_new_tokens=config['max_tokens'],
                    temperature=config['temperature']
                )),
                ('seb', lambda: generate_with_seb(
                    models['model_q1'], models['tokenizer'], prompt, models['seb_q1'],
                    max_new_tokens=config['max_tokens'],
                    temperature=config['temperature']
                ))
            ]
            
            for method_idx, (method_name, gen_func) in enumerate(methods_data):
                print(f"  [{method_idx+1}/4] {method_name}")
                
                try:
                    text, stats, gen_time = gen_func()
                    print(f"    Generated {len(text)} chars in {gen_time:.2f}s")
                    
                    result = {
                        'prompt_id': prompt_id,
                        'source_idx': source_idx,
                        'method': method_name,
                        'prompt': prompt,
                        'generated_text': text,
                        'generation_time': gen_time,
                    }
                    
                    for key, val in stats.items():
                        if isinstance(val, dict):
                            for k2, v2 in val.items():
                                result[f'{key}_{k2}'] = v2
                        else:
                            result[key] = val
                    
                    all_generation_results.append(result)
                    
                    print(f"    MIA analysis...")
                    mia_metrics = mia_analyzer.analyze_generated_text(text, source_idx)
                    
                    mia_result = {
                        'prompt_id': prompt_id,
                        'source_idx': source_idx,
                        'method': method_name,
                        **mia_metrics
                    }
                    
                    all_mia_results.append(mia_result)
                    
                    print(f"      Cosine: {mia_metrics['cosine']:.4f}, ROUGE-L: {mia_metrics['rougeL_f1']:.4f}")
                    
                    print(f"    Hidden state analysis...")
                    original_text = train_texts[source_idx]
                    analyzer = models['analyzer_q1']
                    
                    gen_tokens = len(analyzer.tokenizer(text)['input_ids'])
                    orig_tokens = len(analyzer.tokenizer(original_text)['input_ids'])
                    compare_length = min(gen_tokens, orig_tokens)
                    
                    h_generated = analyzer.extract_hidden_state(text, compare_length)
                    h_original = analyzer.extract_hidden_state(original_text, compare_length)
                    similarity = analyzer.compute_cosine_similarity(h_generated, h_original)
                    is_leaked = similarity >= HIDDEN_STATE_CONFIG['similarity_threshold']
                    
                    hidden_result = {
                        'prompt_id': prompt_id,
                        'source_idx': source_idx,
                        'method': method_name,
                        'hidden_state_similarity': similarity,
                        'is_leaked': is_leaked,
                        'generated_length': len(text),
                        'original_length': len(original_text)
                    }
                    
                    all_hidden_results.append(hidden_result)
                    
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    continue
            
            print(f"  Saving checkpoint...")
            save_results_incremental(output_dir, all_generation_results, all_mia_results, all_hidden_results)
            
            checkpoint_data = {
                'last_completed_prompt': prompt_idx,
                'generation_results': all_generation_results,
                'mia_results': all_mia_results,
                'hidden_results': all_hidden_results
            }
            save_checkpoint(output_dir, checkpoint_data)
    
    except KeyboardInterrupt:
        print(f"\nExperiment interrupted!")
        print(f"Results saved up to prompt {prompt_idx}")
        return None
    
    print("\nFinal save...")
    save_results_incremental(output_dir, all_generation_results, all_mia_results, all_hidden_results)
    
    checkpoint_file = f"{output_dir}/checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print("\nCreating comprehensive results...")
    gen_df = pd.DataFrame(all_generation_results)
    mia_df = pd.DataFrame(all_mia_results)
    hidden_df = pd.DataFrame(all_hidden_results)
    
    comprehensive = gen_df.merge(
        mia_df[['prompt_id', 'method', 'cosine', 'rougeL_f1', 'bert_f1', 'jaccard_words', 'levenshtein_ratio']],
        on=['prompt_id', 'method'],
        how='left'
    )
    
    comprehensive = comprehensive.merge(
        hidden_df[['prompt_id', 'method', 'hidden_state_similarity', 'is_leaked']],
        on=['prompt_id', 'method'],
        how='left'
    )
    
    comprehensive_output = f"{output_dir}/COMPREHENSIVE_RESULTS_TABLE.csv"
    comprehensive.to_csv(comprehensive_output, index=False)
    print(f"Saved: {comprehensive_output}")
    
    print("\nExperiment complete!")
    
    return comprehensive

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading prompts...")
    prompts_df = pd.read_csv(args.prompts)
    prompts_data = prompts_df.to_dict('records')
    
    print("Loading training data...")
    train_texts = load_train_data(args.train_data)
    embeddings = np.load(args.embeddings)
    
    print("Loading models...")
    model_paths = {
        'q1': f"{args.models_dir}/modelq1",
        'q2': f"{args.models_dir}/modelq2",
        'ca': f"{args.models_dir}/modelca",
        'cb': f"{args.models_dir}/modelcb"
    }
    models = load_all_models(args.base_model, model_paths)
    
    config = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature
    }
    
    run_experiment(prompts_data, models, train_texts, embeddings, args.output_dir, config)

if __name__ == "__main__":
    main()