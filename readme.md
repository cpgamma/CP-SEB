# Provable and Efficient Copyright Protection for LLMs

**Anonymous Research Submission**

This repository contains the complete code and experimental setup for our paper.

## File List

```
CP-SEB/
├── clustering/
│   ├── compute_embeddings.py         Generate sentence embeddings
│   └── semantic_sharding.py          HDBSCAN + GroupClusters 
├── training/
│   └── train_models.py               Train q1, q2, ca, cb models
├── evaluation/
│   ├── load_models.py                Load all models and analyzers
│   ├── generation_functions.py       Base, CP-Delta, SEB 
│   ├── generate_prompts.py           GPT-based adversarial prompts
│   ├── run_experiment.py             Main experimental pipeline
│   └── analyze_results.py            Results analysis tables
├── utils/
│   ├── metrics.py                    Perplexity, entropy, TV 
│   ├── SEB_config.py                 Semantic Entropy Brake
│   ├── output_hidden_state.py        Output hidden state analysis
│   └── mia_analysis.py         Membership inference attack metrics
└── 
```
---

## 🚀 Complete Pipeline

### Step 1: Setup

```bash
pip install -r requirements.txt
huggingface-cli login
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

### Step 2: Data Preparation

#### 2a. Compute Embeddings

```bash
python clustering/compute_embeddings.py \
    --input data/combined_train_shard.jsonl \
    --output data/embeddings.npy \
    --model sentence-transformers/sentence-t5-xxl \
    --batch-size 32
```

**Output:**
- `data/embeddings.npy` - [N × 768] embedding matrix
- `data/embeddings_metadata.json` - Metadata

---



---

#### 2b. Semantic Sharding

```bash
python clustering/semantic_sharding.py \
    --embeddings data/embeddings.npy \
    --articles data/combined_train_shard.jsonl \
    --output-dir data/shards/ \
    --min-cluster-size 5 \
    --min-samples 3
```

**Outputs:**
- `data/shards/shard_1_semantic.jsonl` - D1
- `data/shards/shard_2_semantic.jsonl` - D2
- `data/shards/anchors_shard1.npy` - A1
- `data/shards/anchors_shard2.npy` - A2
- `data/shards/metadata.json` - Clustering stats

---

### Step 3: Model Training

```bash
python training/train_models.py \
    --base-model meta-llama/Llama-3.2-1B \
    --shard1 data/shards/shard_1_semantic.jsonl \
    --shard2 data/shards/shard_2_semantic.jsonl \
    --output-dir models/ \
    --max-steps 2000 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --seed-q1 42 \
    --seed-q2 10
```

**Outputs:**
- `models/modelq1/` - Fine-tuned on shard 1 (random)
- `models/modelq2/` - Fine-tuned on shard 2 (random)
- `models/modelca/` - Fine-tuned on shard 1 (semantic)
- `models/modelcb/` - Fine-tuned on shard 2 (semantic)


---


### Step 4: Generate Adversarial Prompts

```bash
python evaluation/generate_prompts.py \
    --articles data/combined_train_shard.jsonl \
    --num-prompts 100 \
    --gpt-model gpt-4 \
    --api-key $OPENAI_API_KEY \
    --output results/prompts.csv \
    --max-retries 10
```

**Output:**
- `results/prompts.csv` - 100 adversarial prompts


---

### Step 5: Run Main Experiment

```bash
python evaluation/run_experiment.py \
    --prompts results/prompts.csv \
    --models-dir models/ \
    --base-model meta-llama/Llama-3.2-1B \
    --train-data data/combined_train_shard.jsonl \
    --embeddings data/embeddings.npy \
    --output-dir results/ \
    --max-tokens 250 \
    --temperature 1.0
```


**What it does:**
For each prompt:
1. Generate text with 4 methods (base, CP-Δ random, CP-Δ semantic, SEB)
2. Compute generation metrics (perplexity, entropy, TV distance)
3. Run MIA analysis (cosine, ROUGE, BERTScore, edit distances)
4. Analyze hidden states (memorization detection)
5. Save results incrementally (checkpointing)

**Outputs:**
- `results/ALL_GENERATION_RESULTS.csv` - Generation metrics
- `results/ALL_MIA_RESULTS.csv` - Membership inference
- `results/ALL_HIDDEN_STATE_RESULTS.csv` - Hidden state analysis
- `results/COMPREHENSIVE_RESULTS_TABLE.csv` - Combined metrics
- `results/checkpoint.pkl` - Checkpoint for resumption


**Resume:** If interrupted, rerun same command to resume

---


### Step 6: Analyze Results

```bash
python evaluation/analyze_results.py \
    --input results/COMPREHENSIVE_RESULTS_TABLE.csv \
    --output-dir results/tables/
```

**Outputs:**
- `results/tables/SUMMARY_STATISTICS.csv` - Aggregated metrics

---


