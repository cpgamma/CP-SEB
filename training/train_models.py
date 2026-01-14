import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--shard1', type=str, required=True)
    parser.add_argument('--shard2', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='models/')
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-q1', type=int, default=42)
    parser.add_argument('--seed-q2', type=int, default=10)
    return parser.parse_args()

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def setup_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["k_proj", "v_proj", "q_proj", "gate_proj", "down_proj", "up_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)

def train_model(model, tokenizer, dataset, output_dir, args, seed):
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    training_args = TrainingArguments(
        seed=seed,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=100,
        max_steps=args.max_steps,
        report_to="none",
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        output_dir=output_dir,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def main():
    args = parse_args()
    
    login()
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading datasets...")
    dataset1 = load_dataset("json", data_files=[args.shard1], split="train")
    dataset2 = load_dataset("json", data_files=[args.shard2], split="train")
    print(f"Shard 1: {len(dataset1)} samples")
    print(f"Shard 2: {len(dataset2)} samples")
    
    print("\nTraining Model Q1 (Random Shard 1)...")
    model_q1 = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="cuda")
    model_q1 = setup_lora(model_q1)
    train_model(model_q1, tokenizer, dataset1, f"{args.output_dir}/modelq1", args, args.seed_q1)
    
    print("\nTraining Model Q2 (Random Shard 2)...")
    model_q2 = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="cuda")
    model_q2 = setup_lora(model_q2)
    train_model(model_q2, tokenizer, dataset2, f"{args.output_dir}/modelq2", args, args.seed_q2)
    
    print("\nTraining Model CA (Semantic Shard 1)...")
    model_ca = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="cuda")
    model_ca = setup_lora(model_ca)
    train_model(model_ca, tokenizer, dataset1, f"{args.output_dir}/modelca", args, args.seed_q1)
    
    print("\nTraining Model CB (Semantic Shard 2)...")
    model_cb = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="cuda")
    model_cb = setup_lora(model_cb)
    train_model(model_cb, tokenizer, dataset2, f"{args.output_dir}/modelcb", args, args.seed_q2)
    
    print("\n All models trained successfully!")

if __name__ == "__main__":
    main()