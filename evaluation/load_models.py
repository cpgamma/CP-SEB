import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from utils.SEB_config import SemanticEntropyBrake, SEB_CONFIG
from utils.output_hidden_state import HiddenStateAnalyzer, HIDDEN_STATE_CONFIG

def load_all_models(base_model_path, model_paths, device="cuda"):
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model = base_model.eval()
    
    model_q1 = AutoModelForCausalLM.from_pretrained(
        model_paths['q1'],
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model_q2 = AutoModelForCausalLM.from_pretrained(
        model_paths['q2'],
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model_ca = AutoModelForCausalLM.from_pretrained(
        model_paths['ca'],
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model_cb = AutoModelForCausalLM.from_pretrained(
        model_paths['cb'],
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    sentence_model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
    
    seb_q1 = SemanticEntropyBrake(model_q1, tokenizer, SEB_CONFIG)
    seb_q2 = SemanticEntropyBrake(model_q2, tokenizer, SEB_CONFIG)
    
    analyzer_q1 = HiddenStateAnalyzer(model_q1, tokenizer, HIDDEN_STATE_CONFIG['target_layer'])
    analyzer_q2 = HiddenStateAnalyzer(model_q2, tokenizer, HIDDEN_STATE_CONFIG['target_layer'])
    
    print("All models loaded successfully!")
    
    return {
        'tokenizer': tokenizer,
        'base_model': base_model,
        'model_q1': model_q1,
        'model_q2': model_q2,
        'model_ca': model_ca,
        'model_cb': model_cb,
        'sentence_model': sentence_model,
        'seb_q1': seb_q1,
        'seb_q2': seb_q2,
        'analyzer_q1': analyzer_q1,
        'analyzer_q2': analyzer_q2
    }