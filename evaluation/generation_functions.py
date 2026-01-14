import time
import torch
import torch.nn.functional as F
from utils.metrics import EnhancedMetricsComputer
import numpy as np

def generate_with_base_model(model, tokenizer, prompt, max_new_tokens=600,
                             min_char_length=2500, max_attempts=3, temperature=0.65):
    start_time = time.time()

    for attempt in range(max_attempts):
        metrics = EnhancedMetricsComputer()
        model_half = model.half()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        past_kv = None

        for step in range(max_new_tokens):
            with torch.inference_mode(), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                current_ids = generated_ids[:, -1:] if past_kv else generated_ids
                outputs = model_half(current_ids, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                entropy = metrics.compute_entropy(probs)
                next_token = torch.multinomial(probs, num_samples=1)
                perplexity = metrics.compute_perplexity(probs, next_token.item())
                token_prob = probs[0, next_token.item()].item()
                metrics.add_step_metrics(entropy=entropy, perplexity=perplexity, token_prob=token_prob)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        all_ids = generated_ids[0]
        new_ids = all_ids[input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if len(generated_text) >= min_char_length:
            break

    comprehensive_stats = metrics.get_comprehensive_summary_stats()
    return generated_text, comprehensive_stats, time.time() - start_time


def generate_with_cp_delta(model1, model2, tokenizer, prompt, max_new_tokens=600,
                           delta_type='max', min_char_length=2500, max_attempts=3, temperature=0.65):
    start_time = time.time()

    for attempt in range(max_attempts):
        metrics = EnhancedMetricsComputer()
        model1_half = model1.half()
        model2_half = model2.half()
        inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        past_kv1, past_kv2 = None, None

        for step in range(max_new_tokens):
            with torch.inference_mode(), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                current_ids = generated_ids[:, -1:] if past_kv1 else generated_ids

                outputs1 = model1_half(current_ids, past_key_values=past_kv1, use_cache=True)
                past_kv1 = outputs1.past_key_values
                logits1 = outputs1.logits[:, -1, :] / temperature
                probs1 = F.softmax(logits1, dim=-1)

                outputs2 = model2_half(current_ids, past_key_values=past_kv2, use_cache=True)
                past_kv2 = outputs2.past_key_values
                logits2 = outputs2.logits[:, -1, :] / temperature
                probs2 = F.softmax(logits2, dim=-1)

                if delta_type == 'max':
                    combined_probs = torch.min(probs1, probs2)
                    tv_distance = 0.5 * torch.sum(torch.abs(probs1 - probs2))
                    z_x = 1 - tv_distance
                elif delta_type == 'kl':
                    combined_probs = torch.sqrt(probs1 * probs2)
                    tv_distance = 1 - torch.sum(torch.sqrt(probs1 * probs2))
                    z_x = 1 - tv_distance

                combined_probs = combined_probs / z_x
                entropy = metrics.compute_entropy(combined_probs)
                next_token = torch.multinomial(combined_probs, num_samples=1)
                perplexity = metrics.compute_perplexity(combined_probs, next_token.item())
                token_prob = combined_probs[0, next_token.item()].item()
                perp1 = metrics.compute_perplexity(probs1, next_token.item())
                perp2 = metrics.compute_perplexity(probs2, next_token.item())

                metrics.add_step_metrics(
                    tv_distance=tv_distance.item(),
                    entropy=entropy,
                    perplexity=perplexity,
                    token_prob=token_prob,
                    model_name='q1',
                    model_perplexity=perp1
                )
                metrics.add_step_metrics(model_name='q2', model_perplexity=perp2)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        all_ids = generated_ids[0]
        new_ids = all_ids[input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if len(generated_text) >= min_char_length:
            break

    comprehensive_stats = metrics.get_comprehensive_summary_stats()
    return generated_text, comprehensive_stats, time.time() - start_time


def generate_with_seb(model, tokenizer, prompt, seb, max_new_tokens=600,
                      min_char_length=2500, max_attempts=3, temperature=0.65):
    start_time = time.time()

    for attempt in range(max_attempts):
        metrics = EnhancedMetricsComputer()
        seb_stats = {
            'entropy_checks': 0,
            'semantic_checks': 0,
            'interventions': 0,
            'tokens_generated': 0,
            'entropy_history': [],
            'semantic_similarity_history': [],
            'intervention_positions': [],
            'entropy_below_threshold_count': 0,
            'orthogonal_interventions': 0
        }

        model_half = model.half()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        past_kv = None

        for step in range(max_new_tokens):
            with torch.inference_mode(), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                current_ids = generated_ids[:, -1:] if past_kv else generated_ids
                outputs = model_half(current_ids, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                past_kv = outputs.past_key_values
                hidden_state = outputs.hidden_states[-1][:, -1:, :]
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                entropy = seb.compute_shannon_entropy(probs[0])
                seb_stats['entropy_history'].append(entropy)
                seb_stats['tokens_generated'] += 1

                if entropy < seb.config['entropy_threshold']:
                    seb_stats['entropy_checks'] += 1
                    seb_stats['entropy_below_threshold_count'] += 1

                    top_k_probs, top_k_indices = torch.topk(probs[0], k=seb.config['top_k_tokens'])
                    is_collapse, avg_sim = seb.check_semantic_variance(top_k_indices.cpu())
                    seb_stats['semantic_checks'] += 1
                    seb_stats['semantic_similarity_history'].append(avg_sim)

                    if is_collapse and seb.embedding_matrix is not None:
                        top_k_embeddings = seb.embedding_matrix[top_k_indices.cpu()]
                        top_k_embeddings = top_k_embeddings.to(hidden_state.device).half()
                        v_dominant = torch.mean(top_k_embeddings, dim=0)
                        v_dominant = v_dominant / (torch.norm(v_dominant) + 1e-8)
                        noise = torch.randn_like(hidden_state[0, 0])
                        dot_product = torch.dot(noise, v_dominant)
                        parallel_component = dot_product * v_dominant
                        orthogonal_noise = noise - parallel_component
                        orthogonal_noise = orthogonal_noise / (torch.norm(orthogonal_noise) + 1e-8)
                        orthogonal_noise = orthogonal_noise * seb.config['noise_std']
                        hidden_state[0, 0] = hidden_state[0, 0] + orthogonal_noise
                        logits = model_half.lm_head(hidden_state).squeeze(1)
                        logits = logits / temperature
                        probs = F.softmax(logits, dim=-1)
                        seb_stats['interventions'] += 1
                        seb_stats['orthogonal_interventions'] += 1
                        seb_stats['intervention_positions'].append(step)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            perplexity = metrics.compute_perplexity(probs, next_token[0].item())
            token_prob = probs[0, next_token[0].item()].item()
            metrics.add_step_metrics(entropy=entropy, perplexity=perplexity, token_prob=token_prob)

            if next_token[0].item() == tokenizer.eos_token_id:
                break

        all_ids = generated_ids[0]
        new_ids = all_ids[input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if len(generated_text) >= min_char_length:
            break

    comprehensive_stats = metrics.get_comprehensive_summary_stats()
    comprehensive_stats['seb_stats'] = seb_stats
    comprehensive_stats['seb_intervention_rate'] = seb_stats['interventions'] / max(seb_stats['tokens_generated'], 1)
    comprehensive_stats['seb_orthogonal_rate'] = seb_stats['orthogonal_interventions'] / max(seb_stats['tokens_generated'], 1)

    return generated_text, comprehensive_stats, time.time() - start_time