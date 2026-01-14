import torch

HIDDEN_STATE_CONFIG = {
    'target_layer': 8,
    'pooling_method': 'last_token',
    'similarity_threshold': 0.2687,
}

class HiddenStateAnalyzer:
    def __init__(self, model, tokenizer, target_layer):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer

    def extract_hidden_state(self, text, max_length):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.target_layer]
        attention_mask = inputs.attention_mask

        pooling_method = HIDDEN_STATE_CONFIG.get('pooling_method', 'last_token')

        if pooling_method == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            hidden_state = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        elif pooling_method == 'max':
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask + (1 - mask) * -1e9
            hidden_state = masked_hidden.max(dim=1)[0]

        elif pooling_method == 'weighted':
            num_tokens = hidden_states.size(1)
            positions = torch.arange(num_tokens, device=hidden_states.device).float()
            weights = (positions + 1) / num_tokens
            weights = weights.unsqueeze(0).unsqueeze(-1)
            mask = attention_mask.unsqueeze(-1).float()
            weights = weights * mask
            weight_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / torch.clamp(weight_sum, min=1e-8)
            weighted_hidden = hidden_states * weights
            hidden_state = weighted_hidden.sum(dim=1)

        elif pooling_method == 'last_token':
            last_token_idx = attention_mask.sum(dim=1) - 1
            hidden_state = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]

        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        hidden_state = hidden_state.squeeze(0)
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.squeeze()

        return hidden_state.cpu().float()

    def compute_cosine_similarity(self, h1, h2):
        h1 = h1.flatten()
        h2 = h2.flatten()
        h1_norm = h1 / torch.clamp(h1.norm(), min=1e-8)
        h2_norm = h2 / torch.clamp(h2.norm(), min=1e-8)
        similarity = torch.dot(h1_norm, h2_norm).item()
        return similarity