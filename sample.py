import torch
from dataclasses import dataclass
from transformers import AutoTokenizer
from utils import load_checkpoint

@dataclass
class SampleConfig:
    prompt: str = "\n"
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/best.pth"
    tokenizer_name: str = "./character_tokenizer"
    max_tokens: int = 1024
    temperature: float = 0.8
    top_k: int = 10
    top_p: float = 0.9
    random_seed: int = 1
    
SAMPLE_CONFIG = SampleConfig()

print(SAMPLE_CONFIG)

# set random seed
torch.manual_seed(SAMPLE_CONFIG.random_seed)

DEVICE = torch.device(SAMPLE_CONFIG.device)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    filtered_logits = logits.clone()

    if top_k > 0:
        indices_to_remove = filtered_logits < torch.topk(filtered_logits, top_k)[0][..., -1, None]
        filtered_logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        filtered_logits[indices_to_remove] = filter_value
    
    return filtered_logits

def generate_text(model, tokenizer):
    print(SAMPLE_CONFIG.prompt, end="")
    # tokenize the prompt
    input_ids = tokenizer.encode(SAMPLE_CONFIG.prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(SAMPLE_CONFIG.device)
    
    for _ in range(SAMPLE_CONFIG.max_tokens):
        # take last sequence_length tokens as context
        context = input_ids[:, -model.config.sequence_length:]  # Use model's config
        
        # get predictions
        logits = model(context)
        logits = logits[:, -1, :] / SAMPLE_CONFIG.temperature
        
        # apply top-k and top-p filtering
        filtered_logits = top_k_top_p_filtering(
            logits,
            top_k=SAMPLE_CONFIG.top_k,
            top_p=SAMPLE_CONFIG.top_p
        )
        
        # sample from the filtered distribution
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # append to generated sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)
        decoded_token = tokenizer.decode([next_token.item()])[0]
        print(decoded_token, end="")
    
    print("")

def main():
    if SAMPLE_CONFIG.tokenizer_name == "./character_tokenizer":
        from character_tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(SAMPLE_CONFIG.tokenizer_name)
    
    model = load_checkpoint(SAMPLE_CONFIG.checkpoint_path)
    model.to(DEVICE)
    model.eval()

    generate_text(model, tokenizer)
    
if __name__ == "__main__":
    main()