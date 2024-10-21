from datasets import load_dataset
from tqdm import tqdm

class CharacterTokenizer:
    """
    A simple character level tokenizer.
    """
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = None

        dataset = load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)
        self.fit(dataset["train"])
    
    def fit(self, dataset):
        unique_chars = set()
        for example in tqdm(dataset, desc="Fitting tokenizer"):
            unique_chars.update(example['text'])
        
        sorted_chars = sorted(unique_chars)
        for i, char in enumerate(sorted_chars):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        self.vocab_size = len(sorted_chars)
    
    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, ids):
        return ''.join([self.idx_to_char[id] for id in ids])