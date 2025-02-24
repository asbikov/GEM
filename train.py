import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from model import GEM, GEMConfig
from utils import save_checkpoint

# # swap Memory implementation to debug
# from memory import Memory
# from memory_debug import swap_class_implementation, Memory_Autograd, Memory_Transformer
# swap_class_implementation(Memory, Memory_Autograd)


@dataclass
class TrainConfig:
    device: str = "cuda"
    dataset_name: str = "karpathy/tiny_shakespeare"
    tokenizer_name: str = "./character_tokenizer"
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 12
    learning_rate: float = 1e-3
    checkpoints_dir: str = "checkpoints"
    random_seed: int = 1


TRAIN_CONFIG = TrainConfig()

GEM_CONFIG = GEMConfig(
    vocabulary_size=None,
    sequence_length=64,
    minibatch_size=16,
    memory_size=64,
    embedding_size=128,
    n_layers=4
)

print(TRAIN_CONFIG)
print(GEM_CONFIG)

# set random seed
torch.manual_seed(TRAIN_CONFIG.random_seed)

DEVICE = torch.device(TRAIN_CONFIG.device if torch.cuda.is_available() else "cpu")


def load_tokenized_dataset(dataset_name, tokenizer, max_length):
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    def tokenize_and_chunk(examples):
        tokenized = [tokenizer.encode(text) for text in examples["text"]]
        chunks = []
        for ids in tokenized:
            chunks.extend([ids[i:i + max_length] for i in range(0, len(ids) - max_length, max_length)])
        return {"input_ids": chunks}

    tokenized_dataset = dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    return tokenized_dataset


def epoch(model, data_loader, optimizer=None, scheduler=None, is_training=True):
    model.train() if is_training else model.eval()
    total_loss = 0
    moving_loss = None
    progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validating")

    for i, batch in enumerate(progress_bar):
        with torch.set_grad_enabled(is_training):
            input_ids = batch["input_ids"].to(DEVICE)
            input_sequence = input_ids[:, :-1]
            target_sequence = input_ids[:, 1:]

            logits = model(input_sequence)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_sequence.view(-1), ignore_index=-1)

            if is_training:
                (loss / TRAIN_CONFIG.gradient_accumulation_steps).backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if (i + 1) % TRAIN_CONFIG.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        total_loss += loss.item()
        moving_loss = 0.99 * moving_loss + 0.01 * loss.item() if moving_loss is not None else loss.item()
        progress_bar.set_postfix(moving_loss=f"{moving_loss:.4f}")

    return total_loss / len(data_loader)


def main():
    if TRAIN_CONFIG.tokenizer_name == "./character_tokenizer":
        from character_tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG.tokenizer_name)

    GEM_CONFIG.vocabulary_size = tokenizer.vocab_size
    print("Vocabulary size: ", GEM_CONFIG.vocabulary_size)

    dataset = load_tokenized_dataset(TRAIN_CONFIG.dataset_name, tokenizer, GEM_CONFIG.sequence_length + 1)

    model = GEM(GEM_CONFIG).to(DEVICE)

    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_parameters}")

    train_dataloader = DataLoader(dataset["train"], batch_size=TRAIN_CONFIG.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset["validation"], batch_size=TRAIN_CONFIG.batch_size)

    optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG.learning_rate, betas=(0.9, 0.99))
    scheduler_steps = len(dataset) * TRAIN_CONFIG.epochs / (TRAIN_CONFIG.batch_size * TRAIN_CONFIG.gradient_accumulation_steps)
    min_learning_rate = TRAIN_CONFIG.learning_rate * 0.01
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_steps, eta_min=min_learning_rate)

    start_epoch = 0
    best_val_loss = float("inf")

    for epoch_num in range(start_epoch, TRAIN_CONFIG.epochs):
        print(f"Epoch {epoch_num + 1}/{TRAIN_CONFIG.epochs}")

        train_loss = epoch(model, train_dataloader, optimizer, scheduler, is_training=True)
        val_loss = epoch(model, val_dataloader, is_training=False)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            save_checkpoint(model, os.path.join(TRAIN_CONFIG.checkpoints_dir, "best.pth"))

        save_checkpoint(model, os.path.join(TRAIN_CONFIG.checkpoints_dir, f"epoch{epoch_num+1}.pth"))


if __name__ == "__main__":
    main()
