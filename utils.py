import os
import torch
from model import GEM


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config
    }
    torch.save(checkpoint, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = GEM(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
