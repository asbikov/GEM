import torch
import torch.nn as nn
from dataclasses import dataclass
from memory import Memory

@dataclass
class GEMConfig:
    """
    """
    vocabulary_size: int
    sequence_length: int
    minibatch_size: int
    memory_size: int
    embedding_size: int
    n_layers: int

class MLP(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.non_l = nn.GELU()
        self.l2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.non_l(x)
        x = self.l2(x)
        return x

class GEMBlock(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory = Memory(config.memory_size, config.embedding_size, config.embedding_size)
        self.K = nn.Linear(config.embedding_size, config.embedding_size)
        self.Q = nn.Linear(config.embedding_size, config.embedding_size)
        self.V = nn.Linear(config.embedding_size, config.embedding_size)
        self.proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.MLP = MLP(config.embedding_size)
        self.norm1 = nn.LayerNorm(config.embedding_size, bias=False)
        self.norm2 = nn.LayerNorm(config.embedding_size, bias=False)
    
    def forward(self, x):
        xn = self.norm1(x)
        q, k, v = self.Q(xn), self.K(xn), self.V(xn)
        r = self.memory(q, k, v)
        p = self.proj(r)
        x = x + p
        xn = self.norm2(x)
        p = self.MLP(xn)
        x = x + p
        return x

class GEM(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GEMBlock(config) for _ in range(config.n_layers)])
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.embedding_size)
        self.pos_embeddings = nn.Embedding(config.sequence_length, config.embedding_size)
        self.norm = nn.LayerNorm(config.embedding_size, bias=False)
        self.head = nn.Linear(config.embedding_size, config.vocabulary_size, bias=False)
        
        # TODO: weight tying for last layer
        #self.token_embeddings.weight = self.head.weight

    def forward(self, inputs):
        batch_size, sequence_length = inputs.shape

        if batch_size != 1:
            raise ValueError("Currently, only batch size 1 is supported.")

        token_embeddings = self.token_embeddings(inputs)
        pos_embeddings = self.pos_embeddings(torch.arange(sequence_length, device=inputs.device))
        embeddings = token_embeddings + pos_embeddings

        # Remove batch dimension since batch_size is 1
        embeddings = embeddings[0]

        # Reset memory in each layer
        for layer in self.layers:
            layer.memory.reset()

        outputs_list = []
        minibatch_size = self.config.minibatch_size

        # Process tokens in minibatches
        for i in range(0, sequence_length, minibatch_size):
            end = min(i + minibatch_size, sequence_length)
            minibatch = embeddings[i:end]

            for layer in self.layers:
                minibatch = layer(minibatch)

            logits = self.head(self.norm(minibatch))
            outputs_list.append(logits)

        outputs = torch.cat(outputs_list, dim=0).unsqueeze(0)
        return outputs