import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
For debug purposes, we can replace the implementation of the Memory.
At this point in time we do not plan to support multiple types of memory
in the model, as there may be implementation specific optimizations.
"""
def swap_class_implementation(original_class, new_class):
    """
    Replaces the class implementation in all modules that imported it
    """
    original_class_name = original_class.__name__
    debug_class_name = new_class.__name__
    
    for module in sys.modules.values():
        if hasattr(module, original_class_name):
            if getattr(module, original_class_name) is original_class:
                setattr(module, original_class_name, new_class)
    
    print(f"{original_class_name} has been replaced with: {debug_class_name}")

class Memory_Autograd(nn.Module):
    """
    Uses normal autograd in the Memory class.
    Faster training, but consumes O(N) extra memory.
    No hooks and no inplace addition in write.
    """
    def __init__(self, memory_size, K_dim, V_dim):
        super().__init__()
        self.K_initial_state = nn.Parameter(torch.randn(memory_size, K_dim))
        self.V_initial_state = nn.Parameter(torch.randn(memory_size, V_dim))

    def reset(self):
        self.K = self.K_initial_state.clone()
        self.V = self.V_initial_state.clone()

    def write(self, key, value):
        attention = F.softmax((self.K @ key.t()) / (self.K.shape[1] ** 0.5), dim=0)
        old_value = attention.t() @ self.V
        # note that we do not use an inplace operation here
        self.V = self.V + torch.outer(attention, (value - old_value))
        return old_value

    def read(self, query):
        attention = F.softmax((self.K @ query.t()) / (self.K.shape[1] ** 0.5), dim=0)
        retrieved_value = attention.t() @ self.V
        return retrieved_value

    def forward(self, key, value, query):
        self.write(key, value)
        retrieved_value = self.read(query)
        return retrieved_value

class Memory_Transformer(nn.Module):
    """
    Replicates vanilla Transformer key value cache.
    """
    def __init__(self, memory_size, K_dim, V_dim):
        super().__init__()
        self.memory_size = memory_size
        self.K_dim = K_dim
        self.V_dim = V_dim

    def reset(self):
        self.K = torch.empty(0, self.K_dim)
        self.V = torch.empty(0, self.V_dim)
        self.current_size = 0

    def forward(self, q, k, v):
        self.K = torch.cat([self.K, k.unsqueeze(0)], dim=0)
        self.V = torch.cat([self.V, v.unsqueeze(0)], dim=0)
        self.current_size += 1

        attention_scores = torch.matmul(q.unsqueeze(0), self.K[:self.current_size].transpose(-2, -1)) / (self.K_dim ** 0.5)

        mask = torch.ones(1, self.current_size, dtype=torch.bool, device=q.device)
        mask[:, :self.current_size] = False

        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, self.V[:self.current_size]).squeeze(0)

        return output

