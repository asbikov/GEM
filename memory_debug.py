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
    Faster training, but consumes extra memory.
    No hooks and no inplace addition in write.
    """

    def __init__(self, memory_size, dim_K, dim_V):
        super().__init__()
        self.K_initial_state = nn.Parameter(torch.randn(memory_size, dim_K))
        self.V_initial_state = nn.Parameter(torch.randn(memory_size, dim_V))

    def reset(self):
        self.K = self.K_initial_state.clone()
        self.V = self.V_initial_state.clone()

    def forward(self, q, k, v):
        a_k = F.softmax(k @ self.K.t() / self.K.shape[1] ** 0.5, dim=1)
        a_q = F.softmax(q @ self.K.t() / self.K.shape[1] ** 0.5, dim=1)

        a = a_q @ a_k.t()
        mask = torch.triu(torch.ones_like(a)).bool()
        masked_a = a.masked_fill(mask, 0)

        d = (a_k @ self.V - v)

        r = a_q @ self.V - masked_a @ d

        lr = 1
        # note that we can't modify V inplace
        self.V = self.V - lr * (a_k.t() @ d)

        return r


class Memory_Transformer(nn.Module):
    """
    Replicates vanilla Transformer key value cache.
    """

    def __init__(self, memory_size, dim_K, dim_V):
        super().__init__()
        self.memory_size = memory_size
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.dummy_param = nn.Parameter(torch.empty(0))

    def reset(self):
        self.K = torch.zeros(self.memory_size, self.dim_K, device=self.dummy_param.device)
        self.V = torch.zeros(self.memory_size, self.dim_V, device=self.dummy_param.device)
        self.current_size = 0

    def forward(self, q, k, v):
        b = q.shape[0]
        old_size = self.current_size
        new_size = self.current_size + b

        self.K[old_size:new_size, :] = k
        self.V[old_size:new_size, :] = v
        self.current_size = new_size

        a_q = (q @ self.K[:new_size].t()) / (self.dim_K ** 0.5)

        key_indices = torch.arange(new_size, device=q.device).unsqueeze(0)
        query_indices = torch.arange(old_size, new_size, device=q.device).unsqueeze(1)

        mask = key_indices > query_indices
        a_q = a_q.masked_fill(mask, float('-inf'))
        a_q = F.softmax(a_q, dim=-1)

        r = a_q @ self.V[:new_size]

        return r
