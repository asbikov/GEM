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
        # used to get the device, is there a better way?
        self.dummy_param = nn.Parameter(torch.empty(0))

    def reset(self):
        self.K = torch.empty(0, self.dim_K, device=self.dummy_param.device)
        self.V = torch.empty(0, self.dim_V, device=self.dummy_param.device)
        self.current_size = 0

    def forward(self, q, k, v):
        b = q.shape[0]

        self.K = torch.cat([self.K[:self.current_size], k], dim=0)
        self.V = torch.cat([self.V[:self.current_size], v], dim=0)

        a_q = q @ self.K.t() / self.dim_K ** 0.5

        past_mask = torch.zeros(b, self.current_size, dtype=torch.bool, device=self.dummy_param.device)
        causal_mask = torch.triu(torch.ones(b, b, dtype=torch.bool, device=self.dummy_param.device), diagonal=1)
        mask = torch.cat([past_mask, causal_mask], dim=1)

        masked_a = a_q.masked_fill(mask, float('-inf'))
        a = F.softmax(masked_a, dim=-1)

        r = a @ self.V

        self.current_size += b

        return r
