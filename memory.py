import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    """
    A differentiable memory module that provides read and write operations.
    The memory consists of keys (K) and values (V) that can be updated and queried.
    Normal autograd would consume a lot of memory to keep the intermediate memory states.
    Instead, we reproduce thе intermediate states step by step in the backward pass. 
    """

    def __init__(self, memory_size, K_dim, V_dim):
        super().__init__()
        self.K_initial_state = nn.Parameter(torch.randn(memory_size, K_dim))
        self.V_initial_state = nn.Parameter(torch.randn(memory_size, V_dim))

    def reset(self):
        # model parameters cannot be modified in the
        # forward pass, we have to clone them
        self.K = self.K_initial_state.clone()
        self.V = self.V_initial_state.clone()

    def write(self, key, value):
        # V is adjusted in the direction of the
        # gradient towards the new value.
        # Note that writing the old value back
        # would reverse this operation.
        attention = F.softmax(self.K @ key.t() / self.K.shape[1] ** 0.5, dim=0)
        old_value = attention.t() @ self.V
        self.V += torch.outer(attention, (value - old_value))
        return old_value

    def read(self, query):
        # standard retrieval
        attention = F.softmax(self.K @ query.t() / self.K.shape[1] ** 0.5, dim=0)
        retrieved_value = attention.t() @ self.V
        return retrieved_value

    def pack_hook(self, x):
        if x is self.K:
            return "K"
        if x is self.V:
            return "V"
        return x

    def unpack_hook(self, x):
        if x == "K":
            return self.K
        if x == "V":
            return self.V
        return x

    def forward(self, key, value, query):
        # use pack/unpack hooks to prevent autograd from
        # storing intermediate memory states
        with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
            old_value = self.write(key, value)
            retrieved_value = self.read(query)

        if self.training:
            # cache the key and the old value
            with torch.no_grad():
                cached_key = key.clone()
                cached_value = old_value.clone()

            # by design, writing the old value for this key
            # restores the memory to its previous state
            def backward_hook(_):
                with torch.no_grad():
                    self.write(cached_key, cached_value)
            retrieved_value.register_hook(backward_hook)
                
        return retrieved_value
