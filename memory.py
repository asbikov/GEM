import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """
    A differentiable memory module.
    The memory consists of a keys matrix (K) and a values matrix (V) which can be updated.
    Normal autograd would consume a lot of memory to keep the intermediate memory states.
    Instead, we reproduce the intermediate states step by step in the backward pass. 
    """

    def __init__(self, memory_size, dim_K, dim_V):
        """
        """
        super().__init__()
        self.K_initial_state = nn.Parameter(torch.randn(memory_size, dim_K))
        self.V_initial_state = nn.Parameter(torch.randn(memory_size, dim_V))

    def reset(self):
        """
        Model parameters cannot be modified in the forward pass.
        We have to clone them.
        """
        self.K = self.K_initial_state.clone()
        self.V = self.V_initial_state.clone()

    def pack_hook(self, x):
        """
        Prevent autograd from creating additional copies of the memory state.
        """
        if x is self.K:
            return "K"
        if x is self.V:
            return "V"
        return x

    def unpack_hook(self, x):
        """
        Return the unique memory state.
        """
        if x == "K":
            return self.K
        if x == "V":
            return self.V
        return x

    def forward(self, q, k, v):
        """
        We can think of it as writing a (k, v) pair to the memory, and then retrieving q.
        Inputs and outputs are mini-batches of size b. See 'dual form' from the TTT paper.

        Args:
            q: (b, dim_K)
            k: (b, dim_K)
            v: (b, dim_V)

        Returns:
            r: (b, dim_V)
        """
        with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
            a_k = F.softmax(k @ self.K.t() / self.K.shape[1] ** 0.5, dim=1)
            a_q = F.softmax(q @ self.K.t() / self.K.shape[1] ** 0.5, dim=1)

            a = a_q @ a_k.t()
            mask = torch.triu(torch.ones_like(a)).bool()
            masked_a = a.masked_fill(mask, 0)

            d = (a_k @ self.V - v)

            r = a_q @ self.V - masked_a @ d

            lr = 1
            self.V -= lr * (a_k.t() @ d)

        if self.training:
            def backward_hook(_):
                # restore the memory to its previous state
                # we are reusing k and d, shapes (b, dim_K) and (b, dim_V)
                with torch.no_grad():
                    a_k = F.softmax(k @ self.K.t() / self.K.shape[1] ** 0.5, dim=1)
                    self.V += lr * a_k.t() @ d

            r.register_hook(backward_hook)

        return r
