"""Forward hooks on decoder layers that capture last-token hidden states.

Hooks are registered once and switched on or off per forward pass.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class HookManager:
    """Manages forward hooks on transformer layers for hidden state extraction."""

    def __init__(self, model, layers: list[int], layer_accessor: str = "model.model.layers"):
        self._storage: dict[int, torch.Tensor] = {}
        self._active = False
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # Navigate to layer list
        layer_list = model
        for attr in layer_accessor.split("."):
            layer_list = getattr(layer_list, attr)

        for li in layers:
            def make_hook(layer_idx: int):
                def hook_fn(module, inp, out):
                    if not self._active:
                        return
                    hs = out[0] if isinstance(out, tuple) else out
                    # Last token hidden state
                    self._storage[layer_idx] = hs[0, -1, :].detach().float().cpu()
                return hook_fn

            self._hooks.append(layer_list[li].register_forward_hook(make_hook(li)))

    @contextmanager
    def extract_step(self):
        """Context manager: activates extraction for one forward pass."""
        self._active = True
        self._storage.clear()
        try:
            yield
        finally:
            self._active = False

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        """Return collected hidden states and clear storage."""
        result = dict(self._storage)
        self._storage.clear()
        return result

    def remove_all(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_all()
