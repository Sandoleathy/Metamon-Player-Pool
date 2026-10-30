"""Local workarounds for running metamon inside the DSH Windows sandbox.

Issue: on this machine Python 3.14 maps ``os.mkdir(path, 0o700)`` to a restrictive
DACL.  ``tempfile.mkdtemp`` (and therefore ``huggingface_hub``, which probes
symlink support with a temporary directory) then creates directories that the
current process cannot write into, so model downloads fail with
``PermissionError``.

Fix: force 0o777 for 0o700-ish mkdir calls.  This only relaxes permissions inside
the user's own workspace, so it is safe for this local research setup.
"""
from __future__ import annotations

import os
import sys

_PATCHED = False


def patch_mkdir_permissions() -> None:
    """Make ``os.mkdir`` ignore restrictive 0o700 modes on Windows."""
    global _PATCHED
    if _PATCHED or sys.platform != "win32":
        return
    original_mkdir = os.mkdir

    def mkdir(path, mode=0o777, *, dir_fd=None):
        if (mode & 0o777) == 0o700:
            mode = 0o777
        return original_mkdir(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir
    _PATCHED = True


def patch_torch_dynamo() -> None:
    """Disable ``torch.compile`` only where Inductor cannot work.

    amago decorates several networks with ``torch.compile``.  On this Windows box
    Inductor needs a C++ toolchain and piped compiler subprocesses, which are
    unavailable, so the first forward pass would crash.  On a Linux/CUDA machine
    compilation works and is worth keeping for the 142M policies, so the patch is
    a no-op there.

    Override with ``PLAYER_POOL_TORCH_COMPILE=0`` (force eager) or ``=1`` (force
    compilation, useful to test on a machine that has a compiler).
    """
    choice = os.environ.get("PLAYER_POOL_TORCH_COMPILE")
    try:
        import torch

        if choice is None:
            disable = sys.platform == "win32" or not torch.cuda.is_available()
        else:
            disable = choice.strip().lower() in {"0", "false", "no", "off"}
    except Exception:  # torch unavailable: nothing to patch
        return

    if not disable:
        return
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
    except Exception:  # private API moved
        pass


def patch_flash_attention() -> None:
    """Use a plain-PyTorch fallback when native flash-attn 2 is unusable.

    Eight of the ten Gen9OU policies use ``amago.nets.transformer.FlashAttention``
    for their trajectory encoder.  flash-attn 2 is CUDA-only, so on CPU the
    constructor asserts.  The module shares weights with ``SelfAttention`` (same
    qkv/proj parameters), so we keep the gin-configured ``window_size`` and swap
    only the forward pass for a numerically equivalent implementation that
    supports the causal + sliding-window mask and the KV cache.

    The fallback is only installed when native flash-attn cannot run here (not
    installed, or no CUDA device), so a CUDA box with flash-attn 2 installed uses
    the upstream fast path unchanged.  Override with
    ``PLAYER_POOL_FLASH_ATTENTION=native|cpu`` (``auto`` is the default).
    """
    import math

    import torch
    from amago.nets import transformer

    mode = os.environ.get("PLAYER_POOL_FLASH_ATTENTION", "auto").strip().lower()
    if mode == "native":
        return
    if mode == "auto" and transformer.flash_attn is not None and torch.cuda.is_available():
        return
    if getattr(transformer.FlashAttention, "_metamon_cpu_patched", False):
        return

    # the constructor only checks ``flash_attn is not None`` before calling
    # SelfAttention.__init__; a truthy placeholder lets the gin-configured
    # window_size through untouched.
    if transformer.flash_attn is None:
        transformer.flash_attn = _TruthyPlaceholder()

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        q, k, v = torch.unbind(qkv, dim=2)
        scale = 1.0 / math.sqrt(q.shape[-1])
        window_left, _window_right = getattr(self, "window_size", (-1, -1))

        if key_cache is None or val_cache is None or cache_seqlens is None:
            scores = torch.einsum("blhe,bshe->bhls", q.float(), k.float()) * scale
            idx = torch.arange(q.shape[1], device=q.device)
            mask = torch.zeros((q.shape[1], q.shape[1]), dtype=torch.bool, device=q.device)
            if self.causal:
                mask |= idx[None, :] > idx[:, None]
            if window_left is not None and window_left > 0:
                mask |= idx[None, :] < (idx[:, None] - window_left)
            scores.masked_fill_(mask[None, None], -torch.inf)
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nn.functional.dropout(
                attn, p=self.dropout if self.training else 0.0
            )
            return torch.einsum("bhls,bshd->blhd", attn, v.float())

        # incremental inference with a pre-allocated KV cache
        cache_idxs = torch.arange(key_cache.shape[0], device=key_cache.device)
        key_cache[cache_idxs, cache_seqlens] = k[:, 0].to(key_cache.dtype)
        val_cache[cache_idxs, cache_seqlens] = v[:, 0].to(val_cache.dtype)
        end = cache_seqlens + 1
        max_len = int(end.max())
        k_cache = torch.nan_to_num(key_cache[:, :max_len]).float()
        v_cache = torch.nan_to_num(val_cache[:, :max_len]).float()
        scores = scale * torch.einsum("blhe,blhe->blh", q.float(), k_cache)
        pos = (end - 1)[:, None]
        kidx = torch.arange(max_len, device=q.device)[None, :]
        mask = kidx > pos
        if window_left is not None and window_left > 0:
            mask |= kidx < (pos - window_left)
        scores = scores.masked_fill(mask[:, :, None], -torch.inf)
        attn = torch.softmax(scores, dim=1)
        attn = torch.nn.functional.dropout(
            attn, p=self.dropout if self.training else 0.0
        )
        return torch.einsum("blh,blhd->bhd", attn, v_cache).unsqueeze(1)

    transformer.FlashAttention.forward = forward
    transformer.FlashAttention._metamon_cpu_patched = True


class _TruthyPlaceholder:
    """Stands in for the ``flash_attn`` module in a constructor assertion."""


def patch_amago_traj_length() -> None:
    """Tolerate amago's ``traj_save_len: int = 1e10`` float default.

    ``SequenceWrapper.random_traj_length`` feeds that default straight into
    ``random.randint``, which only accepts ints, so ``Experiment.start()`` fails
    while building training envs.  Rounding keeps the "effectively infinite"
    behaviour and changes nothing for evaluation.
    """
    import random

    from amago.envs.amago_env import SequenceWrapper

    if getattr(SequenceWrapper, "_metamon_patched", False):
        return
    original = SequenceWrapper.random_traj_length

    def random_traj_length(self):
        if not self.save_every:
            return None
        low, high = self.save_every
        return random.randint(int(low), int(high))

    SequenceWrapper.random_traj_length = random_traj_length
    SequenceWrapper._metamon_patched = True
