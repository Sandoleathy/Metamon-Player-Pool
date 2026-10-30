"""Verify that ``player_pool.compat`` selects the right implementation.

Run one case per process (PyTorch cannot be re-imported after ``sys.modules`` is
cleared, so each case needs a fresh interpreter):

    python -m player_pool.check_compat auto_cuda_with_flash
    python -m player_pool.check_compat auto_cuda_without_flash
    python -m player_pool.check_compat forced_native
    python -m player_pool.check_compat force_compile
    python -m player_pool.check_compat force_eager
    python -m player_pool.check_compat default_cpu
"""
from __future__ import annotations

import os
import sys

CASES = [
    "auto_cuda_with_flash",
    "auto_cuda_without_flash",
    "forced_native",
    "force_compile",
    "force_eager",
    "default_cpu",
]


def _auto_cuda_with_flash() -> None:
    os.environ.pop("PLAYER_POOL_FLASH_ATTENTION", None)
    import torch

    torch.cuda.is_available = lambda: True
    from amago.nets import transformer

    transformer.flash_attn = object()
    before = transformer.FlashAttention.forward
    from player_pool.compat import patch_flash_attention

    patch_flash_attention()
    assert transformer.FlashAttention.forward is before
    assert not getattr(transformer.FlashAttention, "_metamon_cpu_patched", False)
    print("auto + cuda + flash-attn -> native path kept OK")


def _auto_cuda_without_flash() -> None:
    os.environ.pop("PLAYER_POOL_FLASH_ATTENTION", None)
    import torch

    torch.cuda.is_available = lambda: True
    from amago.nets import transformer

    transformer.flash_attn = None
    from player_pool.compat import patch_flash_attention

    patch_flash_attention()
    assert getattr(transformer.FlashAttention, "_metamon_cpu_patched", False)
    print("auto + cuda + no flash-attn -> fallback installed OK")


def _forced_native() -> None:
    os.environ["PLAYER_POOL_FLASH_ATTENTION"] = "native"
    from amago.nets import transformer

    before = transformer.FlashAttention.forward
    from player_pool.compat import patch_flash_attention

    patch_flash_attention()
    assert transformer.FlashAttention.forward is before
    print("forced native -> no patch OK")


def _force_compile() -> None:
    os.environ["PLAYER_POOL_TORCH_COMPILE"] = "1"
    import torch

    from player_pool.compat import patch_torch_dynamo

    patch_torch_dynamo()
    assert os.environ.get("TORCHDYNAMO_DISABLE") is None
    assert torch._dynamo.config.disable is False
    print("PLAYER_POOL_TORCH_COMPILE=1 -> dynamo enabled OK")


def _force_eager() -> None:
    os.environ["PLAYER_POOL_TORCH_COMPILE"] = "0"
    import torch

    from player_pool.compat import patch_torch_dynamo

    patch_torch_dynamo()
    assert torch._dynamo.config.disable is True
    print("PLAYER_POOL_TORCH_COMPILE=0 -> dynamo disabled OK")


def _default_cpu() -> None:
    os.environ.pop("PLAYER_POOL_TORCH_COMPILE", None)
    import torch

    from player_pool.compat import patch_torch_dynamo

    patch_torch_dynamo()
    assert torch._dynamo.config.disable is True
    print("default on CPU/Windows -> dynamo disabled OK")


DISPATCH = {
    "auto_cuda_with_flash": _auto_cuda_with_flash,
    "auto_cuda_without_flash": _auto_cuda_without_flash,
    "forced_native": _forced_native,
    "force_compile": _force_compile,
    "force_eager": _force_eager,
    "default_cpu": _default_cpu,
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in DISPATCH:
        print(__doc__)
        print("cases:", ", ".join(CASES))
        return 2
    DISPATCH[argv[0]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
