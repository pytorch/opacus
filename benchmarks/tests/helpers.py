from typing import Optional

import pytest
import torch


skipifnocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA."
)


def get_n_byte_tensor(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns a torch.int8 tensor of size n.

    Args:
        n: size of the tensor to allocate
        device: torch.device to allocate tensor on

    Returns:
        torch.int8 tensor of size n on device

    Notes: Though 1 int8 = 1 byte, memory is allocated in blocks, such that the size
    of the tensor in bytes >= n.
    """
    return torch.zeros(n, dtype=torch.int8, device=device)


def get_actual_memory_allocated(n: int, device: torch.device) -> int:
    """
    Returns the CUDA memory allocated for a torch.int8 tensor of size n.

    Args:
        n: size of the tensor to get allocated memory for
        device: torch.device to allocate tensor on

    Returns:
        Number of bytes of CUDA memory allocated for a torch.int8 tensor of size n.

    Notes: Should only be called on CUDA devices. Resets CUDA memory statistics.
    """
    assert device.type == "cuda"
    prev_memory_allocated = torch.cuda.memory_allocated(device)
    tensor = get_n_byte_tensor(n, device=device)
    memory_allocated = torch.cuda.memory_allocated(device)
    del tensor
    torch.cuda.reset_peak_memory_stats(device)
    assert prev_memory_allocated == torch.cuda.memory_allocated(device)
    return memory_allocated - prev_memory_allocated
