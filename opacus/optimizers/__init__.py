from .ddp_perlayeroptimizer import DistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .optimizer import DPOptimizer
from .perlayeroptimizer import DPPerLayerOptimizer


__all__ = [
    "DistributedPerLayerOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPPerLayerOptimizer",
]


def get_optimizer_class(clipping: str, distributed: bool):
    if clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        return DistributedPerLayerOptimizer

    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
