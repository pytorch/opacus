from opacus.optimizers import (
    AdaClipDPOptimizer,
    DistributedDPOptimizer,
    DistributedDPOptimizerFastGradientClipping,
    DistributedPerLayerOptimizer,
    DPOptimizer,
    DPOptimizerFastGradientClipping,
    DPPerLayerOptimizer,
    SimpleDistributedPerLayerOptimizer,
)

from .KFadaclipoptimizer import KF_AdaClipDPOptimizer
from .KFddp_perlayeroptimizer import (
    KF_DistributedPerLayerOptimizer,
    KF_SimpleDistributedPerLayerOptimizer,
)
from .KFddpoptimizer import KF_DistributedDPOptimizer
from .KFddpoptimizer_fast_gradient_clipping import (
    KF_DistributedDPOptimizerFastGradientClipping,
)
from .KFoptimizer import KF_DPOptimizer
from .KFoptimizer_fast_gradient_clipping import KF_DPOptimizerFastGradientClipping
from .KFperlayeroptimizer import KF_DPPerLayerOptimizer


__all__ = [
    "KF_AdaClipDPOptimizer",
    "KF_DistributedPerLayerOptimizer",
    "KF_DistributedDPOptimizer",
    "KF_DPOptimizer",
    "KF_DPOptimizerFastGradientClipping",
    "KF_DistributedDPOptimizerFastGradientlipping",
    "KF_DPPerLayerOptimizer",
    "KF_SimpleDistributedPerLayerOptimizer",
]


def get_optimizer_class(
    clipping: str, distributed: bool, grad_sample_mode: str = None, kalman: bool = False
):
    if kalman:
        if grad_sample_mode == "ghost":
            if clipping == "flat" and distributed is False:
                return KF_DPOptimizerFastGradientClipping
            elif clipping == "flat" and distributed is True:
                return KF_DistributedDPOptimizerFastGradientClipping
            else:
                raise ValueError(
                    f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
                )
        elif clipping == "flat" and distributed is False:
            return KF_DPOptimizer
        elif clipping == "flat" and distributed is True:
            return KF_DistributedDPOptimizer
        elif clipping == "per_layer" and distributed is False:
            return KF_DPPerLayerOptimizer
        elif clipping == "per_layer" and distributed is True:
            if grad_sample_mode == "hooks":
                return KF_DistributedPerLayerOptimizer
            elif grad_sample_mode == "ew":
                return KF_SimpleDistributedPerLayerOptimizer
            else:
                raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
        elif clipping == "adaptive" and distributed is False:
            return KF_AdaClipDPOptimizer
    elif grad_sample_mode == "ghost":
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
    elif clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks":
            return DistributedPerLayerOptimizer
        elif grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
