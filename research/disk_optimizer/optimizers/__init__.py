# Copyright (c) Xinwei Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def get_optimizer_key(clipping: str, distributed: bool, grad_sample_mode: str = None):
    key = clipping + "_" + str(distributed)
    if grad_sample_mode == "ghost" or (clipping == "per_layer" and distributed):
        key += "_" + grad_sample_mode
    return key


def get_optimizer_class(
    clipping: str, distributed: bool, grad_sample_mode: str = None, kalman: bool = False
):
    if kalman:
        optimizer_dict = {
            "flat_false_ghost": KF_DPOptimizerFastGradientClipping,
            "flat_true_ghost": KF_DistributedDPOptimizerFastGradientClipping,
            "flat_false": KF_DPOptimizer,
            "flat_true": KF_DistributedDPOptimizer,
            "per_layer_false": KF_DPPerLayerOptimizer,
            "per_layer_true_hook": KF_DistributedPerLayerOptimizer,
            "per_layer_true_ew": KF_SimpleDistributedPerLayerOptimizer,
            "adaptive_false": KF_AdaClipDPOptimizer,
        }
    else:
        optimizer_dict = {
            "flat_false_ghost": DPOptimizerFastGradientClipping,
            "flat_true_ghost": DistributedDPOptimizerFastGradientClipping,
            "flat_false": DPOptimizer,
            "flat_true": DistributedDPOptimizer,
            "per_layer_false": DPPerLayerOptimizer,
            "per_layer_true_hook": DistributedPerLayerOptimizer,
            "per_layer_true_ew": SimpleDistributedPerLayerOptimizer,
            "adaptive_false": AdaClipDPOptimizer,
        }
    optimizer_key = get_optimizer_key(clipping, distributed, grad_sample_mode)
    if optimizer_key not in optimizer_dict:
        err_str = "Unsupported combination of parameters."
        err_str += f"Clipping: {clipping}, distributed: {str(distributed)} and grad_sample_mode: {grad_sample_mode}"
        raise ValueError(err_str)
    else:
        return optimizer_dict[optimizer_key]

    #     if grad_sample_mode == "ghost":
    #         if clipping == "flat" and distributed is False:
    #             return KF_DPOptimizerFastGradientClipping
    #         elif clipping == "flat" and distributed is True:
    #             return KF_DistributedDPOptimizerFastGradientClipping
    #         else:
    #
    #             raise ValueError(err_str)
    #     elif clipping == "flat" and distributed is False:
    #         return KF_DPOptimizer
    #     elif clipping == "flat" and distributed is True:
    #         return KF_DistributedDPOptimizer
    #     elif clipping == "per_layer" and distributed is False:
    #         return KF_DPPerLayerOptimizer
    #     elif clipping == "per_layer" and distributed is True:
    #         if grad_sample_mode == "hooks":
    #             return KF_DistributedPerLayerOptimizer
    #         elif grad_sample_mode == "ew":
    #             return KF_SimpleDistributedPerLayerOptimizer
    #         else:
    #             raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    #     elif clipping == "adaptive" and distributed is False:
    #         return KF_AdaClipDPOptimizer
    # elif grad_sample_mode == "ghost":
    #     if clipping == "flat" and distributed is False:
    #         return DPOptimizerFastGradientClipping
    #     elif clipping == "flat" and distributed is True:
    #         return DistributedDPOptimizerFastGradientClipping
    #     else:
    #         err_str = "Unsupported combination of parameters."
    #         err_str+= f"Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
    #         raise ValueError(
    #             err_str
    #         )
    # elif clipping == "flat" and distributed is False:
    #     return DPOptimizer
    # elif clipping == "flat" and distributed is True:
    #     return DistributedDPOptimizer
    # elif clipping == "per_layer" and distributed is False:
    #     return DPPerLayerOptimizer
    # elif clipping == "per_layer" and distributed is True:
    #     if grad_sample_mode == "hooks":
    #         return DistributedPerLayerOptimizer
    #     elif grad_sample_mode == "ew":
    #         return SimpleDistributedPerLayerOptimizer
    #     else:
    #         raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    # elif clipping == "adaptive" and distributed is False:
    #     return AdaClipDPOptimizer
    # raise ValueError(
    #     f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    # )
