# Grad Samples

Computing per sample gradients is an integral part of Opacus framework. We strive to provide out-of-the-box support for
wide range of models, while keeping computations efficient.

We currently provide two independent approaches for computing per sample gradients: hooks-based ``GradSampleModule`` 
(stable implementation, exists since the very first version of Opacus) and ``GradSampleModuleExpandedWeights``
(based on a beta functionality available in PyTorch 1.12).

Each of the two implementations comes with it's own set of limitations, and we leave the choice up to the client 
which one to use. 

``GradSampleModuleExpandedWeights`` is currently in early beta and can produce unexpected errors, but potentially
improves upon ``GradSampleModule`` on performance and functionality.

**TL;DR:** If you want stable implementation, use ``GradSampleModule`` (`grad_sample_mode="hooks"`). 
If you want to experiment with the new functionality - try ``GradSampleModuleExpandedWeights``(`grad_sample_mode="ew"`) 
and switch back to ``GradSampleModule`` if you encounter strange errors or unexpexted behaviour. 
We'd also appreciate it if you report these to us

## Hooks-based approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="hooks"`

Computes per-sample gradients for a model using backward hooks. It requires custom grad sampler methods for every 
trainable layer in the model. We provide such methods for most popular PyTorch layers. Additionally, client can
provide their own grad sampler for any new unsupported layer (see [tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb))

## ExpandedWeigths approach
- Model wrapping class: ``opacus.grad_sample.gsm_exp_weights.GradSampleModuleExpandedWeights``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="ew"`

Computes per-sample gradients for a model using core functionality available in PyTorch 1.12+. Unlike hooks-based
grad sampler, which works on a module level, ExpandedWeights work on the function level, i.e. if your layer is not
explicitly supported, but only uses known operations, ExpandedWeights will support it out of the box. 

At the time of writing, the coverage for custom grad samplers between ``GradSampleModule`` and ``GradSampleModuleExpandedWeights``
is roughly the same.

## Comparative analysis

Please note that these are known limitations and we plan to improve Expanded Weights and bridge the gap in feature completeness


| xxx | Hooks | Expanded Weights |
|:-----:|:-------:|:------------------:|
| Required PyTorch version | 1.8+ | 1.12+ |
| Development status | Underlying mechanism deprecated | Beta |
| Performance | - | ✅ Likely up to 2.5x faster |
| torchscript models | Not supported | ✅ Supported |
| Client-provided grad sampler | ✅ Supported | Not supported |
| `batch_first=False` | ✅ Supported | Not supported |
| Most popular nn.* layers | ✅ Supported | ✅ Supported |
| Recurrent networks | ✅ Supported | Not supported |
| nn.LayerNorm | ✅ Supported | ⚠️ Unstable |
| nn.Conv3d | ✅ Supported | Not supported |