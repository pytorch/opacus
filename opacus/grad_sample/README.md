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

## Functorch approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule (force_functorch=True)``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="functorch"`

[functorch](https://pytorch.org/functorch/stable/) is JAX-like composable function transforms for PyTorch.
With functorch we can compute per-sample-gradients efficiently by using function transforms. With the efficient
parallelization provided by `vmap`, we can obtain per-sample gradients for any function function (i.e. any model) by 
doing essentially `vmap(grad(f(x)))`. 

Our experiments show, that `vmap` computations in most cases are as fast as manually written grad samplers used in 
hooks-based approach.

With the current implementation `GradSampleModule` will use manual grad samplers for known modules (i.e. maintain the
old behaviour for all previously supported models) and will only use functorch for unknown modules.

With `force_functorch=True` passed to the constructor `GradSampleModule` will rely exclusively on functorch. 

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


| xxx                          | Hooks                           | Expanded Weights | Functorch    |
|:----------------------------:|:-------------------------------:|:----------------:|:------------:| 
| Required PyTorch version     | 1.8+                            | 1.13+            | 1.12         |
| Development status           | Underlying mechanism deprecated | Beta             | Beta         | 
| Runtime Performance          | 1x                              | ✅ ~30% faster  | 🟨 50%-300% slower |
| Any DP-allowed† layers       | Not supported                   | Not supported    | ✅ Supported |
| Most popular nn.* layers     | ✅ Supported                    | ✅ Supported    | ✅ Supported  | 
| torchscript models           | Not supported                   | ✅ Supported    | ❓            |
| Client-provided grad sampler | ✅ Supported                    | Not supported   | ✅ Not needed |
| `batch_first=False`          | ✅ Supported                    | Not supported   | ❓            |
| Recurrent networks           | ✅ Supported                    | Not supported   | ❓            |
| Padding `same` in Conv       | ✅ Supported                    | Not supported   | ✅ Supported  |

† Layers that produce joint computations on batch samples (e.g. BatchNorm) are not allowed under any approach    

### Benchmark results

Numbers indicated are ratio compared to a non-private baseline. Larger number indicates worse performance 
(Longer runtime (in seconds) or higher memory footprint)

#### Runtime
| Layer | batch size | Hooks | ExpandedWeights | Functorch |
|:-----:|:----------:|:-----:|:---------------:|:---------:|
| nn.Linear | 16 | 1.91 | 1.53 | 2.80
| nn.Linear | 32 | 2.15 | 1.50 | 3.07
| nn.Linear | 64 | 2.88 | 2.15 | 3.44
| nn.Linear | 128 | 4.51 | 3.11 | 4.41
| nn.Conv2d | 16 | 1.76 | 4.61 (?) 
| nn.Conv2d | 32 | 2.21 | 2.01 |
| nn.Conv2d | 64 | 2.64 | 
| nn.Conv2d | 128 | 2.89