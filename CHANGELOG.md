# Changelog

## v1.1.2
### Bug fixes
* Support tied parameters (#417)
* Fix callsite sensitiveness of `zero_grad()` (#422, #423)
* Improve microbenchmark argument parsing and tests (#425)
* Fix opacus nn.functional import (#426) 
### Miscellaneous
* Add microbenchmarks (#412, #416)
* Add more badges to readme (#424) 

## v1.1.1
### Bug fixes
* Fix accountant when using number of steps instead of epochs
* Add params check when converting BatchNorm to GroupNorm (#390)
* Fix typo in gdp accountant mechansim name (#386)
* Fix linter errors (#392)
* Add friendly and detailed message for unsupported layers (#401)
* Run linter on nightly workflow (#399)
* Add warning for Gaussian DP accounting (#400)
* Clone replacement modules on the same device as original (#356)
* Implementing 3D dilation (#408)
* fix(batch_memory_manager): Ensures split_idxs use native python types (#410)
### Miscellaneous
* Migrate nightly CircleCI flows to scheduled pipelines (#402)
* Migrate from ubuntu 16.04 to 20.04 on CircleCI (#403)

## v1.1.0
### New Feature
* Add support for GDP accounting in get_noise_multiplier (#303)
### Bug fixes
* Conservative search for target epsilon in get_noise_multiplier (#348)
* Warn and ignore "drop_last" when set in DPDataLoader (#357)
* Fix per-layer clipping in distributed (#347)
### Miscellaneous
* Update code of conduct and file headers
* Add "Support Ukraine" banner to opacus website homepage
* Lint fixes

## v1.0.2
### Bug fixes
* DPOptimizer
  * Passes through `.defaults` field to match pytorch Optimizer (#329)
  * Better exception message in `.step()` when p.grad_sample=None (#331)
  * Correct `closure` call after applying DP noise (#330)
* Proper gradient scaling in DDP mode
* Corrections of typos and errors in tutorials
### Miscellaneous
* Opacus can be installed with conda: added recipe in conda-forge (#326)
* Formatting change in accordance with black-22.1.0

## v1.0.1
### Bug fixes
* Hidden states of RNN is passed to device (#314)
* Validate and fix trainable modules only (#316)
### Miscellaneous
* Minor corrections and typo fixes in links, documentation, and tutorials.

## v1.0.0
* This release packs in lot of new features and bug fixes, and most importantly, also brings forth new APIs that are simpler, more modular, and easily extensible.
* We have bumped up the major version number from 0 to 1 and have introduced breaking changes. However, the major version bump also indicates a step-function upgrade in the capabilities.
* See [Release notes](https://github.com/pytorch/opacus/releases/tag/v1.0.0] and [Migration Guide](https://github.com/pytorch/opacus/blob/main/Migration_Guide.md) for more details about the changes.
* PR #273 contains the pointers to all the commits and PRs that went into this release.

## v0.15.0
### New Features
* DDP support for faster distributed training (#196)
* Support of GRU and RNN; refactored LSTM implementation (#222)
* PyTorch Lightning Demo (#244)
### Bug fixes
* Improve nn.Linear grad sampler memory consumption (#192)
* Update Opacus to stop using deprecated torch.set_deterministic (#197)
* Fix optimizer.step after engine.detach()
* Test fixes
### Miscellaneous
* Better validation error reporting (#199)
* grad sampler type checking (#241)

## v0.14.0
### New features
* Major refactoring - per-sample gradient computation is separated into its own module - GradSampleModule (#175)
* Improved RDP to (eps, delta)-DP conversion (#162)
* Multi-GPU support (#166)
### Bug fixes
* Handle empty batches in Poisson sampling (#164)
* Fixed memory leak from no_grad execution (#180)

## v0.13.0
### New features
* PackedSequence support for DPLSTM (#150) (thanks @touqir14 !)
### Miscellaneous
* Pytest moved to dev installation (#144)

## v0.12.0
This version introduces a **mildly-breaking change**: the privacy engine will now support sampling with variable batch size, just like in the Abadi et al. paper. To accommodate this feature, we have made `batch_size` a kwarg (no longer positional). We are also enforcing that all kwargs must not be specified positionally. If you had code that passed kwargs positionally, you will find an error (which will be very simple to fix).
### New features
* Enforce kwargs to Privacy Engine (#136).
* Fix batch construction and privacy engine (#128). (thanks @ConstanceBeguier!)
* Compute required sigma to reach (epsilon, delta) budget (#126)
* Friendly user message for unused parameters (#118).
* Print helpful message when models are not in train mode (#113)
### Bug fixes
* Now the Opacus package has a `__version__` attribute.
* Fix immer security issue, fix website errors
* Updated setup.py version requirements to support 3.6.8 for Windows (#108) (thanks @madhavajay!)
### Miscellaneous
* Rewrote the grad_sample tests to use Hypothesis (#125). (thanks @touqir14!)

## v0.11.0
### New features
* Extend DPLSTM to support multilayer, dropout (#101)
* Modifications to Char LSTM name classification example
* Introduce issue templates for GitHub (#102)
* Added support for Conv3D layers
### Bug fixes
* Linter fixes for Conv3D (#105)
### Miscellaneous
* Make TorchCSPRNG an optional dependency (#106)
* Removed unnecessary calls to zero_grad from examples and tutorials (#96)

## v0.10.1
### Bug fixes
* Fix PyPI deployment (#91).
### Miscellaneous
* Refactor grad sample tests (#90).
* Avoid storing activations in certain scenarios (#87)

## v0.10.0
### New features
* Reimplemented the Embedding layer, making it 9x faster with lower memory footprint (#73).
* Reimplemented the DPLSTM layer, making it 2x faster with lower memory footprint.
* Extended our Conv support to grouped convolutions (#78).
### Bug fixes
* Small fixes to clipping logic (#45).
### Miscellaneous
* Changed docstring style from numpy -> Google.
* Throw an error if sample rate > 1 in privacy engine.
* Migrated our IMDB example from TorchText -> HuggingFace (#85).
* Added PRNG shuffling to our examples.

## v0.9.1
### Bug fixes
* Compatibility with Python 3.6 (Minimum required version changed from 3.7 to 3.6.9).
* Allow DP-LSTM to have null init.

## v0.9.0
### New Features
* Initial commit.
