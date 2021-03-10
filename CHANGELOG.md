# Changelog

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
