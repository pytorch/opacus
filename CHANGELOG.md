# Changelog

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
