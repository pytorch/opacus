# Changelog

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
