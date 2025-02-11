# New Methods and Extensions of Opacus

This directory contains novel methods built on top of Opacus that enhance DP-SGD. These contributions, made by the community, stem from research demonstrating potential improvements in differentially private model training. By consolidating these methods within the Opacus repository, we facilitate new research and provide a broader array of tools for DP-ML practitioners.


## Contributions
We warmly welcome and encourage contributions of new methods! To contribute, please follow these steps:

1. Fork the repo and create your branch from `main`.
2. Place the new method in a separate subfolder within the `research` directory.
3. The new folder should include a `README.md` that explains the method at a high level, demonstrates usage (e.g., introducing new parameters to the `PrivacyEngine`), and cites relevant sources. The subfolder name should aptly represent the method.
4. Format code using `black`, `flake8`, and `isort` following the instructions under `Code Style` [here](https://github.com/pytorch/opacus/blob/main/CONTRIBUTING.md).
5. Add copyright headers to each `.py` file contributed in the format `# Copyright (c) [copy-right holder]`.

More detailed PR instructions can be found [here](https://github.com/pytorch/opacus/blob/main/CONTRIBUTING.md).

Feel free to reach out with any questions about the process or to discuss whether your method is a good fit for the repository.

## Notes
Please note that the code provided in this directory will not be maintained by the Opacus team, which may lead to compatibility issues with future changes. If you have any questions, please reach out to the PR contributor directly.
