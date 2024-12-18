# New Methods and Extensions of Opacus

This directory contains novel methods built on top of Opacus that enhance DP-SGD. These contributions, made by the community, stem from research demonstrating potential improvements in differentially private model training. By consolidating these methods within the Opacus repository, we facilitate new research and provide a broader array of tools for DP-ML practitioners.


## Contributions
We warmly welcome and encourage contributions of new methods! To contribute, please follow these steps:

1. Fork this repository on GitHub.
2. Add your new method locally, placing it in a separate subfolder within the `research` directory.
3. The new folder should include a `README.md` that explains the method at a high level, demonstrates how to use the method (e.g., introducing new parameters to the `PrivacyEngine`), and cites relevant sources. The subfolder name should aptly represent the method.
4. Push your local changes to your fork and then create a Pull Request to the `main` branch. Your Pull Request should pass all our tests. We will review it, request any necessary changes, and then merge it into our repository.
Feel free to reach out with any questions about the process or to discuss whether your method is a good fit for the repository.

## Notes
Please note that the code provided in this directory will not be maintained by the Opacus team, which may lead to compatibility issues with future changes to Opacus. We ask contributors to maintain compatibility to ensure a great user experience with their method.
