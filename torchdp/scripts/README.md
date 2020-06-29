# Directory content

* `compute_dp_sgd_privacy.py` can be used as a **shell script** to compute the privacy of a model trained with DP-SGD.
 Its purpose is to get directly the privacy budget of an iterated Sampled Gaussian Mechanism. As it uses the same Rényi-DP accountant as PyTorch-DP does, the result is identical, but here no training phase is needed, neither is the instantiation of a model.

*Note that the Jupyter notebook  `../../docs/DP_Computation_in_SGM.ipynb` demonstrate how to use this script and provides explainations about the calculations under the hood.*
