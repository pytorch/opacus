# Directory content

* `compute_dp_sgd_privacy.py` can be used as a **shell script** for computing privacy of a model trained with DP-SGD. 
 Its purpose is to get directly the privacy budget of an iterated Sampled Gaussian Mechanism. As it uses the same Rényi-DP accountant as PyTorch-DP does, the result is the same, but here no training phase is needed, neither is the instantiation of a model.

* `DP_Computation_in_SGM.ipynb` is a **Jupyter notebook**.
  * It presents the computation of the Differential Privacy budget for a Stochastic Gradient Descent with PyTorch-DP.
  * Links are provided to the main properties in several articles for justifications.
  * Graphs illustrate the influence of each parameter on ε.
