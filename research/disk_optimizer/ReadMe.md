# DiSK: Differentially Private Optimizer with Simplified Kalman Filter for Noise Reduction

## Introduction
This part of the code introduces a new component to the optimizer named DiSK. The code uses a simplifed Kalman to improve the privatized gradient estimate. Speficially, the privatized minibatch gradient is replaced with:


$$\mathbb{g}_{t+\frac{1}{2}}(\xi) = \frac{1-\kappa}{\kappa\gamma}\nabla f(x_t + \gamma(x_t-x_{t-1});\xi) + \Big(1- \frac{1-\kappa}{\kappa\gamma}\Big)\nabla f(x_t;\xi)$$
$$\mathbb{g_{t+\frac{1}{2}}} = \frac{1}{B}\sum_{\xi \in \mathcal{B}_t} \mathrm{clip}_C\left(\mathbb{g}_{t+\frac{1}{2}}(\xi)\right) + w_t$$
$$g_{t}= (1-\kappa)g_{t-1} + \kappa g_{t+\frac{1}{2}}$$

A detailed description of the algorithm can be found at [Here](https://arxiv.org/abs/2410.03883).

## Usage
The code provides a modified privacy engine with three extra arguments:
* kamlan: bool=False
* kappa: float=0.7
* gamma: float=0.5

To use DiSK, follow the steps:

**Step I:** Import KF_PrivacyEngine from KFprivacy_engine.py and set ```kalman=True```

**Step II:** Define a closure (see [here](https://pytorch.org/docs/stable/optim.html#optimizer-step-closure) for example) to compute loss and backward **without** ```zero_grad()``` and perform ```optimizer.step(closure)```

Example of using the DiSK optimizers:

```python
from KFprivacy_engine import KF_PrivacyEngine
# ...
# follow the same steps as original opacus training scripts
privacy_engine = KF_PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=max_grad_norm,
        clipping=clipping,
        grad_sample_mode=args.grad_sample_mode,
        kalman=True, # need this argument
        kappa=0.7, # optional
        gamma=0.5 # optional
    )

# ...
# during training:
def closure(): # compute loss and backward, an example adapting the one used in examples/cifar10.py
    output = model(images)
    loss = criterion(output, target)
    loss.backward()
    return output, loss
output, loss = optimizer.step(closure)
optimizer.zero_grad() 
# compute other matrices
# ...
```

## Citation 
Consider citing the paper is you use DiSK in your papers, as follows:

```
@article{zhang2024disk,
  title={{DiSK}: Differentially private optimizer with simplified kalman filter for noise reduction},
  author={Zhang, Xinwei and Bu, Zhiqi and Balle, Borja and Hong, Mingyi and Razaviyayn, Meisam and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2410.03883},
  year={2024}
}
```

Contributer: Xinwei Zhang. Email: [zhan6234@umn.edu](mailto:zhan6234@umn.edu)

