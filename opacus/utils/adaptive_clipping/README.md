# Adaptive Clipping (with Ghost Clipping)

Adaptive clipping [1] adapts the clipping norm (and amount of noise) during training to a quantile of per-sample gradient norms. It can reduce hyper-parameter tuning efforts and improve model accuracy by injecting less noise.

It is supported with:
- Ghost clipping
- Distributed data parallel training

It is **not** currently supported with:
- Vanilla DP-SGD
- Virtual batch sizes via Batch Memory Manager

## Overview

`PrivacyEngineAdaptiveClipping` is the entry-point for adaptive clipping training. It extends `PrivacyEngine` with additional arguments for adaptive clipping:

* `target_unclipped_quantile`: the quantile of per-sample gradient norms at which to clip (between 0 and 1)
* `min_clipbound`: the minimum allowed clipping norm
* `max_clipbound`: the maximum allowed clipping norm
* `clipbound_learning_rate`: the learning rate for tracking the true quantile
* `max_grad_norm`: the initial clipping norm (used at step 0)

The main hyper-parameter to tune is `target_unclipped_quantile`, which replaces tuning the clipping norm (`max_grad_norm`) in constant clipping DP-SGD. This parameter can be easier to tune, since the search is over a smaller range of values.


## Example usage

```python
from opacus.utils.adaptive_clipping.adaptive_clipping_utils import PrivacyEngineAdaptiveClipping

# ...
privacy_engine = PrivacyEngineAdaptiveClipping()
model, optimizer, criterion, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    criterion=criterion,
    noise_multiplier=args.sigma,
    max_grad_norm=10,  # initial clipping norm
    grad_sample_mode="ghost",
    target_unclipped_quantile=0.5,  # key parameter, may need tuning
    min_clipbound=1,  # default value
    max_clipbound=1e8,  # default value
    clipbound_learning_rate=0.2  # default value, tuning not recommended
)
# ...
```

Note that `grad_sample_mode` must be set to `"ghost"` for adaptive clipping to work.

## References

[1] Galen Andrew, Om Thakkar, H. Brendan McMahan, Swaroop Ramaswamy, "Differentially Private Learning with Adaptive Clipping", NeurIPS, 2021.
