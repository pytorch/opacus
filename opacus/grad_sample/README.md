# Grad Samples
An integral part of Opacus is to compute per-sample gradients. In order to make this computation as fast as possible, we provide vectorized code for each of the most common "basic modules" that are the building blocks of most ML models. If your model uses these building blocks, then you don't have to do anything!

We always welcome PRs to add nn.Modules we don't yet support, but we also support registering custom grad_sample functions that can expand support just for your project, or even override Opacus's default implementations if they don't suit your needs.

Override as following:

```python
from opacus.grad_sample import register_grad_sampler

@register_grad_sampler(nn.MyCustomClass)
def compute_grad_sample(module, activations, backprops):
    pass
```

Note that you can also pass a list to the decorator, and register one function against multiple nn.Module classes.
