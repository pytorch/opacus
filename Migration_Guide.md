# Guide: How to migrate to 1.0 API

This guide will help you update your code from `opacus==0.x` to `opacus==1.x`.

With the new release we're introducing a slightly different approach to the user-facing library API. While heavily based on the old API, updated API better represents abstractions and algorithms used in DP in ML, enabling private training exactly as it's described in the papers, with no assumptions or simplifications. And in doing so we maintain our focus on high performance training.

On the downside, however, the new API lacks backward compatibility. If you've been using older versions of Opacus and want to continue using Opacus 1.0, you'll need to perform certain manual steps. In the vast majority of cases the changes required are trivial, but this can vary depending on your exact setup. This guide will help you through this process.

# Table of Contents
  * [New API intro](#new-api-intro)
  * [Simple migration](#simple-migration)
    + [Basics](#basics)
    + [Privacy accounting](#privacy-accounting)
    + [Zero grad](#zero-grad)
  * [Your model has BatchNorm](#your-model-has-batchnorm)
  * [If you're using virtual steps](#if-youre-using-virtual-steps)
  * [When you know privacy budget in advance](#when-you-know-privacy-budget-in-advance)
  * [Distributed](#distributed)
  * [No DataLoader](#no-dataloader)

## New API intro

First, a quick recap on how the new API looks.

The first difference you'll notice is increased focus on data handling. Batch sampling is an important component of DP-SGD (e.g. privacy accounting relies on amplification by sampling) and Poisson sampling is quite tricky to get right. So now Opacus takes control of 3 PyTorch training objects: model, optimizer, and data loader.

Here's a simple example:

```python
# define your components as usual
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

# enter PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
# Now it's business as usual
```

What actually happens in the `make_private` method deserves more attention and we'll cover it later in this doc. For now all we need to know is that `make_private` takes three fully initialized objects (model, optimizer and data loader), along with privacy configuration parameters. `make_private` method then returns wrappers, each taking some additional privacy-related responsibilities, while also doing everything the original modules do.

- model is wrapped with `GradSampleModule`, which computes per sample gradients
- optimizer is wrapped with `DPOptimizer`, which does gradient clipping and noise addition
- data_loader is now a `DPDataLoader`, which performs uniform with replacement batch sampling,
  as required by privacy accountant

## Simple migration

### Basics

Let's take the simplest (and hopefully the most common) migration use case. We assume that we're using standard PyTorch `DataLoader` and take an example we used to demonstrate the old API with in the Opacus readme.

```diff
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)

+ # in addition to model and optimizer you now need access to a data loader
+ data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

+ # PrivacyEngine's constructor doesn't accept training artifacts - they're instead passed to make_private
+ privacy_engine = PrivacyEngine()

+ model, optimizer, data_loader = privacy_engine.make_private(
- privacy_engine = PrivacyEngine(
+     module=model, # Parameter names are required
+     optimizer=optimizer,
+     data_loader=data_loader,
-     sample_rate=0.01, # It's automatically inferred from the data loader
-     alphas=[10, 100], # Not required at this stage. You can provide custom alpha when computing epsilon
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
- privacy_engine.attach(optimizer) # Just continue training using returned objects

# Now it's business as usual
```

### Privacy accounting

This part is mostly unchanged, except that the API is now adapted to a more generic concept of privacy accountant. We've already implemented two accountants: RDP (default and recommended one) and Gaussian DP accountant.

In most cases, here's what you'll need to change:
```diff
+ eps = privacy_engine.get_epsilon(delta=target_delta)
- eps, alpha = privacy_engine.get_privacy_spent(delta=target_delta)
```

Note, that you no longer have access to alpha, as it's RDP-specific parameter and isn't applicable to other privacy accountants. If you need to provide custom alphas, you can pass it as an argument to `get_epsilon`:
```python
eps = privacy_engine.get_epsilon(delta=target_delta, alphas=alphas)
```

And if you need access to the `best_alpha` corresponding to your epsilon, you can get it from the accountant object itself, assuming you've initialized `PrivacyEngine` with the default accounting mechanism.

```python
eps, alpha = privacy_engine.accountant.get_privacy_spent(delta=target_delta, alphas=alphas)
```

### Zero grad

The previous Opacus version didn't require you to call `optimizer.zero_grad()` - Opacus would clear gradients after optimization steps regardless. Now we rely on user to call the method (but will still detect and throw an exception if it's not done)

## Your model has BatchNorm

By default `PrivacyEngine` only does module validation - you have to pass a module that already meets the expectations. We've aggregated all known module fixes, including `BatchNorm -> GroupNorm` replacement into `ModuleValidator.fix()`

`ModuleValidator.fix()` also performs other known remediations like replacing `LSTM` with `DPLSTM`. For the full list of actions see `opacus.validators` package docs

```diff
+ model = ModuleValidator.fix(model)
- model = module_modification.convert_batchnorm_modules(model)
```

## If you're using virtual steps

Old Opacus featured the concept of virtual steps - you could decouple the logical batch size (which defined how often model weights are updated and how much DP noise is added) and physical batch size (which defined the maximum physical batch size processed by the model at any one time).
While the concept is extremely useful, it suffers from some serious flaws:
- Not compatible with poisson sampling. Two subsequent poisson batches with `sample_rate=x` are not equivalent
  to a single batch with `sample_rate=2x`. Therefore simulating larger batches by setting lower sampling rate isn't
  really Poisson anymore.
- It didn't protect from occasional large Poisson batches. When working with Poisson sampling, setting batch size
  (or rather sampling rate) was quite tricky. For long enough training loops, peak batch size (and therefore memory
  consumption) could be much larger than the average.
- It required careful manual crafting inside the training loop.

```python
BATCH_SIZE = 128 # that's the logical batch size. You'll mostly be using this one across your code
MAX_PHYSICAL_BATCH_SIZE = 32 # physical limit on batch size. You'll use this once

model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

model, optimizer, data_loader = privacy_engine.make_private(...)

with BatchMemoryManager(
        data_loader=data_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
) as new_data_loader:
    for data, label in new_data_loader: # Note: you have to use new data loader initialized by the context manager
      # continue training as normal
```

This approach addressed all of the issues above: it simulated proper poisson batches, can be used as a safeguard against occasional large batches even if you don't want to use virtual batches (just set `max_physical_batch_size=batch_size`) and is easy to use.

## When you know privacy budget in advance

To avoid mutually exclusive method parameters, we're now providing separate method to initialize training loop if epsilon is to be provided instead of noise_multiplier

```python
model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
```

## Distributed

Nothing has changed here. The only thing you should know is that `DifferentiallyPrivateDistributedDataParallel` is moved to a different module:

```diff
+ from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
- from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
```

## No DataLoader

Now, if you're using something else as your data source, things get interesting. You'll still be able to use Opacus, but will need to do a little more.

`PrivacyEngine` is intentionally designed to expect and amend `DataLoader`, as this is the right thing to do in the majority of cases. However, the good news is that `PrivacyEngine` itself is not absolutely necessary - if you know what you're doing, and are happy with whatever data source you have, here's how to plug in Opacus.

NB: This is only a brief example of using Opacus components independently of `PrivacyEngine`.
See [this tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb) for extended guide.

```python
# business as usual
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)

# initialize privacy accountant
from opacus.accountants import RDPAccountant
accountant = RDPAccountant()

# wrap model
from opacus import GradSampleModule
dp_model = GradSampleModule(model)

# wrap optimizer
from opacus.optimizers import DPOptimizer
dp_optimizer = DPOptimizer(
  optimizer=optimizer,
  noise_multiplier=1.0, # same as make_private arguments
  max_grad_norm=1.0, # same as make_private arguments
  expected_batch_size=batch_size # if you're averaging your gradients, you need to know the denominator
)

# attach accountant to track privacy for an optimizer
dp_optimizer.attach_step_hook(
    accountant.get_optimizer_hook_fn(
      # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
      sample_rate=sample_rate
    )
)
```
