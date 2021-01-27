# Developer Notes
These developer notes can help you ramp up on this codebase. For any question, hit us up on the [forums](https://discuss.pytorch.org/c/opacus/29)!

## Code structure
The code in Opacus is structured as such:

    .
    ├── layers/                              # (folder) Reimplemented layers made compatible with Opacus
    ├── utils/                               # (folder) Extra utils
    ├── tests/                               # (folder) All the tests are here.
    |
    ├── autograd_grad_sample.py             # All the logic to compute per-sample gradients
    ├── dp_model_inspector.py               # Checks that your nn.Module is compatible with DP Training
    ├── per_sample_gradient_clipping.py     # Contains the clipper class and logic
    ├── privacy_engine.py                   # Main entry point for customers
    ├── supported_layers_grad_samplers.py   # Contains the list of supported layers
    ├── privacy_engine.py                   # Main entry point for customers
    └── README.md                           # This file


## Supported modules
Opacus only works with supported ``nn.Module``s. The following modules are supported:

1. Modules with no trainable parameters (eg ``nn.ReLU``, `nn.Tanh`)
2. Modules which are frozen. A nn.Module can be frozen in PyTorch by unsetting ``requires_grad``
in each of its parameters, ie `for p in module.parameters(): p.requires_grad = False`.
3. Explicitly supported modules (we keep a dictionary in opacus.SUPPORTED_LAYERS), eg ``nn.Conv2d``.
4. Any complex nn.Module that contains only supported nn.Modules. This means that most models
will be compatible, given that we support most of the common building blocks. This however also
means that Opacus support depends on how a specific ``nn.Module`` is implemented. For example,
``nn.LSTM`` *could* be written by using ``nn.Linear`` (which we support), but its actual
implementation does not use it (so that it can fuse operators and be faster). Any layer that
needs a rewrite to be supported is in the `/layers` folder.

As an example, the following ``nn.Module`` is supported, because it's made entirely of supported
nn.Modules:

```python
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x
```

## Limitations of backward hooks
The implementation of gradient clipping in autograd_grad_sample.py uses backward hooks to capture per-sample gradients.
The `register_backward hook` function has a known issue being tracked at https://github.com/pytorch/pytorch/issues/598. However, this is the only known way of implementing this as of now (your suggestions and contributions are very welcome). The behavior has been verified to be correct for the layers currently supported by opacus.
