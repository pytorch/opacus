<p align="center"><img src="https://github.com/pytorch/opacus/blob/main/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

[![CircleCI](https://circleci.com/gh/pytorch/opacus.svg?style=svg)](https://circleci.com/gh/pytorch/opacus)

[Opacus](https://opacus.ai) is a library that enables training PyTorch models with differential privacy. It supports training with minimal code changes required on the client, has little impact on training performance and allows the client to online track the privacy budget expended at any given moment.

## Target audience
This code release is aimed at two target audiences:
1. ML practitioners will find this to be a gentle introduction to training a model with differential privacy as it requires minimal code changes.
2. Differential Privacy scientists will find this easy to experiment and tinker with, allowing them to focus on what matters.


## Installation
The latest release of Opacus can be installed via `pip`:
```bash
pip install opacus
```

You can also install directly from the source for the latest features (along with its quirks and potentially ocassional bugs):
```bash
git clone https://github.com/pytorch/opacus.git
cd opacus
pip install -e .
```

## Getting started
To train your model with differential privacy, all you need to do is to instantiate a `PrivacyEngine` and pass your model, data_loader, and optimizer to the engine's `make_private()` method to obtain their private counterparts.

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

The [MNIST example](https://github.com/pytorch/opacus/tree/main/examples/mnist.py) shows an end-to-end run using opacus. The [examples](https://github.com/pytorch/opacus/tree/main/examples/) folder contains more such examples.

## FAQ
Checkout the [FAQ](https://opacus.ai/docs/faq) page for answers to some of the most frequently asked questions about Differential Privacy and Opacus.

## Contributing
See the [CONTRIBUTING](https://github.com/pytorch/opacus/tree/main/CONTRIBUTING.md) file for how to help out.
Do also check out the README files inside the repo to learn how the code is organized.

## Citation
To cite Opacus in your papers (much appreciated!), please use the following:
```
@article{opacus,
  title={Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author={A. Yousefpour and I. Shilov and A. Sablayrolles and D. Testuggine and K. Prasad and M. Malek and J. Nguyen and S. Ghosh and A. Bharadwaj and J. Zhao and G. Cormode and I. Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

## License
This code is released under Apache 2.0, as found in the [LICENSE](https://github.com/pytorch/opacus/tree/main/LICENSE) file.
