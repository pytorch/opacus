## Developer Note:
The implementation of gradient clipping in autograd_grad_sample.py uses backward hooks to capture per-sample gradients.
The `register_backward hook` function has a known issue being tracked at https://github.com/pytorch/pytorch/issues/598. However, this is the only known way of implementing this as of now (your suggestions and contributions are very welcome). The behaviour has been verified to be correct for the layers currently supported by pytorch-dp.
