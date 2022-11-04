import torch.nn as nn
from opacus.layers.dp_rnn import RNNLinear


def prepare_layer(layer, batch_first=True):
    """
    Prepare a layer to compute grad samples using functorch.
    The grad samples are computed by redoing the forward and
    backward passes on the functional version of the module.

    Args:
        layer: the layer to prepare
        batch_first: whether the input is batch_first or not
    """
    from functorch import grad, make_functional, vmap

    if len(list(layer.buffers())) > 0:
        raise NotImplementedError(
            "This layer has buffers and is not supported by Opacus"
        )
    if type(layer) is nn.EmbeddingBag:
        raise NotImplementedError("Functorch does not support EmbeddingBag yet")
    flayer, _ = make_functional(layer)

    def compute_loss_stateless_model(params, activations, backprops):
        if batch_first or type(layer) is RNNLinear:
            batched_activations = activations.unsqueeze(0)
            batched_backprops = backprops.unsqueeze(0)
        else:
            # If batch_first is False, the batch dimension is the second dimension
            batched_activations = activations.unsqueeze(1)
            batched_backprops = backprops.unsqueeze(1)

        output = flayer(params, batched_activations)
        loss = (output * batched_backprops).sum()

        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    # Note that the vmap is done on the first dimension, regardless of batch_first
    # This is because the activations and backprops given by the GradSampleModule
    # are always batch_first=True
    layer.ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))


def ft_compute_per_sample_gradient(layer, activations, backprops):
    """
    Compute the per-sample gradient of the layer.
    Args:
        layer: the layer on which to compute the gradient
        activations: the input to the layer
        backprops: the  gradient of the loss w.r.t. outputs of the layer
    """
    parameters = list(layer.parameters(recurse=True))
    if not hasattr(layer, "ft_compute_sample_grad"):
        prepare_layer(layer)

    per_sample_grads = layer.ft_compute_sample_grad(
        parameters, activations[0], backprops
    )

    ret = {}
    for i_p, p in enumerate(parameters):
        ret[p] = per_sample_grads[i_p]

    return ret
