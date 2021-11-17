from opacus.accountants import get_accountant


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    accounting_mechanism: str = "rdp",
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        alphas: the list of orders at which to compute RDP
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    eps = float("inf")
    sigma_min = 0.01
    sigma_max = 10.0
    accountant = get_accountant(accounting_mechanism)

    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        accountant.steps = [(sigma_max, sample_rate, int(epochs / sample_rate))]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        accountant.steps = [(sigma, sample_rate, int(epochs / sample_rate))]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma
