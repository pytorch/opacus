from opacus.accountants import create_accountant


# min range bound when searching for sigma given epsilon
DEFAULT_SIGMA_MIN_BOUND = 0.01
# starting point for a max range bound when searching for sigma given epsilon
DEFAULT_SIGMA_MAX_BOUND = 10
# condition to halt binary search for sigma given epsilon
SIGMA_PRECISION = 0.01
# max possible value for returned sigma.
# Noise higher than MAX_SIGMA considered unreasonable
MAX_SIGMA = 2000


def get_epsilon(
    *,
    delta: float,
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    accountant: str,
    **kwargs,
):
    """Get epsilon given all parameters."""
    accountant = create_accountant(mechanism=accountant)
    for _ in range(num_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta, **kwargs)


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    accountant: str = "rdp",
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
        accountant: accounting mechanism used to estimate epsilon
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    eps = float("inf")
    sigma_min = DEFAULT_SIGMA_MIN_BOUND
    sigma_max = DEFAULT_SIGMA_MAX_BOUND

    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        eps = get_epsilon(
            accountant=accountant,
            delta=target_delta,
            noise_multiplier=sigma_max,
            sample_rate=sample_rate,
            num_steps=int(epochs / sample_rate),
        )
        if sigma_max > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > SIGMA_PRECISION:
        sigma = (sigma_min + sigma_max) / 2
        eps = get_epsilon(
            accountant=accountant,
            delta=target_delta,
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            num_steps=int(epochs / sample_rate),
        )

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma
