#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Command-line script for computing privacy of a model trained with DP-SGD.
The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism.

The code is mainly based on Google's TF Privacy:
https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py


Example:

    To call this script from command line, you can enter:

    $ python -m opacus.scripts.compute_dp_sgd_privacy --epochs=3 --delta=1e-5 --sample-rate 0.01 --noise-multiplier 1.0 --alphas 2 5 10 20 100

    DP-SGD with
    - sampling rate = 1%,
    - noise_multiplier = 1.0,
    - iterated over 300 steps

    satisfies differential privacy with
    - epsilon = 2.39,
    - delta = 1e-05.

    The optimal alpha is 5.0.
"""
import argparse
import math
from typing import List, Tuple

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


def _apply_dp_sgd_analysis(
    *,
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    alphas: List[float],
    delta: float,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Computes the privacy Epsilon at a given delta via RDP accounting and
    converting to an (epsilon, delta) guarantee for a target Delta.

    Args:
        sample_rate : The sample rate in SGD
        noise_multiplier : The ratio of the standard deviation of the Gaussian
            noise to the L2-sensitivity of the function to which the noise is added
        steps : The number of steps
        alphas : A list of RDP orders
        delta : Target delta
        verbose : If enabled, will print the results of DP-SGD analysis

    Returns:
        Pair of privacy loss epsilon and optimal order alpha
    """
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, alphas)
    eps, opt_alpha = get_privacy_spent(alphas, rdp, delta=delta)

    if verbose:
        print(
            f"DP-SGD with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {noise_multiplier},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {delta}."
            f"\nThe optimal alpha is {opt_alpha}."
        )

        if opt_alpha == max(alphas) or opt_alpha == min(alphas):
            print(
                "The privacy estimate is likely to be improved by expanding "
                "the set of alpha orders."
            )
    return eps, opt_alpha


def compute_dp_sgd_privacy(
    *,
    sample_rate: float,
    noise_multiplier: float,
    epochs: int,
    delta: float,
    alphas: List[float],
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Performs the DP-SGD privacy analysis.

    Finds sample rate and number of steps based on the input parameters, and calls
    DP-SGD privacy analysis to find the privacy loss epsilon and optimal order alpha.

    Args:
        sample_rate : probability of each sample from the dataset to be selected for a next batch
        noise_multiplier : The ratio of the standard deviation of the Gaussian noise
            to the L2-sensitivity of the function to which the noise is added
        epochs : Number of epochs
        delta : Target delta
        alphas : A list of RDP orders
        verbose : If enabled, will print the results of DP-SGD analysis

    Returns:
        Pair of privacy loss epsilon and optimal order alpha

    Raises:
        ValueError
            When batch size is greater than sample size
    """
    if sample_rate > 1:
        raise ValueError("sample_rate must be no greater than 1")
    steps = epochs * math.ceil(1 / sample_rate)

    return _apply_dp_sgd_analysis(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=steps,
        alphas=alphas,
        delta=delta,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate privacy of a model trained with DP-SGD using RDP accountant",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=float,
        required=True,
        help="Input sample rate (probability of each sample from the dataset to be selected for a next batch)",
    )
    parser.add_argument(
        "-n",
        "--noise-multiplier",
        type=float,
        required=True,
        help="Noise multiplier",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "-d", "--delta", type=float, default=1e-5, help="Targeted delta (default: 1e-5)"
    )
    parser.add_argument(
        "-a",
        "--alphas",
        action="store",
        dest="alphas",
        type=float,
        nargs="+",
        default=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        help="List of alpha values (alpha orders of Renyi-DP evaluation). "
        "A default list is provided. Else, space separated numbers. E.g.,"
        "-a 10 100",
    )

    args = parser.parse_args()

    compute_dp_sgd_privacy(
        sample_rate=args.sample_rate,
        noise_multiplier=args.noise_multiplier,
        epochs=args.epochs,
        delta=args.delta,
        alphas=args.alphas,
    )


if __name__ == "__main__":
    main()
