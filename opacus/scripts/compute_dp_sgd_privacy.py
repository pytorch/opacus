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

    >>>  python compute_dp_sgd_privacy.py --dataset-size=60000 --batch-size=256 --noise_multiplier=1.12 --epochs=60 --delta=1e-5 --a 10 20 100

    The training process with these parameters satisfies (epsilon,delta)-DP of (2.95, 1e-5).
"""
import argparse
import math
from typing import List, Tuple

from opacus import privacy_analysis


def _apply_dp_sgd_analysis(
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
    rdp = privacy_analysis.compute_rdp(sample_rate, noise_multiplier, steps, alphas)
    eps, opt_alpha = privacy_analysis.get_privacy_spent(alphas, rdp, delta=delta)

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
    sample_size: int,
    batch_size: int,
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
        sample_size : The size of the sample (dataset)
        batch_size : Batch size
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
    sample_rate = batch_size / sample_size
    if sample_rate > 1:
        raise ValueError("sample_size must be larger than the batch size.")
    steps = epochs * math.ceil(sample_size / batch_size)

    return _apply_dp_sgd_analysis(
        sample_rate, noise_multiplier, steps, alphas, delta, verbose
    )


def main():
    parser = argparse.ArgumentParser(description="RDP computation")
    parser.add_argument(
        "-s",
        "--dataset-size",
        type=int,
        default=60000,
        help="Training dataset size (default: 60_000)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        help="Input batch size (default: 256)",
    )
    parser.add_argument(
        "-n",
        "--noise_multiplier",
        type=float,
        default=1.12,
        help="Noise multiplier (default: 1.12)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=60,
        help="Number of epochs to train (default: 60)",
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
        args.dataset_size,
        args.batch_size,
        args.noise_multiplier,
        args.epochs,
        args.delta,
        args.alphas,
    )


if __name__ == "__main__":
    main()
