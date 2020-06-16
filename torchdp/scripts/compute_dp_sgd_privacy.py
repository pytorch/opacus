#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Command-line script for computing privacy of a model trained with DP-SGD.
The script applies the RDP accountant to estimate privacy budget of an iterated
 Sampled Gaussian Mechanism.

Both `apply_dp_sgd_analysis()` and `get_privacy_spent()` functions
 are based on Google's TF Privacy:
 https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py


Example:
  compute_dp_sgd_privacy
    --dataset-size=60000 \
    --batch-size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5 \
    --a 10 20 100
The calculated privacy with these parameters satisfies (2.95, 1e-5)-DP.

The argument -a or --alphas is for entering the list of RDP alpha orders.
"""
import argparse
import math
from torchdp import privacy_analysis as tf_privacy


def apply_dp_sgd_analysis(
    sample_rate, noise_multiplier, steps, alphas, delta, printed=True
):
    """Compute and print results of DP-SGD analysis.

    Args:
        sample_rate: the sample rate in SGD.
        noise_multiplier: the noise_multiplier in compute_rdp(), ratio of the
               standard deviation of the Gaussian noise to
               the l2-sensitivity of the function to which it is added
        steps: the number of steps.
        alphas: an array (or a scalar) of RDP alpha orders.
        printed: boolean, True (by default) to print results on stdout.
    """
    rdp = tf_privacy.compute_rdp(sample_rate, noise_multiplier, steps, alphas)
    #  Slight adaptation from TF version, in which
    # `get_privacy_spent()` has one more arguments and one more element in
    #  returned tuple, because it can also compute delta for a given epsilon
    #  (and not only compute epsilon for a targeted delta).
    eps, opt_alpha = tf_privacy.get_privacy_spent(alphas, rdp, delta=delta)

    if printed:
        print(
            f"DP-SGD with\n\tsampling rate = {100 * sample_rate:.3g}% and"
            f"\n\tnoise_multiplier = {noise_multiplier}"
            f"\n\titerated over {steps} steps\n  satisfies "
            f"differential privacy with\n\tƐ = {eps:.3g} "
            f"and\n\tδ = {delta}."
            f"\n  The optimal α is {opt_alpha}."
        )

        if opt_alpha == max(alphas) or opt_alpha == min(alphas):
            print(
                "The privacy estimate is likely to be improved by expanding "
                "the set of alpha orders."
            )

    return eps, opt_alpha


def compute_dp_sgd_privacy(
    sample_size, batch_size, noise_multiplier, epochs, delta, alphas=None, printed=True
):
    """Compute epsilon based on the given parameters.
    """
    sample_rate = batch_size / sample_size
    if sample_rate > 1:
        raise ValueError("sample_size must be larger than the batch size.")
    steps = epochs * math.ceil(sample_size / batch_size)

    return apply_dp_sgd_analysis(
        sample_rate, noise_multiplier, steps, alphas, delta, printed
    )


def main():
    # Settings w.r.t. parameters on command line or default values if missing
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

    print(
        "=================================================================="
        "\n* Script was called with arguments:\n\t"
        f"-s = --dataset-size = {args.dataset_size}\n\t"
        f"-b = --batch-size = {args.batch_size}\n\t"
        f"-n = --noise_multiplier = {args.noise_multiplier}\n\t"
        f"-e = --epochs = {args.epochs}\n\t"
        f"-d = --delta = {args.delta}\n\t"
        f"-a = --aplhas = {args.alphas}\n\n"
        "* Result is:",
        end="\n  ",

    )
    compute_dp_sgd_privacy(
        args.dataset_size,
        args.batch_size,
        args.noise_multiplier,
        args.epochs,
        args.delta,
        args.alphas,
    )
    print("==================================================================")


if __name__ == "__main__":
    main()
