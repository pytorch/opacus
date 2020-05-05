#!/usr/bin/env python3
"""
Both `apply_dp_sgd_analysis()` and `get_privacy_spent()` functions
 are based on Google's TF Privacy:
 https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py

Command-line script for computing privacy of a model trained with DP-SGD.
The script applies the RDP accountant to estimate privacy budget of an iterated
 Sampled Gaussian Mechanism.

Example:
  compute_dp_sgd_privacy
    --dataset-size=60000 \
    --batch-size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5
The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.

In addition, there is an argument -a or --alphas for the list of RDP α orders
(usually denoted by alpha).
"""
import argparse
import math
from privacy_analysis import compute_rdp, get_privacy_spent

def apply_dp_sgd_analysis(sample_rate,
                          noise_multiplier,
                          steps,
                          alphas,
                          delta,
                          printed=True
):
    """Compute and print results of DP-SGD analysis.

    Args:
        sample_rate: the sample rate in SGD.
        noise_multiplier: the noise_multiplier in compute_rdp(), ratio of the
               standard deviation of the Gaussian noise to
               the l2-sensitivity of the function to which it is added
        steps: the number of steps.
        alphas: an array (or a scalar) of RDP α orders.
        printed: boolean, True (by default) to print results on stdout.
  """
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, alphas)
    #  Slight adaptation from TF version, in which
    # `get_privacy_spent()` has one more arguments and one more element in
    #  returned tuple, because it can also compute delta for a given epsilon
    #  (and not only compute epsilon for a targeted delta).
    eps, opt_alpha = get_privacy_spent(alphas, rdp, delta=delta)

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
                "the set of α orders."
            )

    return eps, opt_alpha


def compute_dp_sgd_privacy(sample_size,
                           batch_size,
                           noise_multiplier,
                           epochs,
                           delta,
                           alphas=None,
                           printed=True,
):
    """Compute epsilon based on the given parameters.
    """
    if alphas is None:
        alphas = ([1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))
            + [128, 256, 512]
    )

    sample_rate = batch_size / sample_size # the sampling ratio
    if sample_rate > 1:
        raise ValueError("sample_size must be larger than the batch size.")
    steps = int(math.ceil(epochs * sample_size / batch_size))

    return apply_dp_sgd_analysis(sample_rate,
                                noise_multiplier,
                                steps,
                                alphas,
                                delta,
                                printed
    )


def main():
    # Settings w.r.t. parameters on command line or default values if missing
    parser = argparse.ArgumentParser(description="RDP computation")
    parser.add_argument(
        "-s",
        "--dataset-size",
        type=int,
        default=60_000,
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
        "-d",
        "--delta",
        type=float,
        default=1e-5,
        help="Targeted delta (default: 1e-5)",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        type=str,
        default=("[1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, "
            + str(list(range(5, 64))).strip('[]')
            + "128, 256, 512]"
        ),
        help="List of alpha values (α orders of Rényi-DP evaluation). "
             "A default list is provided. Else, write it quoted, e.g. "
             "\"[1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0]\"",
    )

    args = parser.parse_args()

    # From str to list of floats for alphas
    args.alphas = [float(a) for a in args.alphas.strip('[]').split(',')]

    print("=================================================================="
          "\n* Script was called with arguments:\n\t"
          f"-s = --dataset-size = {args.dataset_size}\n\t"
          f"-b = --batch-size = {args.batch_size}\n\t"
          f"-n = --noise_multiplier = {args.noise_multiplier}\n\t"
          f"-e = --epochs = {args.epochs}\n\t"
          f"-d = --delta = {args.delta}\n\t"
          f"-a = --aplhas = {args.alphas}\n\n"
          "* Result is:",
          end="\n  "
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
