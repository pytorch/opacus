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
    --sigma=1.12 \
    --epochs=60 \
    --delta=1e-5
The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""
import argparse
import math
from privacy_analysis import compute_rdp, get_privacy_spent


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis.

    Args:
        q: the sample rate in SGD.
        sigma: the noise_multiplier in compute_rdp(), ratio of the
               standard deviation of the Gaussian noise to
               the l2-sensitivity of the function to which it is added
        steps: the number of steps.
        orders: an array (or a scalar) of RDP orders.
  """
    rdp = compute_rdp(q, sigma, steps, orders)

    #  Slight adaptation from TF version, in which
    # `get_privacy_spent()` has one more arguments and one more element in
    #  returned tuple, because it can also compute delta for a given epsilon
    #  (and not only compute epsilon for a targeted delta).
    eps, opt_order = get_privacy_spent(orders, rdp, delta=delta)

    print(
        f"DP-SGD with\n\tsampling rate = {100 * q:.3g}% and"
        f"\n\tnoise_multiplier = {sigma}"
        f"\n\titerated over {steps} steps\n  satisfies "
        f"differential privacy with\n\teps = {eps:.3g} and\n\tdelta = {delta}."
        f"\nThe optimal RDP order is {opt_order}."
    )

    if opt_order == max(orders) or opt_order == min(orders):
        print(
            "The privacy estimate is likely to be improved by expanding "
            "the set of orders."
        )

    return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters."""
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise ValueError("n must be larger than the batch size.")
    orders = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )
    steps = int(math.ceil(epochs * n / batch_size))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


def main():
    # Settings w.r.t. parameters on command line or default values if missing
    parser = argparse.ArgumentParser(description="RDP computation")
    parser.add_argument(
        "-n",
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
        "-s",
        "--sigma",
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
    args = parser.parse_args()

    print("=================================================================="
          "\n* Script called with those arguments:\n\t"
          f"-n = --dataset-size = {args.dataset_size}\n\t"
          f"-b = --batch-size = {args.batch_size}\n\t"
          f"-s = --sigma = {args.sigma}\n\t"
          f"-e = --epochs = {args.epochs}\n\t"
          f"-d = --delta = {args.delta}\n"
          "* Result is:",
          end="\n  "
    )
    compute_dp_sgd_privacy(
        args.dataset_size,
        args.batch_size,
        args.sigma,
        args.epochs,
        args.delta,
    )
    print("==================================================================")


if __name__ == "__main__":
    main()
