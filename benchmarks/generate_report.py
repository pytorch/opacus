# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from benchmarks.utils import generate_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-results",
        default="./results/raw",
        type=str,
        help="the path that `run_benchmarks.py` has saved results to.",
    )
    parser.add_argument(
        "--save-path",
        default="./results/report.csv",
        type=str,
        help="path to save the output.",
    )

    parser.add_argument(
        "--format",
        default="csv",
        type=str,
        help="output format",
        choices=["csv", "pkl"],
    )
    args = parser.parse_args()

    generate_report(args.path_to_results, args.save_path, args.format)
