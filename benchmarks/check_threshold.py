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

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-path",
        type=str,
        help="path to the report produced by generate_report.py",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to be checked",
        choices=["runtime", "memory"],
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Report column to be checked",
    )
    parser.add_argument(
        "--threshold",
        type=float,
    )
    args = parser.parse_args()

    r = pd.read_pickle(args.report_path).fillna(0)
    if (r.loc[:, (args.metric, args.column)] < args.threshold).all():
        exit(0)
    else:
        exit(1)
