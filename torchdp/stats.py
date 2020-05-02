#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This helper file provides simple wrapper around tensorboard for gathering
statistics as a research tool to get insight into differential privacy and
to observe how clipping, and noising affects loss, accuracy, ...
Internal Privacy metrics (`StatType.PRIVACY` and StatType.CLIPPING` are
already added to the code and need only be activated by adding the stat
as shown in the example. Other stat types need to be added to the stat
and updated properly using `update` function. Example below tries to
show simple use cases.

Example:
    If an instance of `tensorboard.SummaryWriter` exists it can be used
    for stat gathering, o.w. one can be created, e.g.

        stats.set_global_summary_writer(tensorboard.SummaryWriter())

    If you like to get stats about clipping you can add the following line
    to your main file. That is it. The frequency, defines the sampling frequency
    of the statistics. By default the samples are averaged and the average is
    reported every `1 / frequency` times.

        stats.add(
            stats.Stat(stats.StatType.CLIPPING, 'AllLayers', frequency=0.1))

    To add stats about test accuracy you can do:

        stats.add(stats.StatType.TEST, 'accuracy', frequency=0.1))

    and then update the stat in the proper location using:

        # calculate accuracy@1 for example
        stats.update(stats.StatType.TEST, acc1=acc1_value)
"""

from enum import IntEnum
from typing import Any
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # dummy SummaryWriter
    print('Warning! Tensorboard library was not found.')

    class SummaryWriter:
        def add_scalar(self, *args, **kwargs):
            pass


class StatType(IntEnum):
    LOSS = 1
    CLIPPING = 2
    PRIVACY = 3
    TRAIN = 4
    TEST = 5


class Stat:
    """
    Simple wrapper around tensorboard's SummaryWriter.add_scalar
    to allow sampling and easier interface.

     Args:
        stat_type: Type of the statistic from `StatType`.
        name: Name of the stat that is used to identify this `Stat`
        for update or to view in tensorboard.
        frequency: The frequency of stat gathering a value in [0, 1]
        where e.g. 1 means report to tensorboard any time `log` is
        called and 0.1 means report only 1 out of 10 times.
        aggr: The aggregation strategy used for reporting, e.g. if
        `frequency = 0.1` and `aggr='avg'` then `log` averages 10 samples
        and the reports to tensorboard this average once every 10 samples.
        Current valid values are 'avg' and 'sample'.
        summary_writer:

    Example:

        stat = Stat(StatType.CLIPPING, 'tmp', frequency=0.1);
        for i in range(20):
            stat.log(i)
        # reports (iter:0, value:0) and (iter:10, value:4.5) to tensorboard
    """
    summary_writer: SummaryWriter = None
    """
    The global `SummaryWriter` from tensorboard, if is `None` on construction
    of the first `Stat` object, a summary_writer will be created per instance.
    """

    def __init__(self, stat_type : StatType, name : str,
                 frequency : float = 1., aggr : str = 'avg'):
        self.type : StatType = stat_type
        self.name: str = name
        self.report: int = int(1 / frequency)
        self.aggr : str = aggr
        self.writer : SummaryWriter = Stat.summary_writer\
            if Stat.summary_writer else SummaryWriter()
        self.reset()

    def reset(self):
        self.named_value = {}
        self.iter : int = 0

    def log(self, named_value: Any):
        if self.iter % self.report == 0:
            for k, v in self.named_value.items():
                self.writer.add_scalar(
                    f'{self.type.name}:{self.name}/{k}', v, self.iter)
        self._aggregate(named_value)

    def _aggregate(self, named_value : Any):
        if self.aggr == 'sample':
            self.named_values = named_value
        elif self.aggr == 'avg':
            for k, v in named_value.items():
                self.named_value[k] = self.named_value[k] + float(v) / self.report\
                    if (self.iter % self.report) else float(v) / self.report
        self.iter += 1


# global variable keeping the list of all the stats.
Stats = []


def set_global_summary_writer(summary_writer: SummaryWriter):
    Stat.summary_writer = summary_writer


def add(*args):
    """
    Add statistics gathering to the process.
    """
    [Stats.append(stat) for stat in args]


def clear():
    """
    Clear all stats. After calling this gathering
    all statistics will stop.
    """
    Stats.clear()


def remove(name : str):
    """
    Will remove the stat object nemaed
    `name` from the global statistics gathering.
    """
    global Stats
    Stats = [stat for stat in Stats if stat.name != name]


def reset(stat_type : StatType = None, name : str = None):
    """
    Resets the stat with given `name` and `stat_type`
    """
    [stat.reset() for stat in Stats if
        (stat_type is None or stat.type == stat_type)
        and (name is None or stat.name == name)]


def update(stat_type : StatType = None, name : str = None, **named_values):
    """
    updates the stat(s) with the given `name` and `stat_type`

    Args:
        stat_type: type of the stat from `StatType`, could be `None`
            if `name` is unique.
        name: name of the stat, could be `None` if there is only one
            stat for the `stat_type`
        **named_values: set of values with their names
    """
    [stat.log(named_values) for stat in Stats if
        (stat_type is None or stat.type == stat_type)
        and (name is None or stat.name == name)]
