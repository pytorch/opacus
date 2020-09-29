#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
from copy import deepcopy
from enum import IntEnum
from typing import Any, Dict, Optional


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    warnings.warn("Tensorboard library was not found. Using dummy SummaryWriter")

    class SummaryWriter:
        def add_scalar(self, *args, **kwargs):
            pass


class StatType(IntEnum):
    r"""
    This enum covers all the stat types we currently support.

    1. LOSS: Monitors the training loss.
    2. Grads: Monitors stats about the gradients across iterations
    3. PRIVACY: Logs Epsilon so you can see how it evolves during training
    4. TRAIN: This is a TB namespace where you can attach training metrics
    5. TEST: Similar to TRAIN, just another TB namespace to log things under
    """
    LOSS = 1
    GRAD = 2
    PRIVACY = 3
    TRAIN = 4
    TEST = 5


class Stat:
    r"""
    Wrapper around tensorboard's ``SummaryWriter.add_scalar``, allowing for sampling
    and easier interface.

    Use this to gather and visualize statistics to get insight about
    differential privacy parameters, and to observe how clipping and noising affects the training process
    (loss, accuracy, etc).

    We have already implemented some common ones inside ``opacus.utils.stat.StatType``.

    Internal Privacy metrics (such as ``StatType.PRIVACY`` and ``StatType.GRAD``)
    are already added to the code and need only be activated by adding the stat
    as shown in the example. Other stat types need to be added to the stat
    and updated properly using ``update`` function.

    Examples:
        To get stats about clipping you can add the following line
        to your main file. By default the samples are averaged and the average is
        reported every ``1 / frequency`` times.

        >>> stat = Stat(StatType.GRAD, 'sample_stats', frequency=0.1)
        >>> for i in range(20):
        >>>    stat.log({"val":i})

        If an instance of ``tensorboard.SummaryWriter`` exists it can be used
        for stat gathering by passing it like this:

        >>> stats.set_global_summary_writer(tensorboard.SummaryWriter())

        To add stats about test accuracy you can do:

        >>> stats.add(Stat(stats.StatType.TEST, 'accuracy', frequency=0.1))

        and then update the stat meter in the proper location using:

        >>> acc1_value = compute_accuracy(x, y)  # you can supply your metrics functions, and Stats later displays them
        >>> stats.update(stats.StatType.TEST, acc1=acc1_value)  # pass to Stats the result so that the result gets logged
    """
    summary_writer: Optional[SummaryWriter] = None

    def __init__(
        self,
        stat_type: StatType,
        name: str,
        frequency: float = 1.0,
        reduction: str = "avg",
    ):
        r"""
        Args:
            stat_type: Type of the statistic from ``StatType``.
            name: Name of the stat that is used to identify this ``Stat``
                for update or to view in tensorboard.
            frequency: The frequency of stat gathering. Its value is in [0, 1],
                where e.g. 1 means report to tensorboard any time ``log`` is
                called and 0.1 means report only 1 out of 10 times.
            reduction: The reduction strategy used for reporting, e.g. if
                ``frequency = 0.1`` and ``reduction='avg'`` then ``log`` averages
                10 samples and reports to tensorboard this average once every 10
                samples. Current valid values are 'avg' and 'sample'.
        """
        self.type = stat_type
        self.name = name
        self.report = int(1 / frequency)
        self.reduction = reduction
        self.writer = Stat.summary_writer if Stat.summary_writer else SummaryWriter()
        self.named_values = []
        self.reset()

    def reset(self):
        """
        Resets the accumulated metrics.
        """
        self.named_value = {}
        self.iter = 0

    def log(self, named_value: Dict[str, Any], hist: bool = False):
        r"""
        Logs a metrics to tensorboard.

        Generally not used directly (use ``update`` instead).

        Args:
            named_value: A dictionary of metrics to log
        """
        assert not (self.reduction == "avg" and hist)
        if self.iter % self.report == 0:
            for k, v in self.named_value.items():
                self.writer.add_histogram(
                    f"{self.type.name}:{self.name}/{k}", v, self.iter
                ) if hist else self.writer.add_scalar(
                    f"{self.type.name}:{self.name}/{k}", v, self.iter
                )
        self._aggregate(named_value)

    def _aggregate(self, named_value: Dict[str, Any]):
        """
        Aggregates ``named_value`` using this object's ``reduction`` attribute.

        Args:
            named_value: The value to aggregate
        """
        if self.reduction == "sample":
            self.named_value = deepcopy(named_value)
        elif self.reduction == "avg":
            for k, v in named_value.items():
                self.named_value[k] = (
                    self.named_value[k] + float(v) / self.report
                    if (self.iter % self.report)
                    else float(v) / self.report
                )
        self.iter += 1


# global variable keeping the list of all the stats.
Stats = []


def set_global_summary_writer(summary_writer: SummaryWriter):
    """
    Sets this object's TensorBoard SummaryWriter to an externally provided one.

    Useful if you already have one instantiated and you don't want this to
    create another unnecessarily.

    Args:
        summary_writer: The externally provided SummaryWriter
    """
    Stat.summary_writer = summary_writer


def add(*args: Stat):
    r"""
    Adds statistics gathering to the process.

    Args:
        *args: An iterable of statistics to add
    """
    [Stats.append(stat) for stat in args]


def clear():
    r"""
    Clears all stats and stops collecting statistics.
    """
    Stats.clear()


def remove(name: str):
    r"""
    Removes the Stat of name ``name`` from the global statistics gathering.

    Args:
        name: The name of stats to remove
    """
    global Stats
    Stats = [stat for stat in Stats if stat.name != name]


def reset(stat_type: Optional[StatType] = None, name: Optional[str] = None):
    r"""
    Resets the stat with given `name` and `stat_type`

    Args:
        stat_type: The stat_type to reset
        name: The name of stats to reset
    """
    [
        stat.reset()
        for stat in Stats
        if (stat_type is None or stat.type == stat_type)
        and (name is None or stat.name == name)
    ]


def update(
    stat_type: Optional[StatType] = None,
    name: Optional[str] = None,
    hist: bool = False,
    **named_values: str,
):
    r"""
    Updates the stat(s) with the given ``name`` and ``stat_type``

    Args:
        stat_type: The type of the stat from ``StatType``. Could be
            ``None`` if ``name`` is unique.
        name: The name of the stat. Could be ``None`` if there is only
            one stat for the ``stat_type``
        **named_values: A set of values with their names
    """
    [
        stat.log(named_values, hist)
        for stat in Stats
        if (stat_type is None or stat.type == stat_type)
        and (name is None or stat.name == name)
    ]
