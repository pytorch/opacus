#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys


def filter_out_old_keys(self, state_dict, prefix, local_metadata):
    new_state_dict = {
        param_name: param_value
        for param_name, param_value in state_dict.items()
        if param_name not in self.old_to_new
    }
    return new_state_dict


class ParamRenamedModule(nn.Module):
    """
    This class defines a nn.Module whose parameters are renamed. This is useful when you want to
    reimplement a layer but make sure its state_dict and list of parameters are exactly the same
    as another reference layer so that you can have a drop-in replacement that does not depend on
    how your layer is actually implemented. In Opacus, this is used for DPLSTM, where our
    implementation leverages submodules and requires alignment to the state_dict of nn.LSTM.
    """

    def __init__(self, rename_map: Dict[str, str]):
        """
        Initializes internal state. Subclass this instead of ``torch.nn.Module`` whenever you need
        to rename your model's state.

        Args:
            rename_map: mapping from old name -> new name for each parameter you want renamed.
                Note that this must be a 1:1 mapping!
        """
        super().__init__()
        self.old_to_new = rename_map
        self.new_to_old = {v: k for k, v in rename_map.items()}

        self._register_state_dict_hook(filter_out_old_keys)

    def _register_renamed_parameters(self):
        """
        Internal function. This function simply registers parameters under their new name. They will
        automatically mask their duplicates coming from submodules. This trick works because
        self.parameters() proceeds recursively from the top, going into submodules after processing
        items at the current level, and will not return duplicates.
        """
        for old_name, param in super().named_parameters():
            if old_name in self.old_to_new:
                new_name = self.old_to_new[old_name]
                self.register_parameter(new_name, param)

    def __setattr__(self, name: str, value: Union[Tensor, nn.Module]) -> None:
        """
        Whenever you set an attribute, eg `self.linear`, this is called to actually register it in
        any nn.Module. We rely on the masking trick explained in the docs for
        ``_register_renamed_parameters`` to make sure we replace things only once. If a new parameter
        in the rename list is detected, we rename and mask it so next time this is called we will
        no longer find it.
        """
        super().__setattr__(name, value)
        try:
            self._register_renamed_parameters()
        except AttributeError:
            # At the very beginning of instantiation, this will fail because we do not yet have
            # self._parameters. Safe to ignore.
            pass

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
    ):
        """
        Identical to ``torch.nn.Module.load_state_dict()`` but handles the renamed keys.
        """

        # nn.Module recomputes its state_dict(), without calling the same logic as in self.state_dict()
        # This means that it will find both the old and the renamed parameters. Both point to the
        # same parameter object, so either of them will set it correctly. It will however complain
        # that some keys are missing (the "old" keys). We can safely ignore those and process them
        # accordingly

        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict=False
        )
        missing_keys = [k for k in missing_keys if k not in self.old_to_new]
        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        return _IncompatibleKeys(missing_keys, unexpected_keys)
