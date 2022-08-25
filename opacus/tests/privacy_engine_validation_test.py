import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.grad_sample.gsm_exp_weights import API_CUTOFF_VERSION
from torch.utils.data import DataLoader


class BasicSupportedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=2)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=8)
        self.fc = nn.Linear(in_features=4, out_features=8)
        self.ln = nn.LayerNorm([8, 8])

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.fc(x)
        x = self.ln(x)
        return x


class CustomLinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._weight = nn.Parameter(torch.randn(out_features, in_features))
        self._bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return F.linear(x, self._weight, self._bias)


class MatmulModule(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_features, output_features))

    def forward(self, x):
        return torch.matmul(x, self.weight)


class LinearWithExtraParam(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.extra_param = nn.Parameter(torch.randn(out_features, 2))

    def forward(self, x):
        x = self.fc(x)
        x = x.matmul(self.extra_param)
        return x


class PrivacyEngineValidationTest(unittest.TestCase):
    """
    This test case checks end-to-end model validation performed in `.make_private`
    method. It covers performed in `ModuleValidator`, `GradSampleModule`, as well as
    their interplay
    """

    def setUp(self) -> None:
        self.privacy_engine = PrivacyEngine()

    def _init(self, module, size, batch_size=10):
        optim = torch.optim.SGD(module.parameters(), lr=0.1)
        dl = DataLoader(
            dataset=[torch.randn(*size) for _ in range(100)],
            batch_size=batch_size,
        )

        return module, optim, dl

    def test_supported_hooks(self):
        module, optim, dl = self._init(BasicSupportedModule(), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="hooks",
        )

        for x in dl:
            module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_supported_ew(self):
        module, optim, dl = self._init(BasicSupportedModule(), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)

    def test_custom_linear_hooks(self):
        module, optim, dl = self._init(CustomLinearModule(5, 8), size=(16, 5))
        try:
            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
        except ImportError:
            print("Test not ran because functorch not imported")

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_custom_linear_ew(self):
        module, optim, dl = self._init(CustomLinearModule(5, 8), size=(16, 5))

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)

    def test_unsupported_hooks(self):
        try:
            module, optim, dl = self._init(MatmulModule(5, 8), size=(16, 5))

            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
        except ImportError:
            print("Test not ran because functorch not imported")

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_unsupported_ew(self):
        module, optim, dl = self._init(
            MatmulModule(input_features=5, output_features=10),
            size=(16, 5),
            batch_size=12,
        )

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        with self.assertRaises(RuntimeError):
            for x in dl:
                module(x)

    def test_extra_param_hooks_requires_grad(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        try:
            gsm, _, _ = self.privacy_engine.make_private(
                module=module,
                optimizer=optim,
                data_loader=dl,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="hooks",
            )
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
            gsm._close()
        except ImportError:
            print("Test not ran because functorch not imported")

    def test_extra_param_hooks_no_requires_grad(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module.extra_param.requires_grad = False
        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="hooks",
        )

        for x in dl:
            module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_extra_param_ew(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )
        with self.assertRaises(RuntimeError):
            for x in dl:
                module(x)

    @unittest.skipIf(
        torch.__version__ < API_CUTOFF_VERSION, "not supported in this torch version"
    )
    def test_extra_param_disabled_ew(self):
        module, optim, dl = self._init(LinearWithExtraParam(5, 8), size=(16, 5))
        module.extra_param.requires_grad = False

        module, optim, dl = self.privacy_engine.make_private(
            module=module,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="ew",
        )

        for x in dl:
            module(x)
