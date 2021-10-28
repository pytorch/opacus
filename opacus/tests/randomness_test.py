import unittest
from typing import Optional

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderRandomnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 8
        self.data = torch.randn(self.batch_size * 10, 32)
        self.dataset = TensorDataset(self.data)

    def _read_all_data(self, dp_generator, original_generator=None):
        dl = DataLoader(
            self.dataset, batch_size=self.batch_size, generator=original_generator
        )
        dpdl = DPDataLoader.from_data_loader(dl, generator=dp_generator)
        return torch.cat([x for x, in dpdl], dim=0)

    def assertNotEqualTensors(self, a: torch.Tensor, b: torch.Tensor):
        if a.shape != b.shape:
            return
        self.assertTrue(torch.allclose(a, b))

    def assertEqualTensors(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue(torch.allclose(a, b))

    def test_no_seed(self):
        data1 = self._read_all_data(dp_generator=None)
        data2 = self._read_all_data(dp_generator=None)
        self.assertNotEqualTensors(data1, data2)

    def test_global_seed(self):
        torch.manual_seed(1337)
        data1 = self._read_all_data(dp_generator=None)
        torch.manual_seed(1337)
        data2 = self._read_all_data(dp_generator=None)
        self.assertEqualTensors(data1, data2)

    def test_custom_generator(self):
        gen = torch.Generator()
        gen.manual_seed(1337)
        data1 = self._read_all_data(dp_generator=gen)
        gen.manual_seed(1337)
        data2 = self._read_all_data(dp_generator=gen)
        self.assertEqualTensors(data1, data2)

    def test_custom_generator_with_global_seed(self):
        gen = torch.Generator()
        torch.manual_seed(1337)
        data1 = self._read_all_data(dp_generator=gen)
        torch.manual_seed(1337)
        data2 = self._read_all_data(dp_generator=gen)
        self.assertNotEqualTensors(data1, data2)

    def test_original_generator(self):
        gen = torch.Generator()
        gen.manual_seed(1337)
        data1 = self._read_all_data(dp_generator=None, original_generator=gen)
        gen.manual_seed(1337)
        data2 = self._read_all_data(dp_generator=None, original_generator=gen)
        self.assertEqualTensors(data1, data2)

    def test_custom_generator_overrides_original(self):
        dp_gen = torch.Generator()
        orig_gen = torch.Generator()
        orig_gen.manual_seed(1337)
        data1 = self._read_all_data(dp_generator=dp_gen, original_generator=orig_gen)
        orig_gen.manual_seed(1337)
        data2 = self._read_all_data(dp_generator=dp_gen, original_generator=orig_gen)
        self.assertNotEqualTensors(data1, data2)


def _epoch(model: nn.Module, optim: torch.optim.Optimizer, dl: DataLoader):
    for (x,) in dl:
        optim.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optim.step()


class OptimizerRandomnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.randn(80, 32)

    def _init_training(self, generator, noise: float = 1.0):
        dl_gen = torch.Generator()
        dl_gen.manual_seed(42)
        dl = DataLoader(TensorDataset(self.data), batch_size=8, generator=dl_gen)

        model = nn.Linear(32, 16)
        torch.nn.init.ones_(model.weight)
        torch.nn.init.zeros_(model.bias)
        model = GradSampleModule(model)

        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        dp_optim = DPOptimizer(
            optimizer=optim,
            noise_multiplier=noise,
            max_grad_norm=1.0,
            expected_batch_size=8,
            generator=generator,
        )

        return model, dp_optim, dl

    def test_no_seed(self):
        model1, optim1, dl1 = self._init_training(generator=None)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=None)
        _epoch(model2, optim2, dl2)
        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_no_noise(self):
        model1, optim1, dl1 = self._init_training(generator=None, noise=0.0)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=None, noise=0.0)
        _epoch(model2, optim2, dl2)

        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))

    def test_global_seed(self):
        model1, optim1, dl1 = self._init_training(generator=None)
        torch.manual_seed(1337)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=None)
        torch.manual_seed(1337)
        _epoch(model2, optim2, dl2)
        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))

    def test_generator(self):
        gen = torch.Generator()
        model1, optim1, dl1 = self._init_training(generator=gen)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=gen)
        _epoch(model2, optim2, dl2)
        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_generator_with_global_seed(self):
        gen = torch.Generator()
        model1, optim1, dl1 = self._init_training(generator=gen)
        torch.manual_seed(1337)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=gen)
        torch.manual_seed(1337)
        _epoch(model2, optim2, dl2)
        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_generator_seed(self):
        gen = torch.Generator()
        model1, optim1, dl1 = self._init_training(generator=gen)
        gen.manual_seed(8888)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_training(generator=gen)
        gen.manual_seed(8888)
        _epoch(model2, optim2, dl2)
        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))


class PrivacyEngineSecureModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.randn(80, 32)

    def _init_training(self, dl_generator=None):
        dl = DataLoader(TensorDataset(self.data), batch_size=8, generator=dl_generator)

        model = nn.Linear(32, 16)
        torch.nn.init.ones_(model.weight)
        torch.nn.init.zeros_(model.bias)

        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        return model, optim, dl

    def _init_dp_training(
        self,
        secure_mode: bool,
        dl_seed: Optional[int] = None,
        noise_seed: Optional[int] = None,
        noise: float = 1.0,
    ):
        dl_generator = None
        if dl_seed:
            dl_generator = torch.Generator()
            dl_generator.manual_seed(dl_seed)

        model, optim, dl = self._init_training(dl_generator=dl_generator)
        privacy_engine = PrivacyEngine(secure_mode)

        return privacy_engine.make_private(
            module=model,
            optimizer=optim,
            data_loader=dl,
            noise_multiplier=noise,
            max_grad_norm=1.0,
            noise_seed=noise_seed,
        )

    def test_basic(self):
        model1, optim1, dl1 = self._init_dp_training(secure_mode=False)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(secure_mode=False)
        _epoch(model2, optim2, dl2)

        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_raise_secure_mode(self):
        with self.assertRaises(ValueError):
            self._init_dp_training(secure_mode=True, noise_seed=42)

    def test_global_seed(self):
        model1, optim1, dl1 = self._init_dp_training(secure_mode=False)
        torch.manual_seed(1337)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(secure_mode=False)
        torch.manual_seed(1337)
        _epoch(model2, optim2, dl2)

        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))

    def test_secure_mode_global_seed(self):
        model1, optim1, dl1 = self._init_dp_training(secure_mode=True)
        torch.manual_seed(1337)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(secure_mode=True)
        torch.manual_seed(1337)
        _epoch(model2, optim2, dl2)

        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_dl_seed_with_noise(self):
        model1, optim1, dl1 = self._init_dp_training(secure_mode=False, dl_seed=96)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(secure_mode=False, dl_seed=96)
        _epoch(model2, optim2, dl2)

        self.assertFalse(torch.allclose(model1._module.weight, model2._module.weight))

    def test_dl_seed_no_noise(self):
        model1, optim1, dl1 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise=0.0
        )
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise=0.0
        )
        _epoch(model2, optim2, dl2)

        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))

    def test_seed(self):
        model1, optim1, dl1 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise_seed=17
        )
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise_seed=17
        )
        _epoch(model2, optim2, dl2)

        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))

    def test_custom_and_global_seed(self):
        model1, optim1, dl1 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise_seed=17
        )
        torch.manual_seed(1024)
        _epoch(model1, optim1, dl1)

        model2, optim2, dl2 = self._init_dp_training(
            secure_mode=False, dl_seed=96, noise_seed=17
        )
        torch.manual_seed(2048)
        _epoch(model2, optim2, dl2)

        self.assertTrue(torch.allclose(model1._module.weight, model2._module.weight))
