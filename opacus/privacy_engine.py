from torch import nn, optim
from torch.utils.data import DataLoader
from opacus.accountant import RDPAccountant
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizer import DPOptimizer
from opacus.data_loader import DPDataLoader

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class PrivacyEngine:

    def __init__(self, secure_mode=False):
        self.accountant = RDPAccountant()
        self.secure_mode = secure_mode # TODO: actually support it

    # TODO cool name
    def prepare(
            self,
            module: nn.Module,
            optimizer: optim.Optimizer,
            data_loader: DataLoader,
            noise_multiplier: float,
            max_grad_norm: float,
            batch_first: bool = True,
            loss_reduction: str = "mean",
    ):

        # TODO: DP-Specific validation

        module = self._prepare_model(module, batch_first, loss_reduction)
        data_loader = self._prepare_data_loader(data_loader)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
        )

        def accountant_hook(optim: DPOptimizer):
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate*optim.accumulated_iterations
            )
        optimizer.attach_step_hook(accountant_hook)

        return module, optimizer, data_loader

    def _prepare_model(self, module: nn.Module, batch_first: bool = True,
                       loss_reduction: str = "mean", ) -> GradSampleModule:
        if isinstance(module, GradSampleModule):
            return module
        else:
            return GradSampleModule(module, batch_first=batch_first, loss_reduction=loss_reduction)

    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            noise_multiplier: float,
            max_grad_norm: float,
            expected_batch_size: int,
            loss_reduction: str = "mean",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            # TODO: lol rename optimizer optimizer optimizer
            optimizer = optimizer.optimizer

        return DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction
        )

    def _prepare_data_loader(self, data_loader: DataLoader) -> DPDataLoader:
        if isinstance(data_loader, DPDataLoader):
            return data_loader

        return DPDataLoader.from_data_loader(data_loader)

    # TODO: default delta value?
    def get_privacy_spent(self, delta, alphas=None):
        if not alphas:
            alphas = DEFAULT_ALPHAS

        return self.accountant.get_privacy_spent(delta, alphas)
