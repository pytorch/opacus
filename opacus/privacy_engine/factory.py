from opacus import PrivacyEngine
from opacus.privacy_engine.catalog import (
    PrivacyEngine,
    PrivacyEngineNonPoisson,
    PrivacyEnginePerLayer,
    PrivacyEnginePerLayerNonPoisson,
)


class PrivacyEngineFactory:
    @classmethod
    def get(
        cls,
        accountant: str = "rdp",  # TODO: str or enum?
        clipping: str = "flat",  # TODO: str or enum?
        poisson_sampling: bool = True,
        secure_mode: bool = False,
    ):
        if accountant != "rdp":
            raise NotImplementedError

        if clipping == "flat":
            if poisson_sampling:
                return PrivacyEngine(secure_mode)
            else:
                return PrivacyEngineNonPoisson(secure_mode)
        elif clipping == "per_layer":
            if poisson_sampling:
                return PrivacyEnginePerLayer(secure_mode)
            else:
                return PrivacyEnginePerLayerNonPoisson(secure_mode)

        raise NotImplementedError
