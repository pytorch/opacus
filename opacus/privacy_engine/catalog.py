from opacus.privacy_engine.base import (
    PoissonDataLoaderMixin,
    PrivacyEngineFlatClippingBase,
    PrivacyEnginePerLayerClippingBase,
    RDPAccontantMixin,
    SequentialBatchDataLoaderMixin,
)


class PrivacyEngine(
    PoissonDataLoaderMixin, RDPAccontantMixin, PrivacyEngineFlatClippingBase
):
    pass


class PrivacyEngineNonPoisson(
    SequentialBatchDataLoaderMixin, RDPAccontantMixin, PrivacyEngineFlatClippingBase
):
    pass


class PrivacyEnginePerLayer(
    PoissonDataLoaderMixin, RDPAccontantMixin, PrivacyEnginePerLayerClippingBase
):
    pass


class PrivacyEnginePerLayerNonPoisson(
    SequentialBatchDataLoaderMixin, RDPAccontantMixin, PrivacyEnginePerLayerClippingBase
):
    pass
