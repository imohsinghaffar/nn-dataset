"""Separate model identity for the shard-aware cached BLIP-2 experiment.

The neural architecture deliberately inherits the verified baseline unchanged;
only the paired ``blip2_cached_shard_aware`` data-access strategy differs.
"""

from ab.nn.nn.Blip2Cached import Net as BaselineNet
from ab.nn.nn.Blip2Cached import supported_hyperparameters


class Net(BaselineNet):
    pass


__all__ = ["Net", "supported_hyperparameters"]

