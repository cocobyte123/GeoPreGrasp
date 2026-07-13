"""Actor-critic used only for bounded residual actions."""

import torch.nn as nn

from algo.ppo_onestep.module import ActorCritic


class ResidualActorCritic(ActorCritic):
    """Start from an exact zero-mean residual policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_layer = next(
            layer
            for layer in reversed(self.actor_mean)
            if isinstance(layer, nn.Linear)
        )
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
