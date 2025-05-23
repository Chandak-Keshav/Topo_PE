import torch
import torch.nn as nn
from models.base_model import BaseModel

from models.layer.layers import EctLayer
from models.config import EctConfig


class EctLinearModel(BaseModel):
    def __init__(self, config: EctConfig):
        super().__init__(config)
        self.ectlayer = EctLayer(config.ectconfig)

        self.linear = nn.Sequential(
            nn.Linear(self.config.ectconfig.num_thetas * self.config.ectconfig.bump_steps, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(
            -1, self.config.ectconfig.num_thetas * self.config.ectconfig.bump_steps
        )
        x = self.linear(x)
        return x


from loaders.factory import register


def initialize():
    register("model", EctLinearModel)
