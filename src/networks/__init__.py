# Networks package for  residual RL
from src.networks.data_aug import RandomShiftsAug, NoAug
from src.networks.q_agent import QAgent, SimpleQAgent
from src.networks.replay_buffer import NStepReplayBuffer
from src.networks.residual_networks import (
    ResidualActor,
    EnsembleCritic,
    SimpleResidualActor,
    SimpleEnsembleCritic,
)
from src.networks.vit_encoder import VitEncoder

__all__ = [
    "RandomShiftsAug",
    "NoAug",
    "QAgent",
    "SimpleQAgent",
    "NStepReplayBuffer",
    "ResidualActor",
    "EnsembleCritic",
    "SimpleResidualActor",
    "SimpleEnsembleCritic",
    "VitEncoder",
]
