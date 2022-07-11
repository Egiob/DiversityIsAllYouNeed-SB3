import io
import pathlib
import sys
import csv
from datetime import datetime
import time
from collections import deque
from logging import log
from types import FunctionType as function
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.distributions.categorical import Categorical

from numpy.core.fromnumeric import mean
from numpy.lib.index_tricks import diag_indices
from scipy.special import expit as sigm
from torch.distributions import beta
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import (
    ReplayBuffer,
    ReplayBufferZ,
    ReplayBufferZExternalDisc,
    ReplayBufferZExternalDiscTraj,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.exp_utils import DiscriminatorFunction
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file,
)
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturnZ,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_linear_fn,
    polyak_update,
    safe_mean,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.siliayn import disc
from stable_baselines3.siliayn.disc import Discriminator
from stable_baselines3.siliayn.policies import DIAYNPolicy
from stable_baselines3.siliayn.siliayn import SILIAYN
import random


class SILIAYN_Discrete(SILIAYN):
    def __init__(
        self,
        policy: Union[str, Type[DIAYNPolicy]],
        env: Union[GymEnv, str],
        prior: th.distributions,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        disc_on: Union[list, str, DiscriminatorFunction] = "all",
        discriminator_kwargs: dict = {},
        combined_rewards: bool = True,
        beta: float = 1,
        smerl: int = None,
        eps: float = 0.05,
        beta_temp: float = 20.0,
        beta_momentum: float = 0.8,
        beta_smooth: bool = None,
        max_steps: int = None,
        episode_buffer_size: int = 100,
        mean_reward: bool = None,
        adaptive_beta: float = None,
        qd_grid=None,
        behaviour_descriptor=None,
        metric_loggers=None,
    ):

        super(SILIAYN_Discrete, self).__init__(
            policy, env, prior,
            learning_rate,  buffer_size,
            learning_starts, batch_size,
            tau, gamma,
            train_freq,gradient_steps,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete),
        )

        if _init_setup_model:
            self._setup_model()
            self._sil_setup_model()
