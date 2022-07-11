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
import random


class SILIAYN(SAC):
    """
    Diversity is All You Need
    Built on top of SAC

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param prior: The prior distribution for the skills p(z), usually uniform categorical
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param disc_on: A list of index, or a DiscriminatorFunction or 'all'. It designates which component or
        transformation of the state space you want to pass to the discriminator.
    :param combined_rewards: whether or not you want to learn the task AND learn skills, by default this is
        False in DIAYN (unsupervised method).
    :param beta: balance parameter between the true and the diayn reward, beta = 0 means only the true reward
        is considered while beta = 1 means it's only the diversity reward. Only active when combined_rewards
        is set to True. beta = "auto" is incompatible with smerl.
    :param smerl: if not None, it sets the target value for SMERL algorithm, see https://arxiv.org/pdf/2010.14484.pdf
    :param eps: if smerl is not None, it sets the margin of the reward where under esp*smerl, DIAYN reward is
        set to 0.
    :param beta_temp: only if beta='auto', sets the temperature parameter of the sigmoid for beta computation.
    :patam beta_momentum: only if beta='auto', sets the momentum parameter for beta auto update.
    """

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
        supported_action_spaces=(gym.spaces.Box),
    ):

        super(SAC, self).__init__(
            policy, env, DIAYNPolicy,
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
            supported_action_spaces=supported_action_spaces,
        )
        self.episode_buffer_size = episode_buffer_size
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        # Initialization of the discriminator
        self.discriminator_kwargs = discriminator_kwargs
        if self.discriminator_kwargs.get("net_arch") is None:
            self.discriminator_kwargs["net_arch"] = [256, 256]
        if self.discriminator_kwargs.get("arch_type") is None:
            self.discriminator_kwargs["arch_type"] = "Mlp"
        assert (
            disc_on == "all"
            or isinstance(disc_on, list)
            or isinstance(disc_on, DiscriminatorFunction)
        ), "Please pass a valid value for disc_on parameter"

        self.log_p_z = prior.logits.detach().cpu().numpy()
        self.prior = prior
        self.combined_rewards = combined_rewards
        self.beta = beta
        self.smerl = smerl
        self.eps = eps
        self.n_skills = self.prior.event_shape[0]
        self.beta_smooth = beta_smooth
        self.disc_on = disc_on
        self.max_steps = max_steps
        self.behaviour_descriptor = behaviour_descriptor
        self.metric_loggers = metric_loggers

        if self.behaviour_descriptor:
            self.disc_obs_space_shape = tuple([len(env.descriptors_names)])
        else:
            self.disc_obs_space_shape = env.observation_space.shape[0]

        if self.behaviour_descriptor:
            self.observation_space = env.observation_space["observation"]
        self.beta_momentum = beta_momentum
        self.beta_temp = beta_temp
        self.adaptive_beta = adaptive_beta
        self.diayn_reward_buffer = np.zeros(self.n_skills)
        self.mean_reward = mean_reward

        if smerl:
            assert beta != "auto", 'You must chose between SMERL and beta="auto"'

        if _init_setup_model:
            self._setup_model()
            self._sil_setup_model()

    def _setup_model(self) -> None:
        # not calling super() because we change the way policy is instantiated
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.MODEL_CHKPOINT_PERIOD = 1_000_000 # Save checkpoint model for every 1M steps
        self.MODEL_SAVE_STEP = self.MODEL_CHKPOINT_PERIOD

        # ReplayBufferZ replaces ReplayBuffer while including z
        if self.beta == "auto" or self.smerl:
            self.beta_buffer = deque(maxlen=self.episode_buffer_size)
            self.beta_buffer.append(np.zeros(self.n_skills))

        if self.disc_on == "all" or self.disc_on == ...:
            self.disc_on = ...
            self.disc_obs_shape = self.disc_obs_space_shape
        elif isinstance(self.disc_on, list):
            self.disc_obs_shape = self.disc_obs_space_shape
            # assert min(self.disc_on) >= 0 and max(self.disc_on) < self.disc_obs_shape[0]
            self.disc_obs_shape = len(self.disc_on)

        elif isinstance(self.disc_on, DiscriminatorFunction):
            self.disc_obs_shape = self.disc_on.output_size

        else:
            self.disc_obs_shape = None
        out_size = self.prior.param_shape[0] if self.prior.param_shape else 1
        self.discriminator = Discriminator(
            self.disc_obs_shape,
            out_size,
            device=self.device,
            **self.discriminator_kwargs,
        )

        self.replay_buffer = ReplayBufferZ (self.buffer_size,self.observation_space,
                                            self.action_space,self.prior,self.device,
                                            optimize_memory_usage=self.optimize_memory_usage)

        # print(self.policy_class)
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.prior,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self._create_aliases()
        self._convert_train_freq()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(
                np.float32
            )
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto') is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        if self.num_timesteps >= self.MODEL_SAVE_STEP :
            now = datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            kilo_steps = int(self.num_timesteps / 1000)
            self.save(f"autosave_siliayn_{kilo_steps}k_{current_time}")
            self.MODEL_SAVE_STEP += self.MODEL_CHKPOINT_PERIOD


        ent_coef_losses, ent_coefs = deque(maxlen=100), deque(maxlen=100)
        (actor_losses, critic_losses, disc_losses) = (
            deque(maxlen=100),
            deque(maxlen=100),
            deque(maxlen=100),
        )

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            # starting by beta
            betas = np.zeros(self.n_skills)
            for z_idx in range(self.n_skills):
                if self.combined_rewards:
                    mean_true_rewards = [ep_info.get(f"r_true_{z_idx}")
                                            for ep_info in self.ep_info_buffer]

                    mean_true_reward = safe_mean(mean_true_rewards, where=~np.isnan(mean_true_rewards))
                    if np.isnan(mean_true_reward):
                        mean_true_reward = 0.0

                    mean_diayn_reward = [ep_info.get(f"r_diayn_{z_idx}")
                                            for ep_info in self.ep_info_buffer]
                    mean_diayn_reward = safe_mean(mean_diayn_reward, where=~np.isnan(mean_diayn_reward))

                    if np.isnan(mean_diayn_reward):
                        mean_diayn_reward = 0.0

                    beta = self.beta
                    betas[z_idx] = beta


            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            rep = replay_data
            obs, zs, next_obs, true_reward, dones, actions = (rep.observations, rep.zs,    rep.next_observations,
                                                              rep.rewards,      rep.dones, rep.actions)

            ep_index = replay_data.ep_index.flatten()
            len_episodes = th.Tensor(self.len_episodes)[ep_index]
            # Get or compute vector to pass to the discriminator

            if isinstance(self.disc_on, DiscriminatorFunction):
                disc_obs = self.disc_on(replay_data.observations)
            else:
                disc_obs = replay_data.observations[:, self.disc_on]

            disc_obs = disc_obs.type(th.FloatTensor).to(self.device)
            # print(disc_obs)
            log_q_phi = self.discriminator(disc_obs)
            discriminator_loss = self.discriminator.loss(log_q_phi, zs)

            diayn_reward = log_q_phi.clone().detach() - self.log_p_z[0]

            if self.combined_rewards:
                betas = th.Tensor(betas).to(self.device) * zs
                diayn_reward = diayn_reward * betas
                if self.mean_reward:
                    rewards = (true_reward + diayn_reward.sum(dim=1, keepdim=True)
                        / len_episodes[:, None])
                else:
                    rewards = true_reward + diayn_reward.sum(dim=1, keepdim=True)

            else:
                diayn_reward = diayn_reward * zs
                rewards = diayn_reward.sum(dim=1, keepdim=True)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # We concatenate state with current one hot encoded skill
            obs = th.cat([obs, zs], dim=1)

            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.view(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                # We concatenate next state with current one hot encoded skill
                new_obs = th.cat([next_obs, zs], dim=1)
                next_actions, next_log_prob = self.actor.action_log_prob(new_obs)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(new_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.view(-1, 1)
                # td error + entropy term
                # print(rewards.shape, th.Tensor(dones).shape, next_q_values.shape)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer

            current_q_values = self.critic(obs, actions)
            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values)
                                            for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(obs, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )

            disc_losses.append(discriminator_loss.item())
            self.discriminator.optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/discriminator_loss", np.mean(disc_losses))

        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))



    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        self.len_episodes = np.zeros(total_timesteps)
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        self.ep_info_buffer = deque(maxlen=self.episode_buffer_size)
        self.ep_success_buffer = deque(maxlen=self.episode_buffer_size)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # sample skill z according to prior before generating episode
            z = self.prior.sample().to(self.device)
            z_idx = np.argmax(z.cpu())

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                z=z,
            )
            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps > 0
                    else rollout.episode_timesteps
                )

                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                self.train_sil(z_idx)

        callback.on_training_end()
        return self


    ######### SIL PART #########
    ######### SIL PART #########
    ######### SIL PART #########
    ######### SIL PART #########
    ######### SIL PART #########

    def _sil_setup_model(self) -> None:
        self.replay_buffer_skill = []
        self.running_episodes = [[] for _ in range(self.n_skills)]
        self.skill_coeff = 1e-2
        # print(self.observation_space.shape)
        self.sil_buffer_size = 4096
        self.sil_batch_size = 64
        self.one_hot_list = F.one_hot(th.arange(0, self.n_skills))
        
        now = datetime.now()
        current_time = now.strftime("%y_%m_%d_%H_%M_%S")
        self.file_name = f"skill_logger_SILIAYN_{current_time}.csv"

        file_handler = open(self.file_name, "at")
        file_handler.write("#{}\n")
        skill_logger = csv.DictWriter(file_handler, fieldnames=("timestep", "z", "prob"))
        skill_logger.writeheader()
        file_handler.flush()
        file_handler.close()

        np.set_printoptions(precision = 4)
        
        for i in range(self.n_skills):
            new_skill_replay_buffer = SilReplayBuffer (self.sil_buffer_size)
            self.replay_buffer_skill.append(new_skill_replay_buffer)

    def step_sil(self, obs, n_obs, action, reward, done, z):
        # print("SKILL: {}".format(z_idx))
        z_idx = z.argmax()
        self.running_episodes[z_idx].append([obs, n_obs, action, reward])
        if done:
            self.update_buffer(self.running_episodes[z_idx], z)
            self.running_episodes[z_idx] = []

    def update_buffer(self, trajectory, z):
        # positive_reward = False
        # for (ob, n_ob, a, r) in trajectory:
        #     if r > 0:
        #         positive_reward = True
        #         break
        # print(trajectory)
        # if positive_reward:
        self.add_episode(trajectory, z)

    def add_episode(self, trajectory, z):
        z_idx = z.argmax()
        obs, n_obs, actions, rewards, dones = [], [], [], [], []
        for (ob, n_ob, action, reward) in trajectory:
            ob_concat   = np.concatenate([ob, z[None]], axis=1)
            # print(n_ob.shape)
            # n_ob_concat = np.concatenate([n_ob, z[None]], axis=1)
            obs.append(ob_concat)
            # n_obs.append(n_ob_concat)
            actions.append(action)
            rewards.append(reward)
            dones.append(False)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.gamma)
        # print("ADD Episode:")
        # print(returns)

        file_handler = open(self.file_name, "at")
        skill_logger = csv.DictWriter(file_handler, fieldnames=("timestep", "z", "prob"))
        skill_logger.writerow( { "timestep": self.num_timesteps, "z":z_idx, "prob": list(self.prior.probs.numpy())})
        file_handler.flush()


        for (ob, action, R, done) in list(zip(obs, actions, returns, dones)):
            self.replay_buffer_skill[z_idx].add(ob, action, R)

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]


    def train_sil(self,z_idx):
        # for n in range(self.args.n_update):
        # self.replay_buffer_skill[z_idx].sample(self.batch_size)
        if( len(self.replay_buffer_skill[z_idx]) < self.sil_batch_size): return
        obs, actions, returns = self.replay_buffer_skill[z_idx].sample(self.sil_batch_size)
        num_samples = self.sil_batch_size

        mean_adv, num_valid_samples = 0, 0

        if obs is not None:
            # need to get the masks
            # get basic information of network..
            obs         = th.tensor(obs,     dtype=th.float32).reshape(self.sil_batch_size, -1).to(self.device)
            actions     = th.tensor(actions, dtype=th.float32).reshape(self.sil_batch_size, -1).to(self.device)
            returns     = th.tensor(returns, dtype=th.float32).reshape(self.sil_batch_size, -1).to(self.device)

            # n_actions_pi, log_prob    = self.actor.action_log_prob(n_obs)
            # next_critic_value = th.cat(self.critic_target(n_obs, n_actions_pi), dim=1)
            # next_critic_value, _ = th.min(critic_value, dim=1, keepdim=True)
            # critic_value = rewards + self.gamma * next_critic_value

            actions_pi, log_prob    = self.actor.action_log_prob(obs)
            critic_value = th.cat(self.critic(obs, actions_pi), dim=1)
            critic_value, _ = th.min(critic_value, dim=1, keepdim=True)

            advantages = returns - critic_value
            mask    = (advantages.cpu().detach().numpy() > 0).astype(np.float32)
            mask    = th.tensor(mask, dtype=th.float32).to(self.device)
            returns      *= mask
            critic_value *= mask

            critic_loss = F.mse_loss(returns, critic_value)
            VAL_COEFF = 0.05
            critic_loss *= 0.5 * VAL_COEFF / num_samples



            # print("Update for SIL part: SKILL {}".format(z_idx))
            # print("VALUE")
            # print(critic_loss)

            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)   # Update critic
            self.critic.optimizer.step()

            critic_value     = th.cat(self.critic.forward(obs, actions), dim=1)
            critic_value, _  = th.min(critic_value*mask, dim=1, keepdim=True)
            adv              = returns*mask - critic_value
            actor_loss       = adv.mean()

            # print("ACTOR")
            # print(actor_loss)

            self.actor.optimizer.zero_grad()
            actor_loss.backward()  # Update actor
            self.actor.optimizer.step()
            # print(returns.view(-1))
            # print(adv.view(-1))
            # print(advantages.view(-1)*mask)


            z = self.one_hot_list[z_idx]
            skill_adv = -self.prior.log_prob(z) * actor_loss.detach().cpu()
            new_prob = self.prior.probs + z * self.skill_coeff * skill_adv
            print("DIFF:")
            print(z * self.skill_coeff * skill_adv)
            # print(actor_loss.detach().cpu())
            SKILL_EPS = 0.01 * (1/self.n_skills * th.ones(self.n_skills))
            new_prob += SKILL_EPS
            new_prob = new_prob / sum(new_prob)
            self.prior = th.distributions.OneHotCategorical(probs = new_prob)
            print("Update skill probability:")
            print(self.prior.probs)

        return 



    ####### SIL PART ENDS #######
    ####### SIL PART ENDS #######
    ####### SIL PART ENDS #######
    ####### SIL PART ENDS #######
    ####### SIL PART ENDS #######



    def collect_rollouts(
        self,
        env: VecEnv,
        z: th.Tensor,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: Union[ReplayBufferZ, ReplayBufferZExternalDisc],
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturnZ:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param z: The one hot encoding of the active skill
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        diayn_episode_rewards, total_timesteps = [], []
        observed_episode_rewards = []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        self.z_idx = z.argmax()
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            done = False
            # we separe true rewards from self created diayn rewards
            true_episode_reward, episode_timesteps = 0.0, 0
            diayn_episode_reward = 0.0
            observed_episode_reward = 0.0
            while not done:

                if (self.use_sde
                    and self.sde_sample_freq > 0
                    and num_collected_steps % self.sde_sample_freq == 0
                ):
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(
                    learning_starts, z, action_noise
                )

                # Rescale and perform action
                if self.behaviour_descriptor:
                    new_obs, true_reward, done, infos = env.step(action)
                    new_obs = new_obs["observation"]
                else:
                    new_obs, true_reward, done, infos = env.step(action)
                true_reward *=10 
                done = done[0]

                # get the observation of the discriminator
                if isinstance(self.disc_on, DiscriminatorFunction):
                    disc_obs = self.disc_on(new_obs)
                else:
                    disc_obs = new_obs[:, self.disc_on]

                # compute the forward pass of the discriminator
                z_idx = np.argmax(z.cpu())
                # print(disc_obs.shape)
                log_q_phi = (self.discriminator(disc_obs)[:, z_idx].detach().cpu().numpy())

                if isinstance(self.log_p_z, th.Tensor):
                    self.log_p_z = self.log_p_z.cpu().numpy()

                # compute diversity reward
                # beta update and logging
                diayn_reward = log_q_phi - self.log_p_z[z.argmax()]
                if self.combined_rewards:
                    reward = self.beta * diayn_reward + true_reward

                else:
                    reward = diayn_reward


                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.

                if callback.on_step() is False:
                    return RolloutReturnZ(
                        0.0,
                        num_collected_steps,
                        num_collected_episodes,
                        continue_training=False,
                        z=z,
                    )

                true_episode_reward += true_reward
                diayn_episode_reward += diayn_reward
                observed_episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                for idx, info in enumerate(infos):
                    maybe_ep_info = info.get("episode")
                    if maybe_ep_info:
                        for i in range(self.n_skills):
                            maybe_ep_info[f"r_true_{i}"] = np.nan

                            maybe_ep_info[f"r_diayn_{i}"] = np.nan
                            if self.combined_rewards:
                                if self.beta == "auto" or self.smerl:
                                    maybe_ep_info[f"beta_{i}"] = betas[i]

                        maybe_ep_info[f"r_true_{z_idx}"] = true_episode_reward[0]

                        maybe_ep_info[f"r_diayn_{z_idx}"] = diayn_episode_reward[0]
                        maybe_ep_info["r"] = observed_episode_reward[0]

                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                z_store = z.clone().detach().cpu().numpy()
                # reward = true_reward
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos, z_store)
                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
                self._on_step()

                # if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                #     break

            if done:
                # print(self._episode_num)
                self.len_episodes[self._episode_num] = num_collected_steps
                num_collected_episodes += 1
                self._episode_num += 1
                diayn_episode_rewards.append(diayn_episode_reward)
                total_timesteps.append(episode_timesteps)
                print("Skill is {}".format(z.argmax()))

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()


        print(f"Episode {self._episode_num}: Returns {true_episode_reward}")

        diayn_mean_reward = (
            np.mean(diayn_episode_rewards) if num_collected_episodes > 0 else 0.0
        )
        callback.on_rollout_end()
        return RolloutReturnZ(
            diayn_mean_reward,
            num_collected_steps,
            num_collected_episodes,
            continue_training,
            z=z,
        )

    def _store_transition(
        self,
        replay_buffer: Union[ReplayBufferZ, ReplayBufferZExternalDisc],
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        z: np.ndarray,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It contains the terminal observations.
        :param z: The active skill
        """
        # Store only the unnormalized version
        if isinstance(self._last_obs, dict):
            self._last_obs = self._last_obs["observation"]
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            if self.behaviour_descriptor:
                next_obs = infos[0]["terminal_observation"]["observation"]
            else:
                next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        # print(reward_)

        replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, z, self._episode_num,)
        self.step_sil    (self._last_original_obs, next_obs, buffer_action, reward_, done, z)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_action(
        self,
        learning_starts: int,
        z: th.Tensor,
        action_noise: Optional[ActionNoise] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param z: The active skill to sample from.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        if isinstance(self._last_obs, dict):
            self._last_obs = self._last_obs["observation"]
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            obs = np.concatenate([self._last_obs, z.cpu().numpy()[None]], axis=1)
            unscaled_action, _ = self.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        exact_match=True,
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects
        )
        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList

        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            prior=data["prior"],
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        # print(data)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        model._sil_setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=exact_match, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is not None:

                    # Set the data attribute directly to avoid issue when using optimizers
                    # See https://github.com/DLR-RM/stable-baselines3/issues/391
                    recursive_setattr(
                        model, name + ".data", pytorch_variables[name].data
                    )

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:

            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )

            for i in range(self.n_skills):

                mean_diayn_reward = np.array(
                    [ep_info.get(f"r_diayn_{i}") for ep_info in self.ep_info_buffer]
                )

                mean_diayn_reward = safe_mean(
                    mean_diayn_reward, where=~np.isnan(mean_diayn_reward)
                )

                if np.isnan(mean_diayn_reward):
                    mean_diayn_reward = 0.0

                self.logger.record(
                    f"diayn/ep_diayn_reward_mean_skill_{i}", mean_diayn_reward
                )

                if self.combined_rewards:

                    if self.beta == "auto" or self.smerl:
                        beta = self.ep_info_buffer[-1].get(f"beta_{i}")
                        self.logger.record(f"train/beta_{i}", beta)

                mean_true_reward = np.array(
                    [ep_info.get(f"r_true_{i}") for ep_info in self.ep_info_buffer]
                )

                mean_true_reward = safe_mean(
                    mean_true_reward, where=~np.isnan(mean_true_reward)
                )
                if np.isnan(mean_true_reward):
                    # print("Mean reward is Nan")
                    mean_true_reward = 0.0

                self.logger.record(
                    f"diayn/ep_true_reward_mean_skill_{i}", mean_true_reward
                )

        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed",
            int(time.time() - self.start_time),
            exclude="tensorboard",
        )
        self.logger.record(
            "time/total timesteps", self.num_timesteps, exclude="tensorboard"
        )

        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success rate", safe_mean(self.ep_success_buffer)
            )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "discriminator"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables


######## Sil Data Structures ########
######## Sil Data Structures ########
######## Sil Data Structures ########

class SilReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, R):
        data = (obs_t, action, R)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, returns= [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
