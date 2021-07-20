from typing import Any, Dict, List, Optional, Tuple, Type, Union
import time
from types import FunctionType as function
import gym
import numpy as np
from numpy.core.fromnumeric import mean
import torch as th
from torch.nn import functional as F
import pathlib
import io
from scipy.special import expit as sigm
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file,
)

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturnZ,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    safe_mean,
    should_collect_more_steps,
    polyak_update,
    check_for_correct_spaces,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.diayn.policies import DIAYNPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBufferZ
from stable_baselines3.common.exp_utils import DiscriminatorFunction
from stable_baselines3.diayn.disc import Discriminator
from stable_baselines3.common.utils import get_linear_fn

class DIAYN(SAC):
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
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
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
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        disc_on: Union[list, str, DiscriminatorFunction] = "all",
        discriminator_kwargs: dict = {},
        combined_rewards: bool = False,
        beta: float = 0.01,
        smerl: int = None,
        eps: float = 0.05,
        beta_temp: float = 20.0,
        beta_momentum: float = 0.8,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            DIAYNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
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
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        # Initialization of the discriminator
        if discriminator_kwargs.get('net_arch') is None:
            discriminator_kwargs['net_arch'] =  [30, 30]


        assert (
            disc_on == "all"
            or isinstance(disc_on, list)
            or isinstance(disc_on, DiscriminatorFunction)
        ), "Please pass a valid value for disc_on parameter"

        if disc_on == "all":
            self.disc_on = ...
            disc_obs_shape = env.observation_space.shape[0]
        elif isinstance(disc_on, list):
            disc_obs_shape = env.observation_space.shape[0]
            assert min(disc_on) >= 0 and max(disc_on) < disc_obs_shape
            disc_obs_shape = len(disc_on)
            self.disc_on = disc_on
        elif isinstance(disc_on, DiscriminatorFunction):
            disc_obs_shape = disc_on.output_size
            self.disc_on = disc_on

        self.discriminator = Discriminator(
            disc_obs_shape, prior, device=self.device, **discriminator_kwargs
        )
        self.log_p_z = prior.logits.detach().cpu().numpy()
        self.prior = prior
        self.combined_rewards = combined_rewards
        self.beta = beta
        self.smerl = smerl
        self.eps = eps

        if self.beta == "auto" or self.smerl:
            self.beta_buffer = [np.zeros(prior.event_shape[0])]
            if self.beta == "auto":
                self.beta_momentum = beta_momentum
                self.beta_temp = beta_temp
        if smerl:
            assert beta != "auto", 'You must chose between SMERL and beta="auto"'

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # not calling super() because we change the way policy is instantiated
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # ReplayBufferZ replaces ReplayBuffer while including z
        self.replay_buffer = ReplayBufferZ(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.prior,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        print(self.policy_class)
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
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, disc_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # We concatenate state with current one hot encoded skill
            obs = th.cat([replay_data.observations, replay_data.zs], dim=1)
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)

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
                new_obs = th.cat([replay_data.next_observations, replay_data.zs], dim=1)
                next_actions, next_log_prob = self.actor.action_log_prob(new_obs)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(new_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer

            current_q_values = self.critic(obs, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(
                [
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                ]
            )
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

            # Get or compute vector to pass to the discriminator
            if isinstance(self.disc_on, DiscriminatorFunction):
                disc_obs = self.disc_on(replay_data.next_observations)
            else:
                disc_obs = replay_data.next_observations[:, self.disc_on]

            log_q_phi = self.discriminator(disc_obs.to(self.device)).to(self.device)
            z = replay_data.zs.to(self.device)
            print(th.exp(log_q_phi)[0])
            print(z.argmax(dim=1)[0])
            discriminator_loss = th.nn.NLLLoss()(log_q_phi, z.argmax(dim=1))
            print(discriminator_loss)
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

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # sample skill z according to prior before generating episode
            z = self.prior.sample().to(self.device)
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

        callback.on_training_end()
        return self

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables

    def collect_rollouts(
        self,
        env: VecEnv,
        z: th.Tensor,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBufferZ,
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
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            done = False
            # we separe true rewards from self created diayn rewards
            true_episode_reward, episode_timesteps = 0.0, 0
            diayn_episode_reward = 0.0
            observed_episode_reward = 0.0
            while not done:

                if (
                    self.use_sde
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
                new_obs, true_reward, done, infos = env.step(action)
                done = done[0]

                if isinstance(self.disc_on, DiscriminatorFunction):
                    disc_obs = self.disc_on(new_obs)
                else:
                    disc_obs = new_obs[:, self.disc_on]

                log_q_phi = (
                    self.discriminator(disc_obs)[:, z.argmax()].detach().cpu().numpy()
                )

                if isinstance(self.log_p_z, th.Tensor):
                    self.log_p_z = self.log_p_z.cpu().numpy()

                diayn_reward = log_q_phi - self.log_p_z[z.argmax()]

                z_idx = np.argmax(z.cpu())

                # beta update and logging
                if self.combined_rewards:
                    if self.beta == "auto":
                        mean_diayn_reward = [
                            ep_info.get(f"r_diayn_{z_idx}")
                            for ep_info in self.ep_info_buffer
                        ]
                        mean_diayn_reward = safe_mean(
                            mean_diayn_reward, where=~np.isnan(mean_diayn_reward)
                        )
                        mean_true_reward = [
                            ep_info.get(f"r_true_{z_idx}")
                            for ep_info in self.ep_info_buffer
                        ]
                        mean_true_reward = safe_mean(
                            mean_true_reward, where=~np.isnan(mean_true_reward)
                        )
                        if np.isnan(mean_true_reward):
                            mean_true_reward = 0.0
                        if np.isnan(mean_diayn_reward):
                            mean_diayn_reward = 0.0
                        last_beta = self.beta_buffer[-1][z_idx]
                        beta = (
                            sigm(
                                (mean_true_reward - mean_diayn_reward) / self.beta_temp
                            )
                            * (1 - self.beta_momentum)
                            + last_beta * self.beta_momentum
                        )
                        reward = beta * diayn_reward + (1 - beta) * true_reward
                        betas = self.beta_buffer[-1].copy()
                        betas[z_idx] = beta
                        self.beta_buffer.append(betas)

                    elif self.smerl:
                        mean_true_reward = [
                            ep_info.get(f"r_true_{z_idx}")
                            for ep_info in self.ep_info_buffer
                        ]

                        # print(mean_true_reward)
                        mean_true_reward = safe_mean(
                            mean_true_reward, where=~np.isnan(mean_true_reward)
                        )
                        if np.isnan(mean_true_reward):
                            mean_true_reward = 0.0

                        beta_on = float(
                            (
                                mean_true_reward
                                >= self.smerl - np.abs(self.eps * self.smerl)
                            )
                        )
                        betas = self.beta_buffer[-1].copy()
                        betas[z_idx] = beta_on
                        self.beta_buffer.append(betas)
                        # add beta*diayn_reward if mean_reward is closer than espilon*smerl to smerl
                        reward = self.beta * diayn_reward * beta_on + true_reward
                    else:
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
                    # print(info)
                    maybe_ep_info = info.get("episode")
                    if maybe_ep_info:
                        for i in range(self.prior.event_shape[0]):
                            maybe_ep_info[f"r_diayn_{i}"] = np.nan
                            maybe_ep_info[f"r_true_{i}"] = np.nan
                            if self.combined_rewards:
                                if self.beta == "auto" or self.smerl:
                                    maybe_ep_info[f"beta_{i}"] = betas[i]
                        maybe_ep_info[f"r_diayn_{z_idx}"] = diayn_episode_reward[0]
                        maybe_ep_info[f"r_true_{z_idx}"] = true_episode_reward[0]
                        maybe_ep_info["r"] = observed_episode_reward[0]
                        # print(info)

                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                z_store = z.clone().detach().cpu().numpy()
                self._store_transition(
                    replay_buffer, buffer_action, new_obs, reward, done, infos, z_store
                )

                self._update_current_progress_remaining(
                    self.num_timesteps, self._total_timesteps
                )

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(
                    train_freq, num_collected_steps, num_collected_episodes
                ):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                diayn_episode_rewards.append(diayn_episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

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
        replay_buffer: ReplayBufferZ,
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
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(
            self._last_original_obs, next_obs, buffer_action, reward_, done, z
        )

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

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

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
            for i in range(self.prior.event_shape[0]):
                mean_diayn_reward = [
                    ep_info.get(f"r_diayn_{i}") for ep_info in self.ep_info_buffer
                ]
                # print(mean_diayn_reward)
                mean_diayn_reward = safe_mean(
                    mean_diayn_reward, where=~np.isnan(mean_diayn_reward)
                )
                if np.isnan(mean_diayn_reward):
                    mean_diayn_reward = 0.0
                self.logger.record(
                    f"diayn/ep_diayn_reward_mean_skill_{i}", mean_diayn_reward
                )

                mean_true_reward = [
                    ep_info.get(f"r_true_{i}") for ep_info in self.ep_info_buffer
                ]

                mean_true_reward = safe_mean(
                    mean_true_reward, where=~np.isnan(mean_true_reward)
                )
                if np.isnan(mean_true_reward):
                    mean_true_reward = 0.0
                self.logger.record(
                    f"diayn/ep_true_reward_mean_skill_{i}", mean_true_reward
                )
                if self.combined_rewards:
                    if self.beta == "auto" or self.smerl:
                        beta = self.ep_info_buffer[-1].get(f"beta_{i}")
                        self.logger.record(f"train/beta_{i}", beta)

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
