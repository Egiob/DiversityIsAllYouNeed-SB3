from typing import Any, Dict, List, Optional, Tuple, Type, Union
import time
from types import FunctionType as function
import gym
import sys
import numpy as np
from numpy.core.fromnumeric import mean
import torch as th
from collections import deque
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
from stable_baselines3.diayn import disc
from stable_baselines3.diayn.policies import DIAYNPolicy
from stable_baselines3.diayn.diayn import DIAYN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBufferZ, ReplayBufferZExternalDisc
from stable_baselines3.common.exp_utils import DiscriminatorFunction
from stable_baselines3.diayn.disc import Discriminator
from stable_baselines3.common.utils import get_linear_fn

class SEQDIAYN(DIAYN):
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
        optimize_memory_usage: bool = True,
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
        external_disc_shape: np.ndarray = None,
        combined_rewards: bool = False,
        beta: float = 0.01,
        smerl: int = None,
        eps: float = 0.05,
        beta_temp: float = 20.0,
        beta_momentum: float = 0.8,
        beta_smooth: bool = False,
        extra_disc_buffer: bool = True,
        extra_disc_buffer_size: int = int(1e4)
    ):
        print(learning_rate)

        super(SEQDIAYN, self).__init__(
            policy,
            env,
            prior,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            disc_on=disc_on,
            discriminator_kwargs=discriminator_kwargs,
            external_disc_shape=external_disc_shape,
            combined_rewards=combined_rewards,
            beta=beta,
            smerl=smerl,
            eps=eps,
            beta_temp=beta_temp,
            beta_momentum=beta_momentum,
            beta_smooth=beta_smooth,
            extra_disc_buffer=extra_disc_buffer,
            extra_disc_buffer_size=extra_disc_buffer_size,

        )



    def _setup_model(self) -> None:
        super(SEQDIAYN, self)._setup_model()
    
        out_size = 2
        self.discriminators = [Discriminator(
            self.disc_obs_shape, out_size, device=self.device, **self.discriminator_kwargs
        ) for i in range(self.n_skills)]
            
         

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = deque(maxlen=1000),deque(maxlen=1000)
        actor_losses, critic_losses, disc_losses = deque(maxlen=1000),deque(maxlen=1000),deque(maxlen=1000)

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
            #print("Zs :",replay_data.zs)
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


            if not self.extra_disc_buffer:
                replay_data_disc = replay_data

            else: 
                replay_data_disc = self.disc_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            if self.external_disc_shape:
                disc_obs = replay_data_disc.disc_obs
            

            else:
                # Get or compute vector to pass to the discriminator
                if isinstance(self.disc_on, DiscriminatorFunction):
                    disc_obs = self.disc_on(replay_data_disc.observations)
                else:
                    disc_obs = replay_data_disc.observations[:, self.disc_on]
            
            cur_disc = self.discriminators[self.training_skill]
            log_q_phi = cur_disc(disc_obs.to(self.device)).to(self.device)
            z = replay_data_disc.zs.to(self.device)
            c = (z.argmax(dim=1)==self.training_skill) * 1

            discriminator_loss = th.nn.NLLLoss()(log_q_phi, c)
            disc_losses.append(discriminator_loss.item())
            cur_disc.optimizer.zero_grad()
            discriminator_loss.backward()
            cur_disc.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/discriminator_loss", np.mean(disc_losses))
        self.disc_loss = np.mean(disc_losses)
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
        self.training_skill = 0
        self.learning_starts_0 = self.learning_starts
        while self.num_timesteps < total_timesteps and self.training_skill < self.n_skills:
            



            # sample skill z according to prior before generating episode
            probs = th.ones(self.training_skill+1)/(self.training_skill+1)
            probs = th.nn.functional.pad(probs, [0,self.n_skills-self.training_skill-1])
            prior = th.distributions.OneHotCategorical(probs)
            z = prior.sample().to(self.device)

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                z=z,
                disc_buffer=self.disc_buffer
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

            if self.training_skill == 0:
                objective = self.smerl * (1-self.eps/2)
            else:
                objective = self.smerl * (1-self.eps)
            mean_true_reward = [
                            ep_info.get(f"r_true_{self.training_skill}")
                            for ep_info in self.ep_info_buffer
                        ]
            mean_true_reward = safe_mean(
                            mean_true_reward, where=~np.isnan(mean_true_reward)
                        )
            if np.isnan(mean_true_reward):
                mean_true_reward = 0.0

            if mean_true_reward >= objective and self.disc_loss < 0.1:

                self.learning_starts = self.num_timesteps+self.learning_starts_0
                self.replay_buffer.reset()
                self.training_skill += 1
                



        callback.on_training_end()
        return self

   

    def collect_rollouts(
        self,
        env: VecEnv,
        z: th.Tensor,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: Union[ReplayBufferZ,ReplayBufferZExternalDisc],
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        disc_buffer = None
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



                if self.external_disc_shape:
                    disc_obs = callback.on_step()
                else:
                    if isinstance(self.disc_on, DiscriminatorFunction):
                        disc_obs = self.disc_on(new_obs)
                    else:
                        disc_obs = new_obs[:, self.disc_on]
                #print(disc_obs)

                cur_disc = self.discriminators[z.argmax().detach().cpu()]
                z_idx = np.argmax(z.cpu()).item()
                if self.training_skill == z_idx:
                    c = 1
                else:
                    c = 0
                log_q_phi = (
                    cur_disc(disc_obs)[:, 1].detach().cpu().numpy()
                )



                if isinstance(self.log_p_z, th.Tensor):
                    self.log_p_z = self.log_p_z.cpu().numpy()

                log_p_z = np.log([z_idx/(z_idx+1)+1e-10, 1/(z_idx+1)])
                diayn_reward = log_q_phi - log_p_z[1]



                # beta update and logging
                if self.combined_rewards:
                    if self.beta == "auto":
                        
                        """
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
                        """                                        




                    elif self.smerl:
                        mean_true_reward = [
                            ep_info.get(f"r_true_{z_idx}")
                            for ep_info in self.ep_info_buffer
                        ]


                        mean_true_reward = safe_mean(
                            mean_true_reward, where=~np.isnan(mean_true_reward)
                        )


                        if np.isnan(mean_true_reward):
                            mean_true_reward = 0.0

                        if self.beta_smooth :
                            a = self.smerl+np.abs(self.eps * self.smerl)
                            beta_on = self.beta * sigm(mean_true_reward*2/a - 2)
                        else:
                            beta_on = float(
                            (
                                mean_true_reward
                                >= self.smerl - np.abs(self.eps * self.smerl)
                            ) * self.beta
                        )
                        betas = self.beta_buffer[-1].copy()
                        betas[z_idx] = beta_on
                        self.beta_buffer.append(betas)
                        # add beta*diayn_reward if mean_reward is closer than espilon*smerl to smerl
                        reward =  diayn_reward * beta_on + true_reward
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
                    #print("Before",info)
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
                        #print("After",info)

                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                z_store = z.clone().detach().cpu().numpy()

                if self.external_disc_shape:
                    self._store_transition(
                        replay_buffer, buffer_action, new_obs, reward, done, infos, z_store, disc_obs
                    )

                    if disc_buffer:
                        self._store_transition(
                        disc_buffer, buffer_action, new_obs, reward, done, infos, z_store, disc_obs
                    )


                else:
                    self._store_transition(
                        replay_buffer, buffer_action, new_obs, reward, done, infos, z_store
                    )

                    if disc_buffer:
                        self._store_transition(
                        disc_buffer, buffer_action, new_obs, reward, done, infos, z_store
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
        #print(diayn_episode_rewards)
        return RolloutReturnZ(
            diayn_mean_reward,
            num_collected_steps,
            num_collected_episodes,
            continue_training,
            z=z,
        )