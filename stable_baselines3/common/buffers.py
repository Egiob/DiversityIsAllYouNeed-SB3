import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces


from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    ReplayBufferSamplesZExternalDisc,
    ReplayBufferSamplesZExternalDiscTraj,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
    ReplayBufferSamplesZ,
)
from stable_baselines3.common.vec_env import VecNormalize


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(
        reward: np.ndarray, env: Optional[VecNormalize] = None
    ) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
            )

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class ReplayBufferZ(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prior: th.distributions,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZ, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.zeros((self.buffer_size, z_size[0]), dtype=np.float32)

        self.ep_index = np.zeros((self.buffer_size, self.n_envs),dtype=np.int64)
        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.zs.nbytes
                + self.ep_index.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        z: np.ndarray,
        ep_index: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.zs[self.pos] = np.array(z).copy()
        self.ep_index[self.pos] = np.array(ep_index).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.ep_index[batch_inds]
        )
        return ReplayBufferSamplesZ(*tuple(map(self.to_torch, data)))



class ReplayBufferZExternalDisc(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prior: th.distributions,
        disc_shape: np.ndarray,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZExternalDisc, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.zeros((self.buffer_size, z_size[0]), dtype=np.float32)
        self.disc_shape = disc_shape
        self.disc_obs = np.zeros((self.buffer_size, self.n_envs) + tuple(self.disc_shape),
                                 dtype=np.float32)

        self.ep_index = np.zeros((self.buffer_size, self.n_envs),dtype=np.int64)
        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.zs.nbytes
                + self.disc_obs.nbytes
                + self.ep_index.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        z: np.ndarray,
        disc_obs: np.ndarray,
        ep_index: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.zs[self.pos] = np.array(z).copy()
        self.disc_obs[self.pos] = np.array(disc_obs).copy()
        self.ep_index[self.pos] = np.array(ep_index).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamplesZExternalDisc:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.disc_obs[batch_inds, 0, :],
            self.ep_index[batch_inds]
        )
        return ReplayBufferSamplesZExternalDisc(*tuple(map(self.to_torch, data)))

    def sample_trajectories(self, batch_size, disc_only=False):
        if not self.full:
            ind_dones = np.concatenate([[-1],np.where(self.dones)[0]])
        else:
            ind_dones= np.where(self.dones)[0]

        ind_dones_idx = np.random.randint(0,len(ind_dones)-1,size=batch_size+1)
        trajs_bound = np.concatenate([ind_dones[ind_dones_idx,None],ind_dones[ind_dones_idx+1,None]],axis=1)
        l_big_traj = np.diff(ind_dones).max()
        disc_trajs = np.zeros((batch_size,l_big_traj) + self.disc_shape,dtype=np.float32)
        z_size = self.prior.event_shape[0]
        z_trajs = np.zeros((batch_size,l_big_traj,z_size),dtype=np.float32)
        if not disc_only:
            obs_trajs = np.zeros(
                (batch_size, l_big_traj) + self.obs_shape,dtype=np.float32
            )
            next_obs_trajs = np.zeros(
                (batch_size, l_big_traj) + self.obs_shape,dtype=np.float32
            )
            reward_trajs = np.zeros((batch_size, l_big_traj, 1),dtype=np.float32)

            done_trajs= np.zeros((batch_size, l_big_traj, 1), dtype=np.float32)
            action_trajs= np.zeros((batch_size, l_big_traj, self.action_dim), dtype=np.float32)

        lenghts = np.zeros(batch_size)
        for i in range(batch_size):
            
            start, end = trajs_bound[i]+1
            l = end-start
            disc_traj = np.swapaxes(self.disc_obs[start:end],0,1)
            z_traj = self.zs[start:end]
            if not disc_only:
                next_obs_traj = self.next_observations[start:end, 0, :]
                obs_traj = self.observations[start:end, 0, :]
                reward_traj = self.rewards[start:end]
                done_traj = self.dones[start:end]
                action_traj = self.actions[start:end, 0, :]

                #print(disc_traj.shape, z_traj.shape, next_o
                # s_traj.shape, obs_traj.shape, reward_traj.shape)

            disc_trajs[i] = np.pad(disc_traj,((0,0),(0,l_big_traj-l),(0,0)),constant_values=-1)
            z_trajs[i] = np.pad(z_traj,((0,l_big_traj-l),(0,0)),constant_values=-1)
            if not disc_only:
                next_obs_trajs[i] = np.pad(next_obs_traj,((0,l_big_traj-l),(0,0)),constant_values=-np.inf)
                obs_trajs[i] = np.pad(obs_traj,((0,l_big_traj-l),(0,0)),constant_values=-np.inf)
                reward_trajs[i] = np.pad(reward_traj,((0,l_big_traj-l),(0,0)),constant_values=-np.inf)
                done_trajs[i] = np.pad(done_traj,((0,l_big_traj-l),(0,0)),constant_values=-1)
                action_trajs[i] = np.pad(action_traj,((0,l_big_traj-l),(0,0)),constant_values=-np.inf)
            lenghts[i] = l

        if not disc_only:
        
            return obs_trajs, next_obs_trajs, disc_trajs, reward_trajs, z_trajs, done_trajs, action_trajs, lenghts
        
        else:
            return disc_trajs, z_trajs, lenghts

    def get_current_traj(self):
        ind_dones = np.where(self.dones)[0]
        if self.full and len(ind_dones[ind_dones<self.pos]) == 0:
            #if full and if episode starts at the end of buffer
            start_ind = ind_dones[-1]
            traj_0 = self.disc_obs[start_ind+1:self.buffer_size]
            traj_1 = self.disc_obs[:self.pos+1]
            traj = np.concatenate([traj_0,traj_1])
        else:

            if len(ind_dones)==0:
                #if first episode
                start_ind=-1
            else:
                start_ind = ind_dones[ind_dones<self.pos][-1]
            traj = self.disc_obs[start_ind+1:self.pos+1]
            
        traj = np.swapaxes(traj,0,1)

        lenght = traj.shape[1]-1

        return th.Tensor(traj[0]), lenght


class ReplayBufferZExternalDiscTraj(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        max_steps: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prior: th.distributions,
        disc_shape: np.ndarray,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZExternalDiscTraj, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
        self.max_steps = max_steps
        self.n_episodes = 0
        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.full(
            (self.buffer_size, self.max_steps, self.n_envs) + self.obs_shape, -np.inf,
            dtype=np.float32,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.full(
                (self.buffer_size, self.max_steps, self.n_envs) + self.obs_shape, -np.inf,
                dtype=np.float32,
            )
        self.actions = np.full(
            (self.buffer_size, self.max_steps, self.n_envs, self.action_dim), -np.inf, dtype=np.float32
        ) 
        self.rewards = np.full((self.buffer_size, self.max_steps, self.n_envs), -np.inf, dtype=np.float32)
        self.dones = np.full((self.buffer_size, self.max_steps, self.n_envs), -1, dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.full((self.buffer_size, self.max_steps, z_size[0]), -1, dtype=np.float32)
        self.disc_shape = disc_shape
        self.disc_obs = np.full((self.buffer_size, self.max_steps, self.n_envs) + tuple(self.disc_shape), -np.inf,
                                 dtype=np.float32)
        self.current_episode = 0
        self.lenghts = np.zeros((self.buffer_size), dtype=np.float32)
        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.zs.nbytes
                + self.disc_obs.nbytes
                + self.lenghts.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        z: np.ndarray,
        disc_obs: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference


        idx = np.unravel_index(self.pos,(self.buffer_size, self.max_steps))

        self.observations[idx] = np.array(obs).copy()


        if self.optimize_memory_usage:
            assert self.optimize_memory_usage == False, "Not Implemented"
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[idx] = np.array(next_obs).copy()

        self.actions[idx] = np.array(action).copy()
        self.rewards[idx] = np.array(reward).copy()
        self.dones[idx] = np.array(done).copy()
        self.zs[idx] = np.array(z).copy()
        self.disc_obs[idx] = np.array(disc_obs).copy()

        #current_episode = self.pos//self.max_steps
        #if self.pos%self.max_steps == 0:
        #    self.lenghts[self.current_episode]=0
        #    print("reset_episode",current_episode)

        if self.current_episode*self.max_steps == self.pos:
            self.lenghts[self.current_episode] = 0
            
    
        self.lenghts[self.current_episode] +=1
        
        if done.item():    
            if self.pos%self.max_steps:
                self.current_episode += 1
                self.pos = self.current_episode * self.max_steps - 1

        

 
        self.pos += 1

        if self.pos == self.buffer_size*self.max_steps:
            self.full = True
            self.pos = 0
            self.current_episode = 0

        

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.current_episode
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamplesZExternalDisc:
        if self.optimize_memory_usage:
            assert self.optimize_memory_usage == False, "Not implemented"
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, :, 0, :], env
            )
        data = (
            self._normalize_obs(self.observations[batch_inds, :, 0, :], env),
            self.actions[batch_inds, :, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.disc_obs[batch_inds, :,  0, :],
            self.lenghts[batch_inds],
        )
        return ReplayBufferSamplesZExternalDiscTraj(*tuple(map(self.to_torch, data)))

    def get_current_traj(self):
        if self.current_episode*self.max_steps == self.pos:
            self.lenghts[self.current_episode] = 0

        _,_,_,_,_,_,traj, lenght = self._get_samples(np.array(self.current_episode))
        return traj, lenght




class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.episode_starts, self.values, self.log_probs = (
            None,
            None,
            None,
            None,
        )
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictReplayBuffer must be used with Dict obs space only"
        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert (
            optimize_memory_usage is False
        ), "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs) + _obs_shape,
                dtype=observation_space[key].dtype,
            )
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs) + _obs_shape,
                dtype=observation_space[key].dtype,
            )
            for key, _obs_shape in self.obs_shape.items()
        }

        # only 1 env is supported
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = (
                obs_nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
            )
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs(
            {key: obs[batch_inds, 0, :] for key, obs in self.observations.items()}
        )
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, 0, :] for key, obs in self.next_observations.items()}
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds] * (1 - self.timeouts[batch_inds])
            ),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds], env)
            ),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.episode_starts, self.values, self.log_probs = (
            None,
            None,
            None,
            None,
        )
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictRolloutBufferSamples:

        return DictRolloutBufferSamples(
            observations={
                key: self.to_torch(obs[batch_inds])
                for (key, obs) in self.observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
