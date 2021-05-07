import numpy as np
import torch as th
from scipy.spatial.distance import jensenshannon as jsd
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DIAYN
import gym

def generate_trajectory(model, skill_idx, episode_length, seed=0, return_actions=True):
    states = np.zeros((episode_length, *model.observation_space.shape))
    actions = np.zeros((episode_length, *model.action_space.shape))
    env = model.env
    skill = th.zeros(model.prior.event_shape)
    skill[skill_idx] = 1

    obs = env.reset()
    states[0] = obs.flatten()
    for i in range(episode_length-1):
        obs = np.concatenate([obs,skill[None,:]],axis=1)
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        actions[i+1] = action.flatten()
        states[i+1] = obs.flatten()
    if return_actions:
        return states, actions
    else:
        return states




def compute_jsd(states_1, states_2, model, bins=50, states=True):
    if states:
        states_low = model.observation_space.low
        states_high = model.observation_space.high
    else:
        states_low = model.action_space.low
        states_high = model.action_space.high
        
    states_hist_1 = []
    states_hist_2 = []
    jsd_l = []
    for i in range(len(states_low)):
        state_low = states_low[i]
        state_high = states_high[i]
        hist, bin_edges = np.histogram(states_1.T[i],
                                       bins=bins,
                                       range=[state_low,state_high],
                                       density=True)
        states_hist_1.append(hist*np.diff(bin_edges))
        hist, bin_edges = np.histogram(states_2.T[i],
                                       bins=bins,
                                       range=[state_low,state_high],
                                       density=True)
        states_hist_2.append(hist*np.diff(bin_edges))
        jsd_l.append(jsd(states_hist_1[i],states_hist_2[i]))
    return jsd_l


def record_skills(env_id, model, directory, name_prefix, video_length=400):

    env = DummyVecEnv([lambda: gym.make(env_id)])
    model = DIAYN.load(model, env)
    prior = model.prior
    k=0
    for z in prior.enumerate_support():
        k+=1
        video_env = VecVideoRecorder(env, directory, record_video_trigger=lambda x: x == 0,
                             video_length=video_length, name_prefix=f"skill-{k}-"+name_prefix )

        obs = video_env.reset()
        for _ in range(video_length + 1):
            obs = np.concatenate([obs, z[None,:]],axis=1)
            action, next_state = model.predict(obs)
            obs, _, _, _ = video_env.step(action)
            # Save the video
        env.close()
        env = DummyVecEnv([lambda: gym.make(env_id)])
        video_env.close()
    env.close()