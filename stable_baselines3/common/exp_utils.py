import numpy as np
import torch as th
from scipy.spatial.distance import jensenshannon as jsd
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DIAYN
import gym


def get_paths(env_id, n_skills, prior, train_freq, t_start, t_end, gradient_steps, disc_on, seed, ent_coef, combined_rewards, beta):
    train_freq_name = "".join([str(x)[:2] for x in train_freq])
    disc_on_name = "".join([str(x) for x in disc_on])
    env_name = env_id.split(':')[-1].split('-')[0].lower()
    run_name = f"{env_name}__skills-{n_skills}__disc-{disc_on_name}__tf-{train_freq_name}__gs-{gradient_steps}__ent-{ent_coef}__start-{t_start}__end-{t_end:.2}__s-{seed}"
    if combined_rewards:
        run_name = f"{env_name}__skills-{n_skills}__disc-{disc_on_name}__tf-{train_freq_name}__gs-{gradient_steps}__ent-{ent_coef}__beta-{beta:.2}__start-{t_start}__end-{t_end:.2}__s-{seed}"
    log_path = "./logs/"+ env_name  + '/' + combined_rewards*"combined_rew/"+ f"{n_skills}-skills/" + "__".join(run_name.split("__")[2:])
    save_path = "./models/" + env_name + '/' +combined_rewards*"combined_rew/"+ f"{n_skills}-skills/" + run_name
    video_path = "./video/" + env_name + '/' +combined_rewards*"combined_rew/"+ f"{n_skills}-skills/" + run_name
    
    return log_path, save_path, video_path

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


def record_skills(env_id, model_path, directory, name_prefix="", video_length=400):

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