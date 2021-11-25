import numpy as np
import torch


# Crawler
def crawler_disc_on_y_legs(obs):
    """
    Return only the position of the legs relative to the body.
    """
    if isinstance(obs, np.ndarray):
        disc_obs = np.zeros((len(obs), 4))
    elif isinstance(obs, torch.Tensor):
        disc_obs = torch.zeros((len(obs), 4))
    for i in range(4):
        idx = 8 + i * 14
        disc_obs[:, i] = obs[:, idx]
    return disc_obs


def crawler_is_flying(obs):
    """
    Return only the position of the legs relative to the body.
    """
    if isinstance(obs, np.ndarray):
        disc_obs = np.zeros((len(obs), 12))
        is_flying = np.zeros((len(obs), 1))
    elif isinstance(obs, torch.Tensor):
        disc_obs = torch.zeros((len(obs), 12))
        is_flying = torch.zeros((len(obs), 1))
    for i in range(4):
        idx_start = 3 + 6 * i
        idx_end = 6 + 6 * i
        disc_obs[:, 3 * i : 3 * (i + 1)] = obs[:, idx_start:idx_end] - obs[:, :3]

    is_flying[:, 0] = (disc_obs == 0).all(axis=1)
    return is_flying


def crawler_sensor_outputs(obs):
    return obs[:, :128]


def crawler_forelegs_contact(obs):
    if isinstance(obs, np.ndarray):
        disc_obs = np.zeros((len(obs), 4))
    elif isinstance(obs, torch.Tensor):
        disc_obs = torch.zeros((len(obs), 4))

    for i in range(4):
        disc_obs[:, 4 - (i + 1)] = obs[:, -4 * (i + 1) + 2]
    return disc_obs


"""
def discretize_xy(obs, dims, min_values, max_values, n=2):
    if isinstance(obs, np.ndarray):
        obs = obs.copy()[:, :n]
        for i in range(n):
            bins = np.linspace(min_values[i], max_values[i], dims[i])
            obs[:, i] = bins[
                np.clip(np.digitize(obs[:, i], bins, right=True), 0, dims[i] - 1)
            ]
    elif isinstance(obs, torch.Tensor):
        obs = obs.clone()[:, :n]
        for i in range(n):
            bins = torch.linspace(min_values[i], max_values[i], dims[i])
            obs[:, i] = bins[
                torch.clamp(
                    torch.bucketize(obs[:, i], bins, right=True), max=dims[i] - 1
                )
            ]
    return obs[:, :n]
"""


def discretize_xy(obs, dims, min_values, max_values, n=2):
    if isinstance(obs, np.ndarray):

        for i in range(n):
            pos = np.linspace(min_values[i], max_values[i], dims[i])
            obs[:, i] = pos[np.abs((pos[None] - obs[:, i, None])).argmin(axis=1)]

    elif isinstance(obs, torch.Tensor):
        device = obs.device
        for i in range(n):
            pos = torch.linspace(min_values[i], max_values[i], dims[i]).to(device)
            obs[:, i] = pos[torch.abs((pos[None] - obs[:, i, None])).argmin(axis=1)]
    return obs[:, :n]


behaviour_registry = {
    "crawler_y_legs": (crawler_disc_on_y_legs, 4),
    "crawler_sensor_outputs": (crawler_sensor_outputs, 128),
    "crawler_forelegs_contact": (crawler_forelegs_contact, 4),
    "crawler_is_flying": (crawler_is_flying, 1),
    "discretize_xy": (discretize_xy, 2),
}
