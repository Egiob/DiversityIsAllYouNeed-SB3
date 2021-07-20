import numpy as np
import torch



#Crawler
def crawler_disc_on_pos_rel_legs(obs):
    """
    Return only the position of the legs relative to the body.
    """
    if isinstance(obs, np.ndarray):
        disc_obs = np.zeros((len(obs),12))
    elif isinstance(obs, torch.Tensor):
        disc_obs = torch.zeros((len(obs),12))
    for i in range(4):
        idx_start = 3+6*i
        idx_end = 6+6*i
        disc_obs[:,3*i:3*(i+1)] = obs[:,idx_start:idx_end] - obs[:,:3]
    return disc_obs

def crawler_sensor_outputs(obs):
    return obs[:, :128]



def crawler_forelegs_contact(obs):
    if isinstance(obs, np.ndarray):
        disc_obs = np.zeros((len(obs),4))
    elif isinstance(obs, torch.Tensor):
        disc_obs = torch.zeros((len(obs),4))

    for i in range(4):
        disc_obs[:,4-(i+1)] = obs[:,-4*(i+1)+2]
    
    return disc_obs

registry = {"crawler_relative_legs" : (crawler_disc_on_pos_rel_legs, 12),
            "crawler_sensor_outputs" : (crawler_sensor_outputs, 128),
            "crawler_forelegs_contact" : (crawler_forelegs_contact, 4)

            }
