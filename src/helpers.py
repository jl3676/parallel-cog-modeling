import numpy as np

def get_episodes(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the episodes from the data
    """
    if len(data.shape) == 3:
        return data[:,:,0]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,0]
        else:
            return int(data[trial_idx,0])
    else:
        raise ValueError("Data must be a 2D or 3D array")

def get_trials(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the trials from the data
    """
    if len(data.shape) == 3:
        return data[:,:,1]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,1]
        else:
            return int(data[trial_idx,1])
    else:
        raise ValueError("Data must be a 2D or 3D array")

def get_states(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the states from the data
    """
    if len(data.shape) == 3:
        return data[:,:,2]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,2]
        else:
            return int(data[trial_idx,2])
    else:
        raise ValueError("Data must be a 2D or 3D array")

def get_correct_actions(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the correct actions from the data
    """
    if len(data.shape) == 3:
        return data[:,:,3]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,3]
        else:
            return int(data[trial_idx,3])
    else:
        raise ValueError("Data must be a 2D or 3D array")

def get_actions(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the actions from the data
    """
    if len(data.shape) == 3:
        return data[:,:,4]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,4]
        else:
            return int(data[trial_idx,4])
    else:
        raise ValueError("Data must be a 2D or 3D array")

def get_rewards(data: np.ndarray, trial_idx: int=None) -> np.ndarray:
    """
    Get the rewards from the data
    """
    if len(data.shape) == 3:
        return data[:,:,5]
    elif len(data.shape) == 2:
        if trial_idx is None:
            return data[:,5]
        else:
            return int(data[trial_idx,5])
    else:
        raise ValueError("Data must be a 2D or 3D array")