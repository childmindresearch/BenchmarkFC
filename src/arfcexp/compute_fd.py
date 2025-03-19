import numpy as np


def compute_hcp_mean_fd(
    motion_regressors: np.ndarray, head_radius: float = 50.0
) -> float:
    """Compute mean framewise displacement for a given run from HCP.

    Args:
        motion_regressors: (n_samples, 12) array of motion regressors

    Returns
        mean_fd: mean framewise displacement

    See the HCP manual, p 96 for a description of the motion regressors:
    https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf

    See here for a definition of mean FD:
    https://wiki.cam.ac.uk/bmuwiki/FMRI#Framewise_Displacement
    """
    assert motion_regressors.ndim == 2 and motion_regressors.shape[1] == 12

    # trans_x (mm), trans_y (mm), trans_z (mm), rot_x (deg), rot_y (deg), rot_z (deg)
    motion_derivatives = np.diff(motion_regressors[:, :6], axis=0)
    trans_displacement = np.sum(np.abs(motion_derivatives[:, :3]), axis=1)
    rot_rad = np.deg2rad(np.abs(motion_derivatives[:, 3:]))
    assert rot_rad.min() >= 0 and rot_rad.max() <= np.pi
    rot_displacement = np.sum(head_radius * rot_rad, axis=1)
    mean_fd = np.mean(trans_displacement + rot_displacement)
    return mean_fd
