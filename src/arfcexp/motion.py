import numpy as np
from scipy import ndimage


def compute_fd(motion_regressors: np.ndarray, head_radius: float = 50.0) -> np.ndarray:
    """Compute mean framewise displacement for a given run.

    Args:
        motion_regressors: (n_samples, 6) array of motion regressors. First six columns
            should be:

            - trans_x (mm)
            - trans_y (mm)
            - trans_z (mm)
            - rot_x (deg)
            - rot_y (deg)
            - rot_z (deg)

    Returns
        fd: array of framewise displacement values, shape (n_samples,)

    See the HCP manual, p 96 for a description of the motion regressors:
    https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf

    See here for a definition of mean FD:
    https://wiki.cam.ac.uk/bmuwiki/FMRI#Framewise_Displacement
    """
    assert motion_regressors.ndim == 2 and motion_regressors.shape[1] >= 6

    motion_derivatives = np.diff(
        motion_regressors[:, :6], axis=0, prepend=motion_regressors[:1, :6]
    )
    trans_displacement = np.sum(np.abs(motion_derivatives[:, :3]), axis=1)
    rot_rad = np.deg2rad(np.abs(motion_derivatives[:, 3:]))
    assert rot_rad.min() >= 0 and rot_rad.max() <= np.pi
    rot_displacement = np.sum(head_radius * rot_rad, axis=1)
    fd = trans_displacement + rot_displacement
    return fd


def censor_motion_spikes(
    motion_values: np.ndarray,
    threshold: float = 0.2,
    segment_threshold: int = 5,
):
    """Motion spike based censoring following Li et al., NeuroImage 2019.

    - Find motion spikes > motion threshold.
    - Also censor volumes one before and two after.
    - Censor any segments of good volumes shorter than segment threshold.

    Args:
        motion_values: array of motion values, e.g. FD, shape (n_samples,)
        threshold: motion spike threshold. Volumes with motion values greater
            than this threshold will be censored.
        segment_threshold: segment length threshold. Segments of good volumes shorter
            than this threshold will be censored.

    Returns:
        sample_mask: array of good sample mask, shape (n_samples,) (True = include).

    References:
        https://www.sciencedirect.com/science/article/abs/pii/S1053811919303027
    """
    bad_mask = motion_values > threshold

    # one volume before and two after
    # see Li et al., 2019 NeuroImage
    kernel = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    bad_mask = np.convolve(bad_mask, kernel, mode="same") > 0

    sample_mask = remove_short_segments(~bad_mask, threshold=segment_threshold)
    return sample_mask


def remove_short_segments(sample_mask: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Remove short contiguous segments from a sample mask.

    Args:
        sample_mask: mask array of included samples, shape (n_samples,) (True = include).
        threshold: segments shorter than this threshold will be removed.

    Returns:
        keep_mask: sample mask after filtering out short segments.
    """
    assert threshold > 1, f"Expected segment length threshold > 1; got {threshold}."

    # Assign each contiguous segment an integer label 1, ..., n_segments.
    # Background is assigned 0.
    segment_label, _ = ndimage.label(sample_mask)

    # Size of each segment, including background.
    counts = np.bincount(segment_label)
    # Drop background count.
    counts[0] = 0
    # Find IDs of long enough segments.
    (keep_segment_ids,) = (counts >= threshold).nonzero()

    keep_mask = np.isin(segment_label, keep_segment_ids)
    return keep_mask
