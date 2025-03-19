from pathlib import Path


def parse_hcp_metadata(path: Path) -> dict[str, str]:
    """Parse metadata from HCP file path."""
    sub = path.parents[3].name
    acq = path.parent.name
    if "7T" in acq:
        mod, task, mag, dir = acq.split("_")
    else:
        mod, task, dir = acq.split("_")
        mag = "3T"
    clean = "hp2000_clean" in path.name
    metadata = {
        "sub": sub,
        "mod": mod,
        "task": task,
        "mag": mag,
        "dir": dir,
        "clean": clean,
    }
    return metadata
