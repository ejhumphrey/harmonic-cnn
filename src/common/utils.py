import os


def create_directory(directory):
    """Create the output directory recursively if it doesn't already exist.

    Returns
    -------
    output_dir : str
        Expanded path, that now certainly exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
